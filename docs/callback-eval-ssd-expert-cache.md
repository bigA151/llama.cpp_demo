# `callback_eval(t, false, user_data)` 中的 Vulkan SSD Expert Cache 同步链路

本文说明在 Vulkan 后端启用 SSD expert offload 时，`callback_eval(t, false, user_data)` 为什么会形成一条包含多次同步的执行路径：

~~~
callback_eval(t, false, user_data)
  ├─ ggml_backend_tensor_get(argsort)       ← GPU → CPU，等待并读取
  ├─ ggml_backend_vk_apply_ssd_expert_cache(...)
  │     ├─ hot_expert_offload_process()     ← SSD 读取并上传 expert，逐片同步
  │     └─ ggml_vk_synchronize()            ← 等待 mapped_topk 上传完成
  └─ cparams.cb_eval(t, false, ...)          ← 用户回调，可能再次同步
~~~

本文基于以下实现文件说明当前行为：

- `src/llama-context.cpp`
- `ggml/src/ggml-backend.cpp`
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/include/ggml-vulkan.h`

## 1. 这条路径解决什么问题

MoE 模型的每个 token 会由 router 选出若干 expert。启用 SSD expert offload 后，并不是所有 expert 的权重都常驻 GPU：

- `n_expert`：该层的 expert 总数；
- `n_expert_used`：每个 token 实际参与计算的 top-k expert 数量；
- `n_hot_expert`：该层允许同时驻留在 GPU cache 中的 expert 数量；
- SSD：保存完整 expert 权重的持久化存储。

GPU 上的 MoE 计算需要使用当前 microbatch 所选中的 expert。如果选中的 expert 不在 GPU cache 中，运行时必须完成以下工作：

1. 从 GPU 取回 router 产生的 expert 排序结果；
2. 判断哪些 expert 已经在 cache 中，哪些 expert 缺失；
3. 为缺失 expert 选择可淘汰的 resident expert；
4. 从 SSD 读取缺失 expert 的 `gate`、`up`、`down` 三组权重并写入 GPU slot；
5. 把逻辑 expert id 转换成 GPU cache slot id；
6. 修改当前计算图，使后续 `MUL_MAT_ID` 使用新的 hot expert tensor 和 slot 映射。

这些操作发生在当前层的 MoE 计算继续之前，所以必须完成后才能安全地执行该层的 expert matmul。

## 2. `ask=true` 与 `ask=false` 的关系

`ggml_backend_sched_eval_callback` 的定义位于 `ggml/include/ggml-backend.h`：

~~~
typedef bool (*ggml_backend_sched_eval_callback)(
        struct ggml_tensor * t, bool ask, void * user_data);
~~~

调度器对图节点通常经历两个阶段。

### 2.1 `ask=true`：询问是否需要观察节点

在 `ggml_backend_sched_compute_splits()` 中，调度器遍历当前 split 的节点：

~~~
bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);
~~~

此时回调回答的是：这个节点是否需要在一个可观察点上完成计算。

`llama_context::eval_callback()` 在 `ask=true` 时会：

1. 调用用户提供的 `cparams.cb_eval(t, true, ...)`，记录用户是否请求观察该 tensor；
2. 检查该 tensor 是否是目标层的 MoE argsort tensor，并且是否需要执行 SSD expert cache 管理。

SSD cache 路径只关心名字形如 `ffn_moe_argsort-<layer>`、操作类型为 `GGML_OP_ARGSORT` 且位于 Vulkan backend 的 tensor。满足条件时，回调返回 `true`，迫使调度器在该节点处切分计算范围。

### 2.2 图计算与中间同步

调度器确认节点需要观察后，会先计算一段 graph view：

~~~
struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);
ggml_backend_graph_compute_async(split_backend, &gv);
ggml_backend_synchronize_intermediate(split_backend);
~~~

因此，`callback_eval(t, false, ...)` 被调用时，前面这段图已经提交，并且已经执行了 intermediate synchronize。对 Vulkan 来说，这会提交尚未提交的命令并等待 fence，但不做完整 graph cleanup。

之后调度器才调用：

~~~
const bool keep_going = sched->callback_eval(
        t, false, sched->callback_eval_user_data);
~~~

所以 `false` 不是“异步通知”，而是一个位于 GPU 图计算之后的同步观察点。

## 3. 回调入口：`eval_callback_trampoline()`

`llama_context` 把调度器回调设置为 trampoline：

~~~
bool llama_context::eval_callback_trampoline(
        ggml_tensor * tensor, bool ask, void * user_data) {
    llama_context * ctx = static_cast<llama_context *>(user_data);
    return ctx != nullptr && ctx->eval_callback(tensor, ask);
}
~~~

所以用户看到的调用实际是：

~~~
eval_callback_trampoline(t, false, user_data)
  └─ llama_context::eval_callback(t, false)
~~~

## 4. 第一个同步点：读取 GPU 上的 argsort tensor

在 `src/llama-context.cpp` 中，SSD cache 路径会验证 argsort tensor 的形状和大小，然后执行：

~~~
ssd_expert_argsort.resize(n_elements);
ggml_backend_tensor_get(tensor, ssd_expert_argsort.data(), 0, n_bytes);
~~~

当前实现要求 argsort tensor 满足：

~~~
type  = GGML_TYPE_I32
shape = [n_expert, n_tokens, 1, 1]
bytes = n_tokens * n_expert * sizeof(int32_t)
~~~

对每个 token，argsort 数组包含全部 `n_expert` 个 expert 的排序结果。前 `n_expert_used` 个元素是实际 top-k expert，后面的元素会作为 cache eviction 的候选。

### 4.1 为什么是 GPU → CPU 同步

`ggml_backend_tensor_get()` 通过 tensor 所属 buffer 的 `get_tensor` 接口读取数据。Vulkan buffer 的实现最终调用：

~~~
ggml_vk_buffer_read(buf, offset, data, size);
~~~

在非 UMA 设备上，`ggml_vk_buffer_read()` 会：

1. 创建临时 transfer context；
2. 录制 GPU buffer 到 staging buffer 的 copy；
3. 提交 Vulkan command buffer；
4. 等待 fence；
5. 把 staging buffer 中的数据复制到 CPU 目标地址。

因此该调用不能在 GPU copy 尚未完成时返回。即使设备是 UMA 并可以直接从 host-visible memory 读取，前面的 scheduler intermediate synchronize 仍然保证 argsort 的生产计算已经完成。

这一步的同步对象是 router 的结果：CPU 后续的 cache planner 必须先看到完整的 expert 排序，才能决定加载和淘汰哪些 expert。

## 5. `ggml_backend_vk_apply_ssd_expert_cache()` 的职责

拿到 CPU 上的 argsort 后，`eval_callback()` 调用：

~~~
ggml_backend_vk_apply_ssd_expert_cache(
        cache_backend, eval_graph, layer,
        ssd_expert_argsort.data(), n_tokens);
~~~

该函数负责把 CPU 侧的路由结果转换为 GPU 可执行的 expert cache 状态。

### 5.1 找到当前层的三个 MoE 计算节点

函数根据 layer 构造以下节点名称：

~~~
ffn_moe_gate-<layer>
ffn_moe_up-<layer>
ffn_moe_down-<layer>
~~~

然后在 `eval_graph` 中找到三个 `GGML_OP_MUL_MAT_ID` 节点。如果任何一个节点缺失，cache 更新失败，因为只替换其中一部分权重会破坏该层 MoE 计算的一致性。

### 5.2 保存完整 argsort

argsort 被复制到 Vulkan offload manager 的 `argsort_buffer`：

~~~
om_519.argsort_buffer.assign(argsort, argsort + argsort_elements);
~~~

随后 `hot_expert_offload_process()` 使用这个 CPU 数组进行 cache planning。

## 6. `hot_expert_offload_process()`：SSD 读取与 cache 更新

这个函数的目标是：根据当前 token 的 top-k expert 集合，更新该 layer 的 GPU resident expert 映射。

### 6.1 cache planner 先计算 required set

`llama_vk_ssd_expert_prepare_cache()` 遍历每个 token 的前 `n_expert_used` 个 argsort 项：

~~~
const int expert = plan->argsort[token_offset + topk];
~~~

它会建立：

- `required`：当前 microbatch 必须可用的逻辑 expert 集合；
- `missing`：当前需要但尚未映射到 GPU slot 的 expert；
- `expert_map`：逻辑 expert id 到 GPU cache slot 的映射。

如果当前 microbatch 需要的唯一 expert 数量超过 `n_hot_expert`，planner 会失败，因为 GPU cache 无法同时容纳这些 expert。

### 6.2 选择 victim 并生成 load plan

对每个 missing expert，planner 会从 argsort 的较低优先级区域搜索可淘汰的 resident expert：

~~~
rank = n_expert - 1 ... n_expert_used
~~~

候选必须满足：

- 不是当前 required 集合中的 expert；
- 当前仍然驻留在 GPU cache；
- 对应的 slot 有效。

每一次替换会生成一个 `ggml_vk_ssd_expert_cache_load`：

~~~
expert = 要加载的逻辑 expert
victim = 要淘汰的逻辑 expert
slot   = victim 当前占用的 GPU slot
~~~

planner 同时在 `mapped_topk` 中写入最终的 slot id。这个数组不再保存逻辑 expert id，而是保存 GPU cache slot id；后续 `MUL_MAT_ID` 使用的正是这个 slot 映射。

### 6.3 每个缺失 expert 会触发 SSD 读取

`hot_expert_offload_process()` 遍历 `plan.loads`。每个 load 都需要加载该 expert 的三类权重：

~~~
gate
up
down
~~~

`offload_expert_load_slice()` 的行为是：

1. 分配一个 CPU 临时 buffer；
2. 调用 `provider.read(...)` 从 SSD expert provider 读取指定 layer、expert 和 kind 的权重；
3. 调用 `ggml_backend_vk_set_tensor_async()` 将 CPU buffer 写入对应 GPU cache slot；
4. 立即调用 `ggml_vk_synchronize(ctx)`。

当前 llama.cpp 注册的 provider 中，`provider.read` 对应 `llama_vk_ssd_expert_read()`，最终通过模型的 `ssd_expert_data()` 读取 SSD-backed expert 数据。因此，SSD 读取是同步发生在当前执行线程中的。

虽然 GPU 上传函数名包含 `async`，但它后面的 `ggml_vk_synchronize()` 会等待 transfer/compute queue 完成并等待 fence。因此一份 expert 的三个 slice 都是“提交上传后立即等待”。如果本次 planner 产生 `N` 个 load，最坏情况下会有约 `3N` 次 slice 级别的 SSD 读取和 GPU 上传等待。

每个 load 成功后，`expert_map` 被更新：

~~~
expert_map[load.expert] = load.slot;
expert_map[load.victim] = EXPERT_NOT_LOADED;
~~~

这份 map 是按 layer 保存的，所以后续相同 layer 的 microbatch 可以复用已经加载的 hot expert。

## 7. 第二个同步点：上传 `mapped_topk`

cache planner 完成后，`ggml_backend_vk_apply_ssd_expert_cache()` 调整 `mapped_topk_tensor` 的大小，并将 CPU 侧的 `topk_buffer` 上传到 Vulkan buffer：

~~~
ggml_backend_vk_set_tensor_async(
        backend,
        om_519.mapped_topk_tensor,
        om_519.topk_buffer.data(),
        0,
        ggml_nbytes(om_519.mapped_topk_tensor));
ggml_vk_synchronize(ctx);
~~~

`mapped_topk` 的形状为：

~~~
[n_expert_used, n_tokens, 1, 1]
~~~

其每个元素是 GPU cache slot，而不是原始逻辑 expert id。同步的必要性在于，紧接着函数会把该 tensor 绑定到当前图的三个节点：

~~~
cgraph->nodes[gate_idx]->src[2] = om_519.mapped_topk_tensor;
cgraph->nodes[up_idx]->src[2]   = om_519.mapped_topk_tensor;
cgraph->nodes[down_idx]->src[2] = om_519.mapped_topk_tensor;
~~~

如果上传尚未完成，后续 `MUL_MAT_ID` 可能读取旧的 slot 映射或未完成写入的数据。这个 `ggml_vk_synchronize()` 因此是 host → GPU 更新与后续 MoE 计算之间的明确执行屏障。

## 8. 最后的用户回调

SSD cache 更新成功后，`eval_callback()` 会检查该 tensor 是否在 `ask=true` 阶段被用户请求观察：

~~~
if (eval_callback_user_requested.erase(tensor) > 0 &&
        cparams.cb_eval != nullptr) {
    keep_going = cparams.cb_eval(
            tensor, false, cparams.cb_eval_user_data) && keep_going;
}
~~~

因此，用户回调并不是无条件调用：

- 如果用户在 `ask=true` 时返回 `true`，这里才会调用 `cparams.cb_eval(t, false, ...)`；
- 如果只启用了 SSD expert cache，且用户没有请求观察，SSD cache 逻辑仍会执行，但用户回调可能不会执行；
- 用户回调返回 `false` 会取消后续 graph compute，因为 `keep_going` 会被置为 `false`。

用户回调本身可能读取 tensor、打印调试信息、统计数据或执行额外同步。因此从端到端看，SSD cache 的同步完成后仍可能追加用户回调带来的同步开销。

## 9. 完整时序

以一个需要 SSD expert cache 更新的 argsort 节点为例，实际时序可以写成：

~~~
CPU: scheduler callback(t, true)
  ├─ 用户 callback(t, true)
  ├─ 识别 ffn_moe_argsort-<layer>
  └─ 返回 need = true

GPU: 计算包含 argsort 的 graph view
  └─ scheduler 调用 ggml_backend_synchronize_intermediate()

CPU: scheduler callback(t, false)
  ├─ ggml_backend_tensor_get()
  │    └─ 读取 GPU argsort，必要时提交 copy 并等待 fence
  ├─ ggml_backend_vk_apply_ssd_expert_cache()
  │    ├─ provider.prepare_cache()
  │    ├─ 对每个 missing expert：
  │    │    ├─ SSD read(gate) → GPU upload → synchronize
  │    │    ├─ SSD read(up)   → GPU upload → synchronize
  │    │    └─ SSD read(down) → GPU upload → synchronize
  │    ├─ mapped_topk host → GPU upload
  │    └─ ggml_vk_synchronize()
  └─ 用户 callback(t, false)

GPU: 继续计算被更新过输入的 MoE gate/up/down 节点
~~~

## 10. 延迟为什么可能很高

这条路径至少包含三个潜在的串行等待来源：

| 阶段 | 等待对象 | 延迟来源 |
| --- | --- | --- |
| argsort 读取 | GPU → CPU copy 和 fence | router 结果必须先可见 |
| expert 加载 | SSD read + 每个 slice 的 GPU 上传与 fence | 缺失 expert 数量、SSD 吞吐、slice 大小 |
| mapped_topk 上传 | CPU → GPU copy 和 fence | 后续 `MUL_MAT_ID` 必须读取新 slot 映射 |
| 用户回调 | 由 `cparams.cb_eval` 决定 | 读取 tensor、日志、额外同步或其他业务逻辑 |

尤其要注意，`hot_expert_offload_process()` 当前实现不是一次性批量读取全部 missing expert 后再统一上传，而是对每个 `gate/up/down` slice 分别执行 read、upload 和 synchronize。同步次数会随 load 数量增长。

## 11. 失败和停止语义

以下情况会让 SSD cache 路径失败：

- argsort tensor 类型、形状或字节数不符合预期；
- token 数量导致 size 计算溢出；
- provider 缺少 `read`、`get_tensor_info` 或 `prepare_cache`；
- microbatch 需要的唯一 expert 超过 `n_hot_expert`；
- 没有可淘汰的 resident expert；
- SSD slice 读取失败；
- Vulkan 上传或 cache slot 校验失败；
- 当前图缺少 gate/up/down 三个节点。

失败时，`eval_callback()` 会设置 `ssd_expert_cache_failed`，并返回 `keep_going = false`。调度器收到 `false` 后停止继续计算当前 graph。

用户回调返回 `false` 也会停止 graph，但它来自 `cparams.cb_eval` 的业务逻辑，而 SSD cache 失败来自 cache 管理或数据搬运错误。

## 12. 诊断建议

当前实现提供临时的 Vulkan SSD timing 统计，可以关注：

- `plan_us`：cache planner 耗时；
- `expert_load_us`：SSD expert 加载耗时；
- `mapped_upload_us`：`mapped_topk` 上传及等待耗时；
- `total_us`：一次 layer cache 更新的总耗时；
- Vulkan synchronize 中的 transfer submit、compute submit、fence wait 和 output copy 时间。

实际排查时建议依次确认：

1. argsort GPU → CPU 读取是否占据大量时间；
2. `n_loads` 是否过高，是否由于 hot cache 太小导致频繁淘汰；
3. `expert_load_us` 主要来自 SSD 读取还是 GPU 上传等待；
4. `mapped_upload_us` 是否异常增大；
5. 用户 `cparams.cb_eval` 是否在 false 阶段执行了额外读取或同步。

## 13. 相关源码位置

- `src/llama-context.cpp`
  - `llama_context::eval_callback_trampoline()`：调度器回调入口；
  - `llama_context::eval_callback()`：处理 `ask=true/false)、读取 argsort 和调用 SSD cache；
  - `llama_vk_ssd_expert_prepare_cache()`：生成 cache load plan 和 `mapped_topk`。
- `ggml/src/ggml-backend.cpp`
  - `ggml_backend_sched_compute_splits()`：调用 `ask=true)、计算 graph view、同步后调用 `ask=false`；
  - `ggml_backend_tensor_get()`：从 tensor buffer 读取数据。
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
  - `ggml_backend_vk_buffer_get_tensor()`：Vulkan buffer 的同步读取入口；
  - `ggml_vk_buffer_read()`：执行 GPU → CPU copy 并等待 fence；
  - `hot_expert_offload_process()`：加载缺失 expert 并更新 resident map；
  - `offload_expert_load_slice()`：同步读取一个 expert slice 并上传 GPU；
  - `ggml_backend_vk_apply_ssd_expert_cache()`：上传 `mapped_topk` 并替换 gate/up/down 节点输入；
  - `ggml_vk_synchronize()`：提交队列、等待 fence 并完成 host/device copy。

