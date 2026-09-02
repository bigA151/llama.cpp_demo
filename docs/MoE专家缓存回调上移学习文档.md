# 24c39583：MoE 专家缓存回调上移学习文档

本文以提交 `24c39583d1668ba01c8cb657f27cdaa66ce4f485` 为准，说明它如何将 **MoE 专家缓存的决策时机与决策算法** 从 Vulkan 图执行过程上移到 `llama_context`。文中的“上移”是阶段性结果：当前只有 Vulkan 能真正使用这条 SSD 专家缓存路径，尚未实现可直接复用于 CUDA 等后端的完整通用缓存框架。

## 1. 要解决的问题

MoE（Mixture of Experts）层会为每个 token 选择少数几个专家。设：

- `E = n_expert`：该层总专家数；
- `K = n_expert_used`：每个 token 参与计算的专家数（top-k）；
- `H = n_hot_expert`：GPU 上为该层保留的热专家槽位数。

当完整专家权重在 SSD/GGUF 中，而显存只能容纳 `H < E` 个专家时，执行某层 MoE 前必须完成：

1. 取得 router 对每个 token 的专家排序；
2. 找出本 microbatch 的 top-k 并集；
3. 若所需专家未驻留，将其权重换入热缓存槽位；
4. 将原始专家编号转换成槽位编号；
5. 令后续专家矩阵乘法使用热缓存权重及槽位编号。

旧实现让 Vulkan 在执行计算图时自行识别路由节点、读取排序、决定淘汰对象并修改节点输入。这使缓存策略和计算图的具体节点排列、Vulkan 私有状态绑定得过紧。本提交将“选什么、换谁”的语义决策放到更了解模型结构的 `llama_context`，而把设备内存操作留给 Vulkan。

## 2. 改动全景与职责边界

| 位置 | 本提交中的职责 |
| --- | --- |
| `src/llama-graph.cpp` | 为完整 `argsort` 与 top-k 结果赋予稳定名称，提供回调锚点。 |
| `src/llama-context.cpp` | 注册 provider、识别回调目标、读回排序，并在 `prepare_cache` 中生成缓存计划。 |
| `ggml/src/ggml-backend.cpp` | scheduler 在回调节点处暂停并做中间同步，使 CPU 可安全读到路由结果。 |
| `ggml/include/ggml-vulkan.h` | 定义 Vulkan 与上层之间传递缓存计划的 C API。 |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 创建热缓存张量、加载 SSD 权重、上传映射表，并替换本层 MoE 节点的输入。 |

一次 MoE 层的控制流如下：

```text
router 打分
  -> GGML_OP_ARGSORT 生成 [E, T] 的完整排序
  -> scheduler 发现 eval callback 请求该节点
  -> 提交此前图分段并执行“中间同步”
  -> llama_context 将 argsort 读回 CPU
  -> prepare_cache 生成 loads 与 mapped_topk
  -> Vulkan 按 loads 换入 gate/up/down 权重，上传 mapped_topk
  -> Vulkan 将本层 MUL_MAT_ID 节点改接到热缓存
  -> 继续执行后续 MoE 节点
```

这里 `T = n_tokens`。router 和 `argsort` 仍在它们被分配到的 backend 上计算；CPU 仅消费已计算完成的整数排序结果，并不重新计算路由。

## 3. 计算图：为什么要使用完整 argsort

`llm_graph_context::build_moe_ffn()` 调用 `ggml_argsort_top_k()` 后得到两个相关结果：

- `selected_experts->src[0]`：完整排序，命名为 `ffn_moe_argsort-<layer>`；形状为 `[E, T]`，元素类型为 `I32`；
- `selected_experts`：完整排序的前 `K` 项视图，命名为 `ffn_moe_topk-<layer>`；形状为 `[K, T]`。

前者不仅能给出“当前要用哪些专家”，还提供从低优先级候选中选择淘汰者的依据，所以缓存规划订阅的是 `ffn_moe_argsort-<layer>`，不是 top-k 视图。提交只是补充说明性注释，命名本身已是上层和 Vulkan 协作的协议。

后续三个专家计算节点分别命名为 `ffn_moe_gate-<layer>`、`ffn_moe_up-<layer>`、`ffn_moe_down-<layer>`。Vulkan 用“名称 + `GGML_OP_MUL_MAT_ID`”定位它们，替代旧路径中对 `i - 2`、`i + 6` 等固定节点偏移的依赖。这降低了对图排列的耦合，但仍依赖命名协议，也仍会在运行时改写 `src`。

## 4. 上层入口：provider 注册与回调复用

### 4.1 注册条件

构造 `llama_context` 时遍历已启用的 backend，`llama_vk_register_ssd_expert_provider()` 仅在以下条件同时满足时注册：

- 模型架构为 `LLM_ARCH_QWEN35MOE`；
- backend 是 Vulkan；
- 模型拥有 `ssd_expert_addrs()`；
- `n_ssd_expert_cache` 位于 `[n_expert_used, n_expert]`。

任一 Vulkan backend 注册成功后，`ssd_expert_cache_enabled` 设为真。provider 中保存三类能力：`read` 从模型读取某一层/专家/权重种类的 SSD 切片，`get_tensor_info` 给出张量元数据，新增的 `prepare_cache` 则负责规划。

### 4.2 兼容用户原有 eval callback

`process_ubatch()` 原先将用户的 `cparams.cb_eval` 直接交给 scheduler；现在改为注册 `eval_callback_trampoline`。它转发到 `llama_context::eval_callback()`：

- 在 `ask == true` 阶段，先询问用户 callback 是否需要该张量，并记录其请求；同时检查当前张量是否是本缓存机制需要的 `argsort`；任一需要即返回 true；
- 在 `ask == false` 阶段，先执行 SSD 缓存操作；随后仅对先前请求过的张量再调用用户 callback。

因此缓存机制不会吞掉用户 callback。`eval_callback_user_requested` 使用张量指针集合保存“用户请求过”的状态，避免在执行阶段把未请求的节点错误回调给用户。

`llama_ssd_expert_argsort_layer()` 还严格校验节点：操作必须为 `GGML_OP_ARGSORT`，名称必须精确匹配 `ffn_moe_argsort-<非负整数>`。这比猜测相邻节点位置更稳健。

## 5. scheduler 如何在正确位置暂停

`ggml_backend_sched_compute_splits()` 对图的 backend split 逐段执行。它在构图/执行前以 `callback_eval(t, true, ...)` 询问某个节点是否需要回调；当包含该节点的分段已计算后，scheduler：

1. 调用 `ggml_backend_synchronize_intermediate(split_backend)`；
2. 调用 `callback_eval(t, false, ...)`；
3. 若回调返回 false，停止后续计算并使此次图计算失败。

于是回调运行前，该 `argsort` 的设备结果已经可读；回调结束后，本层 gate/up/down 的后续节点尚未执行，恰好允许替换它们的输入。

提交还把 scheduler 中多处“只是为了数据依赖安全”的同步改为中间同步，例如输入覆盖前、跨 backend 拷贝前、读取专家 id 前以及异步拷贝回退路径。默认实现会回退到普通 `synchronize`，所以其它 backend 不需要立即实现新接口。

## 6. 缓存计划：`llama_vk_ssd_expert_prepare_cache()`

该函数接收 `ggml_vk_ssd_expert_cache_plan`，输入包含：当前层号、完整 `argsort`、当前 `expert_map`、热槽位容量及输出数组。`expert_map[expert]` 的值为该专家驻留的槽位；`-1` 表示未加载。

算法可分为四步：

1. 扫描每个 token 的排序前 `K` 项，建立 `required[E]`，统计唯一需要的专家，并收集 map 中为 `-1` 的 `missing` 专家。
2. 若唯一需要的专家数大于 `H`，直接失败。实现不拆分 microbatch，也没有溢出回退；上游必须控制实际 microbatch 使其可容纳。
3. 对每个缺失专家，从每个 token 的排序末端向前搜索。第一个“非 required 且已驻留”的专家成为 victim；复用 victim 的槽位，向 `loads` 写入 `{ expert, victim, slot }`，并更新临时 `next_map`。
4. 再扫描所有 token 的前 `K` 项，以 `next_map` 把原始专家 id 转成槽位号，写入 `mapped_topk[token * K + topk]`。

注意这里的 `next_map` 是规划期副本，避免 provider 在规划尚未完整成功时修改 Vulkan 实际 map。规划函数逐项检查层号、容量、指针、整数范围、排序中的专家 id、槽位和输出缓冲区大小；任一失败均返回 false。

该淘汰策略并非 LRU/LFU：它依赖当前 batch 的完整 router 排名，优先从低排名候选中寻找不再被本 batch 使用的常驻专家。这是本阶段保留的简单策略，而非通用策略框架。

## 7. Vulkan 如何应用计划

`ggml_backend_vk_apply_ssd_expert_cache()` 是上层在读回 argsort 后调用的 Vulkan API。其顺序为：

1. 校验 backend、图指针、层号、token 数以及 `offload_buffer_manager` 的初始化状态；
2. 通过名称和 `MUL_MAT_ID` 操作定位本层 gate/up/down 节点；
3. 调用 `hot_expert_offload_process()`，后者把 argsort 放入内部缓冲，调用 provider 的 `prepare_cache`，再执行每条 load；
4. 对每个 load，从 provider 的 `read` 取得 gate、up、down 三个专家切片，并写入同一 `slot` 对应的热张量位置；成功后更新 Vulkan 自己维护的 `expert_map`；
5. 根据 token 数调整 `mapped_topk_tensor` 为 `[K, T]`，上传规划生成的 slot id；
6. 将三个节点的 `src[0]` 指向该层的热 gate/up/down 张量，将 `src[2]` 指向 `mapped_topk_tensor`。

这解释了为何 `mapped_topk` 必不可少：原 `selected_experts` 包含 `[0, E)` 的专家 id，但热权重张量的专家维度只有 `H`。替换 `src[0]` 后，`MUL_MAT_ID` 必须收到 `[0, H)` 的槽位 id，否则会索引错误权重。

热张量在 `offload_expert_buffer_manager_init()` 中按层创建。它从 expert 0 到 `H - 1` 预加载每层的 gate/up/down 权重，并设置初始 `expert_map[expert] = expert`；其余专家为 `EXPERT_NOT_LOADED`。因此首个 microbatch 也可能需要换入非初始集合的专家。

## 8. 同步语义（简述）

为避免每层回调都触发昂贵的 Vulkan 图资源清理，backend 接口增加可选的 `synchronize_intermediate`。Vulkan 的实现调用同一同步主体完成提交、fence 等待和异步输出复制，但传入 `cleanup = false`；普通最终同步仍传入 `cleanup = true`。

`cleanup_pending` 只在实际创建图上下文后置位，`ggml_vk_graph_cleanup()` 完成后清零。这样 callback 边界可安全读 CPU 数据，而图执行临时资源只在整张图/ubatch 完成时统一清理。该提交的重点是区分这两种生命周期，不改变 MoE 数值计算本身。

## 9. 错误传播与可观察性

`eval_callback()` 对 argsort 的类型、形状（`[E, T, 1, 1]`）、字节数、层范围及 API 返回值逐项检查。失败时置 `ssd_expert_cache_failed` 并返回 false；`graph_compute()` 会在 scheduler 成功返回后再次检查该标志，把整体状态改为 `GGML_STATUS_FAILED`，避免缓存失败被掩盖。

提交还加入由环境变量 `LLAMA_SSD_EXPERT_TIMING=1` 开启的临时诊断。它记录首次 prefill/decode 的缓存规划、专家加载、映射上传及 Vulkan 同步分项耗时。它不是缓存正确性的组成部分，阅读主路径时可先忽略。

## 10. 当前边界与后续方向

当前已上移的是路由解释与缓存计划；仍未上移的包括 Vulkan 中的 `expert_map` 实体状态、热缓存张量、SSD->GPU 传输、`mapped_topk` 上传，以及对 `src[0]`/`src[2]` 的运行时修改。因此“支持更多后端”在此提交中是架构方向，而不是既有功能。

要真正后端无关，后续至少需要：在上层持有通用缓存状态；抽象“将专家切片写入设备槽位”的 backend 接口；在构图期直接建立可更新的专家权重/索引输入，最终移除 Vulkan 根据节点名称修改图结构的职责。

## 11. 推荐阅读顺序

1. 先看 `src/llama-graph.cpp` 的 `build_moe_ffn()`，理解 argsort、top-k、权重和 `MUL_MAT_ID` 的数据关系；
2. 再看 `src/llama-context.cpp` 的 `llama_ssd_expert_argsort_layer()`、`eval_callback()` 和 `llama_vk_ssd_expert_prepare_cache()`；
3. 阅读 `ggml/src/ggml-backend.cpp` 中 scheduler callback 的 split 执行点，理解为何可在中途暂停；
4. 最后追踪 `ggml_backend_vk_apply_ssd_expert_cache()` 与 `hot_expert_offload_process()`，验证计划如何落到 GPU 张量和计算图。

按此顺序阅读，可以把它理解为一个“模型层制定计划、设备层落地计划”的过渡架构，而不是把整套 SSD 专家缓存已经抽象成通用 backend 服务。
