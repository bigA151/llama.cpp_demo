# Wave-Partitional Prefill：MoE 专家流式加载的原理与源码导读

> **源码版本。** 本文根据 llama.cpp [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) 的
> `feat/moe-streaming-core` 分支（提交 `1248fd8fa8cfebaece5ea992e4d951c1e18bb9d5`）编写。该 PR 在本文编写时尚未合并；下文描述的是该 PR 的实现，不代表当前 `master` 已具有此功能。

> **阅读范围。** PR #25294 包含完整的 MoE routed-expert SSD streaming，而 Wave-Partitional
> Prefill 是其中专门解决长提示词预填充的一层调度机制。本文会先交代 streaming 的整体背景，
> 但重点放在 wave 如何划分、如何复用原有 GEMM、如何保证 cache slot 生命周期，以及为什么结果
> 可以与非 streaming 路径保持一致。

## 0. 结论先行

Wave-Partitional Prefill 解决的不是「怎样让 SSD 更快」，而是一个更具体的容量与调度问题：
当一个 ubatch 实际触及的不同专家数超过设备侧 expert cache 容量时，不再缩小 ubatch，也不丢弃
任何 router 选择；而是把 expert GEMM 拆成多轮，每轮只装入一组专家，再用 mask 保留本轮结果，
最后把各轮结果相加。

它的关键性质可以概括为：

| 性质 | PR #25294 的做法 |
|---|---|
| Router 语义 | 不修改 top-k 选择和 router 权重 |
| Expert 身份 | 原始 expert id 继续用于 scale、bias 等；仅 GEMM id 改写成 cache slot |
| 容量约束 | 用多个 wave 复用有限 slot，不再要求整个 ubatch 的专家同时驻留 |
| I/O 次数 | 同一 expert 在一个 ubatch 内只进入一个 wave，因此至多按需加载一次 |
| I/O/计算重叠 | 当前 wave 计算时，best-effort 预读下一 wave |
| 输出 | 相同 kernel 与 ubatch 下，PR 作者验证可与非 streaming 路径 bit-exact |
| 当前状态 | 截至 2026-08-21，PR 仍为 open、未合并，接口和实现仍可能变化 |

## 1. 要解决什么问题

MoE 层有 `n_expert` 个专家，但每个 token 只由 router 选中 `n_expert_used` 个专家（top-k）。预填充（prefill）会一次处理一个很长的 micro-batch，因而同一层中被所有 token 选到的**不同专家总数**可能很大。

假设：

- 一层有 256 个专家；
- 每个 token 使用 top-2；
- 一个 ubatch 有 128 个 token；
- 专家 cache 只有 64 个 slot。

最坏情况下，这个 ubatch 会触及 `min(256, 128 × 2) = 256` 个不同专家。传统按需加载路径要求「本层本 ubatch 所有被选专家同时驻留」，这时 64 个 slot 放不下，只能缩小 `-ub`，或直接失败。缩小 ubatch 会增加图调度和重复 I/O 的开销，长 prompt 的 prefill 速度会明显变差。

**Wave-Partitional Prefill（分波预填充）**的思路是：不要求所有被选专家同时驻留。它把这些不同 expert 划成多组（wave），一组装入 cache、计算、释放/复用 slot，再处理下一组；最后把各组结果相加。这样 cache 是容量上限，不再是 ubatch 大小的硬上限。

### 1.1 它在完整 SSD streaming 路径中的位置

PR 不把 routed-expert 权重整体常驻 RAM/VRAM。模型加载时，普通 dense 权重仍按原有路径处理，
可 streaming 的 `ffn_*_exps` 张量则只创建每层 `n_slots` 个 expert slab 的 cache tensor；每个
expert 在 GGUF split 文件中的索引、偏移和单个 slab 字节数被记录下来。执行时的数据层级是：

```text
GGUF 文件中的 routed-expert slab
             │  cache miss：I/O worker 按文件偏移读取
             ▼
设备侧每层 n_slots 个 expert cache
             │  原始 expert id → resident slot id
             ▼
ggml_mul_mat_id() 执行 gate/up/down expert GEMM
```

Wave-Partitional Prefill 位于「router 已给出 top-k」与「expert GEMM 开始」之间。它不负责改变
模型路由，只负责决定：哪些专家在本轮驻留、GEMM 应索引哪个 slot、哪些 pair 的输出应被 mask。

## 2. 先理解三个基本概念

### 2.1 GEMM 是什么

GEMM（General Matrix Multiply）就是通用矩阵乘法。MoE expert 的 FFN 主要由这些乘法构成：

```text
token activation × expert 的 gate/up 权重
             ↓
          激活函数
             ↓
          × expert 的 down 权重
             ↓
        该 expert 的输出
```

在 ggml 图里，这些按 expert id 分组的矩阵乘法由 `ggml_mul_mat_id()` 表示。一次 MoE 结果可以抽象为：

```text
Y(token) = Σ [ router_weight(token, e) × Expert_e(token) ]
             e ∈ router 选出的 top-k
```

### 2.2 wave 是什么

wave 是「本次只让其中一部分 expert 真正参与计算」的一轮。若一个 ubatch 的不同 experts 为：

```text
{E0, E2, E3, E4, E7, E9, E12, E15}
```

而每 wave 最多承载 3 个 expert，则可划分为：

```text
wave 0: E0, E2, E3
wave 1: E4, E7, E9
wave 2: E12, E15
```

同一 expert 只属于一个 wave。因此它在这个 ubatch 中最多从 SSD 加载一次。

### 2.3 mask 是什么

为了复用既有 `ggml_mul_mat_id()` kernel，每个 wave 的图仍保留所有 token–expert pair；但只有属于当前 wave 的 pair 对最终结果有贡献。实现生成一个 0/1 mask：

```text
mask_w(token, expert) = 1  若 expert 属于 wave w
                        0  否则
```

当前 wave 的 expert GEMM 输出乘这个 mask；其他 wave 的 pair 即便完成了形式上的索引，也会被乘成零。所有 wave 相加时，每个原始 pair 恰好保留一次：

```text
sum_w [ mask_w × ExpertGEMM_w ] = 原始 MoE expert 输出
```

这解释了为什么该实现不需要 Drop expert、不改变 router 选择，也不需要用相似 expert 替代。PR 声明：在相同 kernel 与 ubatch 条件下，streaming 路径可与非 streaming 路径 bit-exact。

## 3. 总体执行流

以下是单个 MoE 层的 waved prefill 数据流：

```text
router top-k
  │  输出原始 selected_experts（逻辑 expert id）
  ▼
wave 0 的 custom op
  ├─ 收集本 ubatch 的不同 expert
  ├─ 划分 expert → wave
  ├─ 确保 wave 0 专家已装入 slot cache
  └─ 尝试异步预读 wave 1
  ▼
wave 0 的 gate/up/down GEMM
  ▼
wave 0 mask：非本 wave pair 置零
  ▼
wave 1（等 wave 0 GEMM 已消费 slot 后）
  ├─ 装入/等待 wave 1
  ├─ 尝试预读 wave 2
  ├─ GEMM + mask
  └─ 与 wave 0 的输出相加
  ▼
... 直到所有 wave 完成
  ▼
恢复原有 MoE 的 router 权重、偏置和后续图计算
```

注意「专家 id」有两种含义：

- **原始逻辑 id**：router 选出的 `E0`、`E7` 等，router weight、expert scale、bias 必须继续按它索引；
- **cache slot id**：当前 expert 权重暂存在第几个 slot，GEMM 必须按它索引。

实现只把喂给 GEMM 的 id 改成 slot id；`selected_experts` 本身保留原始 id。这是保真性的关键之一。

## 4. cache 容量与 wave 大小

在 `src/llama-graph.cpp` 的 `build_moe_ffn()` 中，代码先保守估计这个 ubatch 最多触及多少不同 expert：

```cpp
n_touch_max = min(n_expert, n_tokens * n_expert_used);
```

只有 `n_touch_max > n_slots` 才进入多 wave 路径。每个 wave 的最大不同 expert 数为：

```text
wave_capacity = floor((n_slots - n_expert_used) / 2)
```

这个公式不是任意选择的。cache 需要同时保护三类 slot：

```text
当前 wave 真正在算的专家      : wave_capacity
下一 wave 已预读、不能被淘汰  : wave_capacity
被 mask 的 pair 临时停靠位置  : n_expert_used
------------------------------------------------
总计                         : 2 × wave_capacity + n_expert_used
```

其中「停靠位置」来自 Metal kernel 的约束：同一个 token 行里，每个 top-k pair 必须能拿到不同的 resident slot，即使它在当前 wave 被 mask 为零。因此 PR 要求：

```text
n_slots >= 3 × n_expert_used
```

否则一轮连最少 top-k 个真实 expert 加上预读/停车空间都无法安全安排，代码会中止并提示增大 `--moe-stream-cache`。

图构建时按 `n_touch_max` 预留 wave 数；真实执行时 `plan_waves_locked()` 再根据实际的 unique expert 集合划分。若真实专家少于最坏情况，末尾的 wave 没有有效工作，mask 使其输出为零。

### 4.1 「建图 wave 数」与「实际 wave 数」

这里有一个容易混淆的细节：ggml 图是预先构建的，而 router 的真实 top-k 结果要到执行时才知道。
因此源码存在两套数量：

- `n_stream_waves`：建图阶段按 `n_touch_max` 计算的保守上限；决定图中创建多少套
  `ids → GEMM → mask → add` 节点；
- `plan_n_waves`：wave 0 执行时，`plan_waves_locked()` 根据实际 `uniq.size()` 得到的真实数量。

如果图为最坏情况准备了 6 个 wave，而实际只需要 3 个，后 3 个 wave 仍在图中，但没有所属
expert；它们使用合法的停车 slot 完成形式上的 GEMM，mask 全为 0，所以不会改变最终结果。
这种设计以少量冗余节点换取图结构稳定，避免根据每次 router 输出重建整张图。

### 4.2 何时不走多 wave

若保守上界 `n_touch_max <= n_slots`，源码只插入一次 `llama_moe_stream_remap()`：它收集
实际 unique expert、装入 miss、等待全部变为 `RESIDENT`，再将每个原始 id 映射为 slot id。
因此 decode 或较小 ubatch 通常走单次 remap 路径；Wave-Partitional Prefill 只在建图时判断
cache 可能容纳不下时启用。

## 5. 关键源码逐段阅读

### 5.1 建图入口：`build_moe_ffn()`

文件：`src/llama-graph.cpp`

它先确认当前层的 `gate_exps/up_exps/down_exps` 的确是 streaming cache tensor，然后决定使用普通 remap 或 wave 路径。

- cache 能容纳整个 ubatch 的 unique experts：调用 `llama_moe_stream_remap()`，一次性将原始 expert id 改写成 slot id；
- cache 放不下：为每个 wave 创建 `ids_w`、执行原有 expert GEMM 管线、生成 `mask_w`、将 `e_w` 累加到 `experts`。

核心结构可简化为：

```cpp
for (uint32_t w = 0; w < n_stream_waves; ++w) {
    ids_w  = custom_op(llama_moe_stream_wave_ids, ...);
    e_w    = build_expert_gemms(cur, ids_w);
    mask_w = custom_op(llama_moe_stream_wave_mask, ...);
    e_w    = ggml_mul(ctx0, e_w, mask_w);
    experts = experts ? ggml_add(ctx0, experts, e_w) : e_w;
}
```

`build_expert_gemms()` 是从原有 gate/up/down 的矩阵乘法流程抽出的 lambda。它使用 `ids_gemm` 查 streaming slot，但 `up_exps_s`、`gate_exps_s`、`down_exps_s` 等按 expert 配置的 scale 仍使用原始 `selected_experts`。这样不会把「slot 的位置」误当成「模型中的 expert 身份」。

### 5.2 wave 规划：`plan_waves_locked()`

文件：`src/llama-moe-stream.cpp`

第 0 个 wave 的 `llama_moe_stream_wave_ids()` 会调用它。函数遍历 router 的 top-k id：

1. 用 `touched` 去重；
2. 把第一次遇到某 expert 的顺序记到 `sl.uniq`；
3. 令 `expert_wave[expert] = index / plan_capacity`；
4. 计算实际 `plan_n_waves`。

同一 expert 即使被 ubatch 中很多 token 选择，也只会在 `uniq` 出现一次，所以一次 ubatch 对它只需要一次 disk-to-slot 加载。

### 5.3 装入当前 wave、预读下一 wave：`stage_wave_locked()`

这个函数是数据搬运策略的核心。

**当前 wave：** 对当前 group 的每一个 expert，先查 `expert_slot`。命中则复用；miss 则用 `pick_victim_locked()` 找空 slot 或可淘汰 slot，再通过 `reserve_slot_locked()` 将 slot 标为 `LOADING`，并把工作项压入 `q_demand`。必要时等待 `cv_done`，直到当前 wave 所有 `demand_slots` 变成 `RESIDENT`。

**下一 wave：** 当前 wave 已可执行后，函数遍历下一 group，若还有可用 slot 就把读取任务压入同一个队列；这部分不等待。GPU 开始执行当前 wave GEMM 时，I/O worker 可以同时读取下一 wave。这就是这里的 read-ahead/预读。

预读是 best-effort：没有可用 victim slot 就跳过，下一 wave 再按 demand miss 正常加载。因此预读失败只影响速度，不影响输出正确性。

### 5.4 为什么 wave 必须按顺序：ordering token

下一 wave 不能在前一 wave 的 GEMM 尚未读完 slot 时重写这些 slot。建图时从第二个 wave 起，`ids_w` 的第二个输入是：

```cpp
ggml_view_1d(ctx0, experts, 1, 0)
```

这只是前一 wave 输出的一个元素视图，不是为了传递数值，而是一个很小的**执行顺序 token**。它让图依赖保证：先完成前一 wave 的 GEMM，才允许下一 wave 的 custom op 替换 cache slot 内容。

### 5.5 slot id 发射：`emit_wave_slots()`

每个 token 有 `n_expert_used` 个 pair。函数两步写入 `ids_w`：

1. 若 pair 的 expert 属于当前 wave，写入该 expert 的真实 cache slot；
2. 其余 pair 已被 mask 掉，但 GEMM kernel 仍需要合法 id，因此从 `plan_pool` 分配一个已 resident 的「停车 slot」。

`pool_used` 确保同一 token 的多个 pair 不复用同一个停车 slot。停车 expert 实际上不会影响结果，因为对应 `mask_w` 是 0。

### 5.6 mask：`llama_moe_stream_wave_mask()`

它逐个 pair 判断：

```cpp
out[i] = (sl->expert_wave[ids[i]] == w) ? 1.0f : 0.0f;
```

随后 `ggml_mul()` 执行逐元素乘法。每个原始 expert pair 只在所属的那一 wave 获得 1；其他所有 wave 都是 0。这正是「多波结果相加仍等价于原始 MoE」的直接代码依据。

## 6. I/O 与 cache 状态

`llama_moe_stream` 有 I/O worker pool；任务队列是 `q_demand`。slot 至少经历：

```text
EMPTY → LOADING → RESIDENT
```

worker 从 GGUF 对应 expert 的文件偏移读取 gate/up/down 等 expert slab，再写入 cache tensor 的 `slot × nb_expert` 偏移。slot 被替换时 `slot_gen` 自增；过期工作项会因 generation 不匹配被跳过，避免晚到的读取覆盖新 expert。

淘汰函数 `pick_victim_locked()` 避开：

- 当前 wave 与停车池用 `keep` 标记保护的 slot；
- 正在读取的 `LOADING` slot。

其余候选优先选择空 slot；没有空位时以 `route_hotness` 最低为主、`slot_last_use` 最早为次序。`route_hotness` 会周期性减半，避免早期热 expert 永久占住 cache。

### 6.1 Worker、generation 与错误处理

I/O worker pool 在第一次 remap 时延迟启动。每个 worker 拿到 `(layer, expert, slot, gen)` 后，
会依次读取该 expert 对应的 fused `gate_up + down`，或独立的 `gate + up + down` slab，并写入
cache tensor 的目标 slot。读取期间互斥锁会释放，因此多个 worker 可以并行处理不同 slab。

`slot_gen` 是防止迟到任务污染 cache 的版本号。一个 slot 被重新预留时 generation 自增；worker
提交前若发现任务中的 `gen`、当前 occupant 或状态已不匹配，就把任务视为 stale 并跳过。任何实际
I/O 失败都会设置全局 `load_failed`，等待方随后中止执行，而不是继续使用不完整权重。

### 6.2 O_DIRECT 与 mmap

PR 提供 `--moe-stream-direct`，在支持的平台和文件系统上以 4096 字节对齐的 `O_DIRECT`
读取绕过操作系统 page cache。其目标场景是模型文件显著大于内存：这时 page cache 无法保留完整
工作集，反而可能与 dense 常驻权重争抢内存并持续抖动。

实现会先探测 direct I/O；若平台或文件系统不支持，会告警并重新以 buffered read 打开文件。
Windows 路径没有对应的 positional direct-read 实现，读取会退化为串行的 seek + read。启用
streaming 时 PR 会自动关闭 mmap，因为 mmap 预取大模型会违背「只读取被路由 expert」的目标。

### 6.3 PR 中的命令行参数

以下参数属于 PR #25294，而非当前 `master` 的稳定接口：

| 参数 | 含义 |
|---|---|
| `--moe-stream` | 启用 routed-expert 按需磁盘流式加载 |
| `--moe-stream-cache <NG>` | 以 GiB 总预算推导每层 slot 数，例如 `40G` |
| `--moe-stream-cache <Ns>` | 直接指定每层 slot 数，例如 `64s` |
| `--moe-stream-io-threads N` | 设置 I/O worker 数；实现默认 9，最大钳制为 18 |
| `--moe-stream-direct` | 尝试使用 `O_DIRECT`；不可用时回退 buffered read |

`--moe-stream-cache` 与其他 streaming 参数会隐式启用 `--moe-stream`。若未给预算或 slot，
实现采用 `clamp(2 × n_expert_used, 16, n_expert)` 的自动值。若配置的 slot 已覆盖全部 expert，
streaming 会被禁用并按普通方式完整加载；若是内存预算，则会用「所有可 streaming 层中单个
expert slab 的总字节数」换算统一的每层 slot 数。

示例命令（模型路径仅为占位）：

```console
llama-cli -m /path/to/model.gguf -ngl 99 \
  --moe-stream-cache 64s \
  --moe-stream-io-threads 9 \
  --moe-stream-direct \
  -c 4096 -p "..."
```

该路径要求能重新打开的文件型 GGUF；stdin 或裸文件描述符不受支持。streaming warmup 会被跳过，
因为 warmup 图可能让所有 expert 同时参与，超出小 cache 的设计前提。

## 7. 性能收益、代价与边界

### 收益

- 解除了「unique experts 必须小于 cache slots」对 ubatch 的限制；
- 每个触及 expert 在一个 ubatch 内至多加载一次；
- 可以把下一 wave 的读盘与当前 wave 的 GPU GEMM 重叠；
- 不改变 router 选择，不需要因 cache miss Drop 或 substitute expert。

### 代价

- 同一 MoE 层会多次执行 expert GEMM 管线，并增加 mask、add、custom op 节点；
- wave 越多，额外计算与调度越多；
- 预读需要保留一部分 slot，导致有效「当前 wave 容量」小于 `n_slots`；
- 当前 wave 的第一次冷加载仍需要等待，不能完全隐藏 SSD 延迟；
- 这是 prefill 优化；decode 每层通常只处理一个 token，主要依赖 cache hit，PR 本身没有跨层 router 预测预取。

### 正确性为什么成立

对某 token 的第 `k` 个路由 pair，记原始 expert 为 `e_k`，其所属 wave 为 `w(e_k)`。每个 wave
都计算形状兼容的输出，但只有所属 wave 的 mask 为 1：

```text
M_w(e_k) = 1,  当且仅当 w = w(e_k)
Σ_w M_w(e_k) × Expert_e_k(x) = Expert_e_k(x)
```

之后仍按原始 router 权重聚合，所以数学上没有删减或替换 expert。实现层面还需要同时满足三点：

1. `selected_experts` 始终保留逻辑 expert id；只有 `ids_gemm`/`ids_w` 改写为 slot id；
2. expert-specific scale 和 bias 继续用 `selected_experts` 索引；
3. 下一 wave 改写 slot 前，图依赖保证前一 wave GEMM 已消费完这些权重。

「数学等价」不自动意味着所有平台都逐 bit 相同。PR 的说明是：在 streaming 与非 streaming
使用相同 kernel 和 ubatch 时可 bit-exact；若 CPU 非 streaming 路径使用了不同的 repack 或
kernel，浮点末位可能有差异。

### PR 给出的基准结果

PR 作者在 GB10（Grace-Blackwell、128 GB unified memory、PCIe 4.0 SSD）上测试约 254 GB 的
GLM-5.2-UD-Q2_K_XL（约 754B 参数、256 experts），使用 `-ngl 99`、4K context、greedy，
并生成 512 tokens：

| Expert cache | Prefill（无 wave） | Prefill（wave） | Decode | Cache hit rate |
|---|---:|---:|---:|---:|
| 64 slots（约 55 GB） | 2.28 tok/s | 5.65 tok/s | 约 1.83 tok/s | 73% |
| 90 slots（约 79 GB） | 2.88 tok/s | 5.69 tok/s | 约 2.20 tok/s | 79% |

这些数据说明 wave 主要改善 prefill：它移除了为适配 cache 而缩小 ubatch 的旧瓶颈。不能把该表
直接外推到所有 SSD、量化格式或后端；实际收益取决于 expert 分布、cache 命中率、单个 slab 大小、
I/O 并行度、设备 GEMM 速度以及预读能否在当前 wave 计算结束前完成。

同一讨论中，作者还给出了 90-slot O_DIRECT streaming 与 `mmap + --cpu-moe` 的对比：prefill
为 5.58 对 1.06 tok/s，decode 为 2.08 对 0.87 tok/s。这里对比的是特定硬件上两条完整执行
路径，不应被理解为对 mmap 本身的孤立微基准。

### 已知限制与不适用场景

- **PR 尚未合并。** CLI、内部结构、支持后端和正确性保证仍可能在评审中变化；
- **同一 streamed model 仅支持单 context。** 多个 `llama_context` 并发共享 cache 时，一个
  context 可能淘汰另一个 in-flight 图仍在使用的 slot，导致错误输出；单 context 内的
  `--parallel N` 可安全使用，因为序列在同一张图中统一预留；
- **不是通用三级缓存。** 本 PR 重点是 SSD → 设备侧 cache；它不与 `--cpu-moe` 组成完整的
  SSD → RAM → VRAM 热度分层，讨论中也明确把该方向留给后续设计；
- **不支持 streamed expert 上的 LoRA。** PR 会拒绝指向未物化 routed-expert tensor 的 LoRA；
- **host cache 与 op offload 有冲突。** 若 streaming cache 实际位于 host memory，context 会
  关闭 op offload，避免图 split 对可变权重做过期快照；
- **不适合小模型或慢介质。** 当模型本来能完整驻留，streaming 的 I/O、custom op、重复 GEMM
  与 mask 开销可能得不偿失；机械硬盘、网络文件系统或 direct I/O 不可用时风险更高。

### 建议监测的统计

PR 的 `print_stats()` 输出：

- `expert hits / misses / cold misses`；
- 普通 remap 的 `load stall`；
- wave 数、非空 wave 数；
- `preloads issued` 与 `ready on arrival`；
- wave 等待时间。

调参时，除了总 prefill tok/s，还要关注「预读到达率」：如果 `preloads issued` 很多但 `ready on arrival` 很低，说明 SSD 吞吐或 I/O 队列不足以在一个 wave 的计算窗口内完成预读；单纯再扩大预读范围通常没有帮助。

## 8. 一个完整的小例子

设：`n_expert_used = 2`、`n_slots = 8`。

```text
wave_capacity = floor((8 - 2) / 2) = 3
```

某层 ubatch 实际触及 7 个 expert：`[E1, E4, E6, E9, E10, E12, E15]`。

```text
wave 0: E1, E4, E6        # 装入并执行；同时尽力预读 wave 1
wave 1: E9, E10, E12      # 装入/复用预读并执行；同时尽力预读 wave 2
wave 2: E15               # 执行；不足 top-k 的停车 slot 从前一 pool 借用
```

对 token `T`，假设 router 选择 `E4` 与 `E12`：

```text
wave 0: E4 的 mask=1，E12 的 mask=0 → 仅保留 E4 输出
wave 1: E4 的 mask=0，E12 的 mask=1 → 仅保留 E12 输出
wave 2: 两者 mask=0                    → 不贡献输出
sum   : 得到 E4 + E12 的原始 MoE 结果
```

## 9. 延伸阅读与修改入口

- [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)：功能动机、CLI、验证结果、基准与评审讨论；
- [`src/llama-graph.cpp`（PR 固定提交）](https://github.com/ggml-org/llama.cpp/blob/1248fd8fa8cfebaece5ea992e4d951c1e18bb9d5/src/llama-graph.cpp)：`build_moe_ffn()` 负责把 wave custom op、GEMM、mask、累加接入 ggml 图；
- [`src/llama-moe-stream.cpp`（PR 固定提交）](https://github.com/ggml-org/llama.cpp/blob/1248fd8fa8cfebaece5ea992e4d951c1e18bb9d5/src/llama-moe-stream.cpp)：`plan_waves_locked()`、`stage_wave_locked()`、`emit_wave_slots()`、`llama_moe_stream_wave_mask()`；
- [`src/llama-moe-stream.h`（PR 固定提交）](https://github.com/ggml-org/llama.cpp/blob/1248fd8fa8cfebaece5ea992e4d951c1e18bb9d5/src/llama-moe-stream.h)：stream manager、layer、slot state 与统计结构；
- [Issue #25257](https://github.com/ggml-org/llama.cpp/issues/25257)：PR 引用的前置讨论。

如果要继续优化这个方向，最安全的顺序是：先用现有统计验证 wave 预读是否真正抵消 I/O 等待；再评估 slot 淘汰策略；最后才考虑跨层预测预读或三级缓存。不要先修改 router 或跳过 expert，因为那会把性能实验和模型质量变化混在一起。
