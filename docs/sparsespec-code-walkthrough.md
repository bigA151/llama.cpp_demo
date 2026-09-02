# SparseSpec 代码导读

本文档介绍外部研究项目 [sspec-project/SparseSpec](https://github.com/sspec-project/SparseSpec) 的代码组织和主要执行流程。SparseSpec 实现了基于稀疏注意力的自推测解码（sparse self-speculative decoding）：同一模型在草拟阶段使用稀疏 KV Cache，在验证阶段使用完整 KV Cache。

> [!IMPORTANT]
> SparseSpec 目前是研究原型。本文档是源码阅读笔记，不表示 llama.cpp 已实现或集成 SparseSpec，也不建议直接把其 Python、PyTorch、Triton 或 FlashInfer 代码移植到 llama.cpp。

本文档参考 SparseSpec `main` 分支提交 [`5c9616c9`](https://github.com/sspec-project/SparseSpec/commit/5c9616c9af109ecec0fdf193221d3f5be6f05922) 及论文 [Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding](https://arxiv.org/abs/2512.01278)。

## 1. 代码目录

SparseSpec 的核心实现位于 `serve/`：

```text
serve/
├── run.py                         # 参数解析、模型和 KV Cache 初始化、调度器入口
├── model/
│   ├── model.py                   # Transformer、KVPool 和模型加载
│   └── graph_model.py             # CUDA Graph 包装
├── scheduler/
│   ├── scheduler.py               # 普通生成、请求准入和 KV 卸载基础逻辑
│   └── spec_scheduler.py          # DRAFT/VERIFY 状态机和推测解码
├── request/
│   ├── request.py                 # 单请求状态及 draft 概率缓冲区
│   └── kv_cache_ptr/
│       ├── base.py                # 完整 KV Cache、页分配、卸载和恢复
│       ├── streaming.py           # 固定窗口稀疏缓存基础实现
│       └── pillar.py              # PillarAttn 的关键 KV 选择与元数据
├── attention/
│   └── backend.py                 # 自定义 FlashInfer attention 和 Top-K 封装
└── sampling/
    ├── sampler.py                 # 普通采样
    └── rejection_sampler.py       # 保持目标分布的拒绝采样
```

## 2. 总体执行流程

一次请求大致经过以下路径：

```text
run.py
  │
  ├─ 初始化 Transformer、KVPool、PillarCachePtr
  ├─ 创建 SpecScheduler
  │
  └─ NORMAL（分块 prefill）
       ↓
     DRAFT × k（稀疏 attention，每次产生一个候选 token）
       ↓
     VERIFY（完整 attention，一次验证候选序列）
       ↓
     RejectionSampler（接受、恢复或加入 bonus token）
       ↓
     用验证阶段的 attention score 更新 Top-K KV 索引
       ↓
     下一轮 DRAFT
```

这里的“自推测”不是跳过 Transformer 层，也不是加载额外的小模型。草拟和验证使用相同的模型权重及完整 MLP；两者的主要差异是 attention 读取的 KV 范围。

## 3. 初始化入口

入口文件是 [`serve/run.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/run.py)。

### 3.1 KV Cache 类型

`dispatch_str_to_cache_ptr()` 支持三种实现：

| 参数 | 类型 | 用途 |
| --- | --- | --- |
| `full` | `FullCachePtr` | 完整 attention，不启用推测调度器 |
| `streaming` | `StreamingCachePtr` | 固定窗口稀疏 attention |
| `pillar` | `PillarCachePtr` | 动态 Top-K 稀疏 attention，即 SparseSpec 核心路径 |

当 `kv_cache != "full"` 时，`run_inference()` 创建 `SpecScheduler`；否则使用 `BaseScheduler`。

### 3.2 关键参数

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `spec_stride` | `16` | 每轮最多草拟的步数 `k` |
| `budget_ratio` | `0.05` | 草拟阶段读取的 KV Cache 比例 |
| `num_min_budget` | `128` | 稀疏 KV 的最小 token 预算 |
| `batch_policy` | CLI 默认为 `random` | 请求在不同推测阶段之间的分配策略 |
| `async_cpu` | 关闭 | 是否启用延迟验证和 CPU/GPU 重叠 |
| `admit_policy` | `naive` | 请求准入策略；稀疏路径实际需要选择可用的非 `naive` 策略 |

`run_inference()` 把 KV 页大小硬编码为 `1`。因此一个 token 对应一个 page，调度器能够直接用 page 索引表示逐 token 的稀疏模式，但会产生更细粒度的页元数据。

`RejectionSampler.MAX_SPEC_LEN` 会根据 `spec_stride` 向 4 的倍数扩容。每个请求还会分配形状为 `[MAX_SPEC_LEN, vocab_size]` 的 FP32 `draft_probs`，因此较大的推测长度、词表和并发量会显著增加显存占用。

## 4. 请求状态机

[`serve/request/request.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/request/request.py) 定义四种执行状态：

| 状态 | 作用 |
| --- | --- |
| `NORMAL` | prompt prefill 或普通生成 |
| `DRAFT` | 使用选中的稀疏 KV 产生一个候选 token |
| `VERIFY` | 使用完整 KV 并行验证本轮候选 token |
| `STALL` | 异步模式下等待 CPU 处理上一轮验证结果 |

主要转换关系为：

```text
NORMAL --prefill 完成--> DRAFT
DRAFT  --累计到 k 步--> VERIFY
VERIFY --同步模式-----> DRAFT
VERIFY --异步模式-----> STALL --下一周期--> DRAFT
```

`SpecScheduler.add_request()` 可以把新请求分散到不同的推测阶段。`greedy` 策略统计各阶段的请求数，把新请求放入负载最轻的 bucket，从而避免所有请求同时进入昂贵的验证阶段。

## 5. 草拟阶段

草拟阶段仍执行完整 Transformer 和 MLP，但 attention 只读取稀疏 KV 索引。

### 5.1 从完整缓存转换为 Pillar 缓存

prefill 完成后，`SpecScheduler.update_requests()` 调用：

```python
PillarCachePtr.from_kv_cache(full_cache, ...)
```

该转换复用已有 KV 页，不复制模型权重和 KV 数据。首次进入草拟阶段时还没有验证阶段提供的 attention score，因此属于 cold start：代码暂时把当前完整 KV 都视为近期窗口。第一轮验证产生 Top-K 结果后，后续 DRAFT 才真正使用稀疏历史 KV。

### 5.2 构造稀疏 KV 索引

[`PillarCachePtr.concat_selected_indices()`](https://github.com/sspec-project/SparseSpec/blob/main/serve/request/kv_cache_ptr/pillar.py) 将两部分索引拼接起来：

```text
draft KV indices = 历史 Top-K 关键 token + 最近生成的 token
```

保留最近 token 是必要的，因为刚生成的候选 token 尚未参与上一轮 Top-K 选择，但后续草拟步骤必须能够关注它们。

### 5.3 Attention 后端分流

[`BatchTopKAttention`](https://github.com/sspec-project/SparseSpec/blob/main/serve/attention/backend.py) 根据 `request_type` 选择索引：

- `DRAFT`：把输入索引重定向到拼接后的 `flatten_indices`；
- `NORMAL` 和 `VERIFY`：使用完整的 `kv_indices`。

因此同一个 batch 可以同时包含稀疏草拟请求和完整验证请求，而不需要分别运行两套模型。

每次 DRAFT forward 后，当前 token 的草拟概率分布会被写入请求自己的 `draft_probs` 缓冲区，稍后交给拒绝采样器。

## 6. 验证与无损采样

进入 `VERIFY` 时，`SpecScheduler.prepare_forward_inputs()` 把本轮候选 token 和一个 bonus 位置组成多 token 输入。目标模型使用完整 KV Cache 一次计算这些位置的概率分布。

[`serve/sampling/rejection_sampler.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/sampling/rejection_sampler.py) 实现标准 speculative decoding 拒绝采样：

1. 按草拟分布 `q` 生成候选 token；
2. 用目标分布 `p` 决定是否接受每个候选；
3. 首次拒绝时，从由 `p` 和 `q` 构造的修正分布中恢复一个 token；
4. 如果候选全部接受，则附加仅由目标分布生成的 bonus token。

`SpecScheduler.verify_outputs()` 随后执行三项修正：

- 删除请求中尚未确认的草拟 token，再写入接受或恢复后的 token；
- 通过 `retract_tokens()` 释放被拒绝位置的 KV 页；
- 根据实际接受长度准备下一轮稀疏索引更新。

稀疏 attention 只影响候选质量。最终 token 经过完整目标分布验证，因此算法保持目标模型的采样分布；候选较差时损失的是接受率和性能，而不是输出正确性。

## 7. PillarAttn 的动态 Top-K

PillarAttn 的关键逻辑位于 [`serve/request/kv_cache_ptr/pillar.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/request/kv_cache_ptr/pillar.py)。

### 7.1 KV 预算

`get_select_metadata()` 按下面的规则计算下一轮草拟预算：

```text
token_budget = max(num_min_budget,
                   ceil(current_kv_length * budget_ratio))
```

默认 `budget_ratio = 0.05`，即长上下文下约读取 5% 的 KV。预算还包括最近接受的 token；剩余预算才用于历史 Top-K token。

### 7.2 从验证结果恢复 attention score

验证 forward 使用完整 KV。`BatchTopKAttention.run_attention()` 返回 attention 输出和 log-sum-exp（LSE），随后 `rematerialize_scores()` 使用验证阶段的 Q、K 和 LSE 恢复归一化 attention score。

当前实现选择单独的 score kernel 重新物化分数，而不是把所有未归一化 logits 直接写入显存。这样减少了中间结果的存储量，代码也更容易维护，但仍会产生一次额外的 Q/K 读取。

### 7.3 更新关键索引

`PillarCachePtr.dispatch_update_selected_indices()` 的主要步骤是：

1. 找出上一周期处于 `VERIFY` 的请求；
2. 按层和 KV head 对验证得到的 attention score 执行 Top-K；
3. 把选出的物理 KV 页索引写入每个请求的 `selected_indices`；
4. 下一轮 DRAFT 将这些历史 Top-K 索引与最近 token 索引拼接。

因此稀疏模式不是固定滑动窗口，而是每轮验证后随推理上下文更新。

## 8. 统一调度与延迟验证

### 8.1 统一 batch

草拟请求通常只有一个输入 token，并且只读稀疏 KV；验证请求包含多个输入 token，并读取完整 KV。若所有请求严格同步执行，GPU 负载会在轻量草拟和重型验证之间反复波动。

`SpecScheduler` 通过错开请求的 `cur_spec_idx`，让同一个 batch 同时包含不同阶段的请求。Attention kernel 再根据每个请求的 `request_type` 分派稀疏或完整索引。

### 8.2 延迟验证

启用 `async_cpu` 后，验证请求从 `VERIFY` 转为 `STALL`。CPU 处理接受长度、回滚和新 Top-K 索引时，其他处于 DRAFT 的请求仍可提交下一轮 GPU 工作。处理完成后，STALL 请求重新进入 DRAFT。

该机制消除的是整批请求等待 CPU 验证的同步点，并没有跳过目标模型验证。

## 9. KV Cache 管理

[`KVPool`](https://github.com/sspec-project/SparseSpec/blob/main/serve/model/model.py) 同时维护 GPU 和 pinned CPU KV 存储：

- GPU 页由 `_free_gpu` 管理；
- CPU 页由 `_free_cpu` 管理；
- `offload()` 使用 non-blocking copy 把 KV 分块复制到 CPU；
- `restore()` 在请求重新准入时复制回 GPU；
- `Request.MAX_OFFLOADING_LEN` 把单次卸载限制为固定 chunk。

`BaseScheduler.admit_requests()` 的 `offloading` 策略在显存不足时逐块卸载请求，并在显存恢复后优先恢复已有请求。Pillar 稀疏索引还需要在物理页和稳定的逻辑页编号之间转换，以免 GPU 页释放并重新分配后指向错误 KV。

## 10. 代码中的重要约束

阅读或复现实验时需要注意：

- 项目 README 将其定义为 proof of concept，并非生产推理服务；
- 当前实现依赖 PyTorch、Triton、定制 FlashInfer 和 RAFT；
- 主要实验面向 NVIDIA H100，不能直接推断其他 GPU 或 CPU 后端的收益；
- `page_size` 固定为 `1`；
- `PillarCachePtr` 使用多个固定上限缓冲区，例如最大稀疏 KV 长度和最大验证 KV 总长度；
- 某些 GQA 配置下，验证 token 数乘以 GQA group size 超过 kernel tile 上限会触发精度警告；
- 每个请求保存多步完整词表概率，较大词表和高并发会带来显著显存成本；
- 草拟阶段仍运行完整 MLP，所以它主要优化长上下文下 memory-bound 的 attention，而不是减少全部计算；
- 短上下文、低并发或 compute-bound 工作负载不一定受益。

## 11. 与 llama.cpp 的对应关系

llama.cpp 已有 speculative decoding 框架，见 [speculative.md](speculative.md)，但 SparseSpec 所需的草拟器与现有 draft model、n-gram 草拟器不同。若进行原型设计，至少需要解决以下接口问题：

| SparseSpec 能力 | llama.cpp 中需要的抽象 |
| --- | --- |
| 同一模型在 DRAFT/VERIFY 使用不同 KV 索引 | 每序列、每层或每 head 的动态 KV 视图 |
| 验证时导出 attention score 或等价重要性统计 | backend 无关的 score/Top-K 数据通路 |
| 稀疏和完整 attention 混合 batch | graph 构建、调度和 kernel 对异构 KV 长度的支持 |
| 拒绝后的 KV 回滚 | 与现有 speculative acceptance 和 sequence rollback 对接 |
| 动态关键 KV 更新 | 不破坏完整 KV Cache 的旁路稀疏索引存储 |
| CPU KV 卸载后的索引稳定性 | 物理 cell/page 与逻辑 token 位置的映射 |

最小可行实验可以先限制为单序列、贪心采样和单一 GPU backend，验证下面三个指标后再扩大范围：

1. 稀疏 attention 的候选接受长度；
2. Top-K 选择和索引构造的额外耗时；
3. 完整验证、KV 回滚后与基线生成结果是否一致。

## 12. 推荐阅读顺序

建议按以下顺序阅读 SparseSpec 源码：

1. [`serve/run.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/run.py)：掌握初始化和参数；
2. [`serve/request/request.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/request/request.py)：了解请求状态；
3. [`serve/scheduler/spec_scheduler.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/scheduler/spec_scheduler.py)：跟踪完整状态机；
4. [`serve/request/kv_cache_ptr/pillar.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/request/kv_cache_ptr/pillar.py)：理解稀疏索引生命周期；
5. [`serve/attention/backend.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/attention/backend.py)：理解 kernel 如何按请求类型分流；
6. [`serve/sampling/rejection_sampler.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/sampling/rejection_sampler.py)：确认无损验证过程；
7. [`serve/scheduler/scheduler.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/scheduler/scheduler.py) 和 [`serve/model/model.py`](https://github.com/sspec-project/SparseSpec/blob/main/serve/model/model.py)：补充请求准入、KV 卸载和模型执行细节。
