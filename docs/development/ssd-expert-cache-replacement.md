# SSD Expert Cache Replacement Policy

本文档说明 Vulkan SSD expert offload 中 expert 热缓存替换逻辑的作用和目的。相关代码
主要位于 `ggml/src/ggml-vulkan/ggml-vulkan.cpp` 的
`hot_expert_offload_process()`、`offload_expert_select_victim()` 和
`offload_expert_record_accesses()` 附近。

## 目的

SSD expert offload 场景下，完整 MoE expert 权重保存在 SSD 或后备存储中，Vulkan
后端只把一部分 expert 放入显存热缓存。每层热缓存容量由 `n_hot_expert` 控制，
通常小于该层全部 expert 数量。

当当前 microbatch 的 token 路由到某个尚未加载到显存的 expert 时，代码需要：

1. 从已经在显存中的 resident expert 里选择一个 victim。
2. 将新的 expert 的 gate/up/down 权重切片加载到 victim 所在 slot。
3. 更新 `expert_map[layer][expert]`，让后续计算使用新的 slot 映射。

这次修改的目的，是让 victim 选择不只依赖当前扫描顺序，而是结合 router 当前排序、
最近访问情况和历史访问频率，减少随意换出带来的缓存抖动。

## 关键状态

新增或使用的缓存状态按层维护：

- `expert_map[layer][expert]`: 记录 expert 当前是否在显存热缓存中；值为 slot 编号，
  或 `EXPERT_NOT_LOADED`。
- `expert_last_access[layer][expert]`: 记录某层某个 expert 最近一次被 token 路由命中
  时的 layer step。
- `expert_access_count[layer][expert]`: 记录某层某个 expert 累计被 token 路由命中的
  次数。
- `layer_steps[layer]`: 每层独立递增的逻辑时间戳。每完成一次该层的路由统计，就递增
  一次。

这里的 layer step 不是 token 序号，也不是 Vulkan 执行 step。它只是每一层自己的
访问轮次，用来判断某个 expert 距离上次被使用已经过去了多久。

## 路由统计

MoE router 会为每个 token 对 expert 打分并排序。代码中的 `argsort_buffer` 保存
这个排序结果，rank 越小表示 router 分数越高。

每个 token 实际使用前 `n_expert_used` 个 expert，也就是 top-k expert。代码会统计
这些 top-k expert 在当前 microbatch 中被命中的次数：

```cpp
routed_counts[expert]++;
```

这些计数用于两个目的：

1. 标记当前 microbatch 必须保留的 `required` expert，避免把马上要用的 expert 换出。
2. 在本轮处理完成后更新 `expert_last_access` 和 `expert_access_count`。

## Victim 选择

`offload_expert_select_victim()` 负责选择要换出的 expert。候选 victim 必须满足：

1. 当前已经在显存热缓存中。
2. 不是当前 microbatch required 的 expert。

对每个候选 expert，代码计算三个指标：

```cpp
const int32_t rank_value = eviction_rank[expert];
const uint64_t age = offload_expert_access_age(last_access, step, expert);
const uint32_t frequency = access_count[expert];
```

`rank_value` 是主要依据。它来自当前 microbatch 的完整 router 排序，表示该 expert
在本轮排序中出现过的最差 rank。值越大，说明该 expert 在当前 router 看来越靠后，
越适合被换出。

`age` 表示该 expert 距离上次被路由命中过了多少个 layer step。值越大，说明越久没
使用。当两个 expert 的 `rank_value` 相同时，优先换出 `age` 更大的 expert。

`frequency` 表示该 expert 历史累计被路由命中的次数。当 `rank_value` 和 `age` 都
相同时，优先换出 `frequency` 更小的 expert。

因此实际优先级是：

1. `eviction_rank` 更大的 expert 优先换出。
2. `eviction_rank` 相同，则更久没访问的 expert 优先换出。
3. 前两者都相同，则历史命中次数更少的 expert 优先换出。

## 访问历史更新

`offload_expert_record_accesses()` 在本次 offload/mapping 完成后更新访问历史。
每调用一次，它会先递增当前层的 `layer_steps[layer]`，然后只更新本轮实际被 top-k
路由命中的 expert：

```cpp
last_access[expert] = step;
access_count[expert] += routed_count;
```

`access_count` 使用饱和累加，超过 `uint32_t` 最大值时保持在最大值，避免计数溢出。

## 效果

这套策略保留了 router rank 作为主判断依据，因此仍然优先换出当前 router 认为较不
重要的 expert。同时，recency 和 frequency 只作为 tie-breaker 使用，用来避免同 rank
情况下由遍历顺序决定 victim。

这部分 victim 选择逻辑不重新计算 router 分数，只决定热缓存 slot 应该由哪个
resident expert 让出。整体目标是在有限显存热缓存容量下，让 expert 换入换出更稳定，
降低刚被频繁使用的 expert 被随意驱逐的概率。
