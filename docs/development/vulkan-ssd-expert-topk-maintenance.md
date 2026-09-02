# Vulkan SSD 专家缓存 Top-K 维护文档

本文档面向维护 `ggml/src/ggml-vulkan/ggml-vulkan.cpp` 中 Qwen3.5MoE SSD 专家缓存路由的开发者，覆盖提交 `21e25dd4c1ccf4d0ed528fb6bac74a270c46460a` 和 `4f66971f08ef32f54cff72b3e6bae017e788d0ba` 引入的逻辑。

## 代码入口和调用顺序

运行时入口在 `ggml_backend_vk_graph_compute()`：

1. 从 argsort 节点读取完整专家 ID 列表到 `om_519.argsort_buffer`。
2. 调用 `hot_expert_offload_process(layer_id)`。
3. 根据更新后的 `argsort_buffer` 查询 `expert_map`，填充 `om_519.topk_buffer`。
4. 将 `topk_buffer` 上传到 `mapped_topk_tensor`。
5. 把 MoE 的 `gate`、`up`、`down` 节点改为使用 `hot_gate_tensor`、`hot_up_tensor`、`hot_down_tensor` 以及映射后的 top-k 槽位。

初始化由 `offload_expert_buffer_manager_init()` 完成：它读取 provider 参数、为每层创建 hot expert tensor，并把前 `n_hot_expert` 个专家装入显存槽位。

## 关键状态和不变量

### `om_519.argsort_buffer`

- 长度为 `n_expert`。
- 输入时应是有效的专家排序结果。
- `hot_expert_offload_process()` 返回成功后，前 `n_expert_used` 项是实际交给 MoE 计算的专家 ID。
- 替换后的前 K 项必须写回该数组，否则后续 `topk_buffer` 会映射到错误的专家。

### `om_519.expert_map[layer]`

- `expert_map[expert] == slot >= 0` 表示专家驻留在该缓存槽位。
- `expert_map[expert] == EXPERT_NOT_LOADED` 表示专家不在显存中。
- 一个缓存槽位只能对应一个驻留专家。
- 选中的专家不能在同一次处理过程中被淘汰。

### `om_519.hot_*_tensor[layer]`

三类 tensor 的专家维度只有 `n_hot_expert`，槽位由 `expert_map` 给出。加载一个专家必须同时写入 `gate`、`up`、`down` 三类权重，且使用同一个槽位。

## 当前替换策略

提交 `4f66971` 的算法可以概括为：

```text
selected = argsort[0 : K]
cold_positions = selected 中未缓存的位置，按低优先级到高优先级排列
replacement_experts = argsort[K : min(n_expert, 2*K)] 中已经缓存的专家

把 replacement_experts 依次填入 cold_positions
对剩余未缓存的 selected 专家执行加载和淘汰
把 selected 写回 argsort[0 : K]
```

需要注意以下实现细节：

- `cold_positions` 从 `K - 1` 向前收集，因此总是优先替换原 top-k 中优先级最低的专家。
- 后续候选按排序顺序收集，因此优先使用更靠前的候选。
- 替换专家已经驻留，不应再次从 SSD 加载。
- 淘汰搜索必须排除 `selected_experts`，否则可能刚选中的专家在后续循环中被淘汰。
- `expert_map` 的槽位分配在加载完成后才对新专家可用；加载失败必须返回失败状态。

## 修改指南

### 调整候选窗口

如果要把“后续 K 个”改成其他窗口，必须同时检查：

- 起点和终点是否都以 `n_expert` 截断；
- `argsort_buffer` 是否确实包含完整的 `n_expert` 个 ID；
- 候选专家是否可能与当前 top-k 重复；
- 替换数量是否仍使用 `min(cold_positions.size(), replacement_experts.size())`；
- 模型特定的 K 值是否仍然适用。

不要把 `8` 或“第 9～16 位”硬编码到通用逻辑中；这些数字只对应常见的 Qwen3.5MoE `n_expert_used = 8` 配置。

### 修改路由列表

任何改变 `selected_experts` 的代码都必须同步考虑：

1. `expert_map` 查询；
2. SSD 加载和缓存槽位分配；
3. 淘汰候选排除规则；
4. `argsort_buffer` 回写；
5. `topk_buffer` 上传。

只修改其中一环会造成“缓存内容”和“路由 ID”不一致，表现为错误输出或显存槽位访问异常。

### 修改加载或失败处理

当前提交中的加载路径会按专家依次写入 `gate`、`up`、`down`。如果将其改成批量加载、异步加载或事务式更新，需要保证：

- 三类权重完成上传前，不把新专家标记为可用；
- 上传失败时不会留下错误的 `expert_map`；
- 设备同步点仍覆盖所有异步写入；
- 失败后下一次 graph compute 不会使用半更新的缓存状态。

当前函数可能在处理多个专家时已经完成前几个专家的更新，然后在后续加载失败。若未来要增强错误恢复，建议先构造计划中的映射和加载列表，全部成功后再一次性提交映射更新。

### 保持输入校验

保留专家 ID 的范围检查。新增入口或改变 argsort 来源时，还应考虑检查：

- argsort tensor 的字节数是否为 `n_expert * sizeof(int32_t)`；
- provider 的 `n_expert_used` 是否不大于 `n_expert`；
- provider 的 `n_hot_expert` 是否位于 `[n_expert_used, n_expert]`；
- argsort 输出是否包含重复 ID，若上游不再保证排列，应在这里拒绝重复输入。

## 日志和故障定位

重点日志包括：

- `invalid routed expert` / `invalid candidate expert`：argsort 输出越界；
- `no resident expert is available for eviction`：缓存状态、容量或淘汰规则无法满足；
- `failed to read SSD expert slice`：provider 读取失败；
- `expert replacement: rank ... <- expert ...`：发生了驻留候选替换。

定位问题时，建议记录同一层的以下信息：`layer_id`、`n_expert`、`n_expert_used`、`n_hot_expert`、原始 top-k、候选窗口、替换后的 top-k，以及每个专家对应的缓存槽位。

## 验证清单

使用支持 SSD 专家读取的 Vulkan 环境和 Qwen3.5MoE 模型进行验证。至少覆盖以下配置：

1. `--ssd-expert-cache-experts` 等于 `n_expert_used`；
2. 缓存数量大于 `n_expert_used` 且小于 `n_expert`；
3. 缓存数量等于 `n_expert`；
4. `n_expert` 小于 `2 * n_expert_used` 的模型配置；
5. 后续候选全部未缓存和部分已缓存两种路由状态。

每种配置至少检查：

- Vulkan 后端能够完成初始化和推理；
- `gate`、`up`、`down` 三类权重都使用同一缓存槽位；
- 发生替换时日志中的专家 ID 与最终 top-k 一致；
- 未发生越界访问、provider 读取失败或无可淘汰专家错误；
- 与未启用替换策略的基线比较推理延迟、SSD 读取量和输出质量。

该策略可能改变实际参与计算的专家集合，因此不能只用“输出字节完全相同”作为唯一验收标准；应结合目标模型的质量指标和性能指标判断。

## 回滚范围

这两个提交只修改 `ggml/src/ggml-vulkan/ggml-vulkan.cpp`。若需要恢复到替换策略之前的实现，应先撤销 `4f66971`，再撤销 `21e25dd`，并重新执行 Vulkan 构建和 Qwen3.5MoE SSD 专家缓存验证。

## 相关接口

- [`ggml/include/ggml-vulkan.h`](../../ggml/include/ggml-vulkan.h)：SSD 专家 provider 和 tensor 信息接口。
- [`common/arg.cpp`](../../common/arg.cpp)：`--ssd-expert-cache-experts` 命令行参数。
- [`src/llama-context.cpp`](../../src/llama-context.cpp)：Qwen3.5MoE Vulkan SSD 专家 provider 注册。

