# Vulkan SSD 专家缓存 Top-K 替换说明

本文档说明以下两个提交在 Vulkan 后端中对 Qwen3.5MoE SSD 专家缓存路由所做的修改：

- `21e25dd4c1ccf4d0ed528fb6bac74a270c46460a`（`top k+1`）
- `4f66971f08ef32f54cff72b3e6bae017e788d0ba`（`topk+8`）

对应实现位于 [`ggml/src/ggml-vulkan/ggml-vulkan.cpp`](../../ggml/src/ggml-vulkan/ggml-vulkan.cpp) 的 `hot_expert_offload_process()`。

## 背景

Qwen3.5MoE 的每一层可能有很多专家，但一次路由只使用 `n_expert_used` 个专家。Vulkan SSD 专家缓存只把 `n_hot_expert` 个专家的 `gate`、`up`、`down` 权重保留在显存中，其余专家需要从 SSD 读取。

路由结果通过 `argsort_buffer` 传入：

- 数组中的每个值是专家 ID；
- 前 `n_expert_used` 个值是当前 top-k 路由候选；
- 后续值是未进入 top-k 的候选；
- `expert_map[layer][expert]` 保存专家当前对应的显存缓存槽位，`-1` 表示不在缓存中。

每层计算前，后端需要保证当前使用的专家已经在显存中，然后把这些专家对应的槽位写入 `mapped_topk_tensor`，供后续 MoE 矩阵计算使用。

## `21e25dd`：第 K+1 个候选替换初版

该提交首次尝试利用第 `n_expert_used + 1` 个候选：如果该专家已经在缓存中，就用它替换当前 top-k 中最低优先级且未缓存的专家，从而避免一次 SSD 读取。

不过这个版本只定义了 `selected_expert_id` 替换函数，后续处理仍然直接读取 `argsort_buffer[i]`。因此替换结果没有真正参与加载、映射或最终 top-k 写回，属于未完成的初版实现。

## `4f66971`：扩展到后续 K 个候选

该提交将替换逻辑完整接入路由处理，并把候选范围扩展为后续最多 `n_expert_used` 个专家：

```text
candidate_end = min(n_expert, 2 * n_expert_used)
```

以 `n_expert_used = 8` 为例，当前 top-k 是排序位置 1～8，候选窗口是位置 9～16。实际窗口会在专家总数不足时截断。

处理步骤如下：

1. 复制并校验当前 top-k 专家 ID。
2. 从低优先级到高优先级记录当前 top-k 中尚未缓存的槽位。
3. 在后续候选窗口中收集已经缓存的专家。
4. 将后续缓存专家依次填入低优先级的冷专家位置，最多替换两边数量的较小值。
5. 对仍未缓存的选中专家执行正常的 SSD 加载，并从路由尾部选择未被选中的驻留专家进行淘汰。
6. 将替换后的专家列表复制回 `argsort_buffer`，使后续 `topk_buffer` 映射使用同一组专家。

## 示例

假设当前 top-k 为 `[A, B, C, D, E, F, G, H]`，其中 `G`、`H` 尚未缓存；后续候选为 `[I, J, ...]`，且 `I`、`J` 已缓存。

低优先级位置先被替换，因此最终列表为：

```text
[A, B, C, D, E, F, J, I]
```

这里的目的不是重新计算路由分数，而是用排序结果中紧随 top-k 的驻留专家减少 SSD 读取。该策略是显存/IO 优化启发式，可能改变原始 top-k 专家集合；需要通过目标模型的输出质量和性能测试确认其适用性。

## 缓存映射和计算链路

一次专家路由的关键链路是：

```text
argsort tensor
    ↓ Vulkan -> argsort_buffer
hot_expert_offload_process(layer_id)
    ↓ 更新 expert_map 和 argsort_buffer
topk_buffer[topk] = expert_map[layer][argsort_buffer[topk]]
    ↓ 上传 mapped_topk_tensor
gate / up / down 使用显存中的 hot_*_tensor
```

`expert_map` 的含义是“专家 ID -> 缓存槽位”，而不是“路由位置 -> 缓存槽位”。因此替换后必须使用替换后的专家 ID 再查询映射；不能只改变缓存内容而保留旧的路由索引。

## 边界条件

- `n_expert_used >= n_expert` 时没有后续候选，逻辑退化为普通缓存加载。
- 专家总数小于 `2 * n_expert_used` 时，候选范围被 `n_expert` 截断。
- 后续候选全部不在缓存中时，不发生替换，仍按原流程加载当前 top-k。
- 当前 top-k 全部已缓存时，不需要替换或加载。
- 任意 argsort 专家 ID 越界都会返回失败，避免访问 `expert_map` 越界。
- 初始化阶段要求 `n_hot_expert >= n_expert_used`；该约束由 SSD 专家 provider 初始化逻辑保证。

该修改没有增加 Vulkan shader 或公开 provider API，也没有改变 `--ssd-expert-cache-experts` 的含义；它只改变运行时选择和替换缓存专家的策略。

