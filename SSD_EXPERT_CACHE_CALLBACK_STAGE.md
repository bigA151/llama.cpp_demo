# MoE 专家缓存 Callback 架构上移阶段说明

## 目标与当前边界

本阶段的目标是将 MoE 专家缓存的**触发与调度逻辑**从 Vulkan 后端逐步上移到 `llama_context` 和通用 backend scheduler，使模型层能够负责解释路由结果和制定缓存计划，而具体 backend 只负责数据传输与数值计算。

这是一项分阶段改造。目前已经完成 callback 触发、上层缓存规划和同步性能修复，但实际热缓存张量、专家权重上传、`mapped_topk` 上传以及 MoE 节点输入替换仍由 Vulkan 实现。因此当前版本仍然只有 Vulkan backend 能使用 SSD 专家缓存，尚不能宣称已经支持 CUDA 等其他 backend。

## 1. 当前已经实现的内容

### 1.1 使用 eval callback 监测路由结果

计算图构建时，完整专家排序张量被命名为：

```text
ffn_moe_argsort-<layer>
```

`llama_context` 的 eval callback 同时检查节点名称和 `GGML_OP_ARGSORT` 操作类型，从名称中解析层号。这样不再依赖“当前节点前后固定偏移几个位置”来判断目标节点。

执行阶段的主要流程为：

```text
scheduler 执行本层截至 argsort 的计算节点
        ↓
在 argsort 节点处进行中间同步
        ↓
CPU 读取该层所有 token 的专家排序结果
        ↓
llama_context 制定本层专家缓存计划
        ↓
Vulkan 执行专家加载、槽位更新和 mapped_topk 上传
        ↓
继续执行本层 MoE 计算及下一层路由计算
```

callback 只负责在正确的计算图位置取得控制权。专家打分和 `argsort` 本身仍由张量所在的 backend 执行，不是由 CPU 重新计算。

### 1.2 将缓存规划逻辑上移到 llama_context

缓存规划函数现在根据以下信息决定本轮操作：

- 当前层号；
- 模型总专家数和每个 token 的 top-k 专家数；
- 当前 microbatch 中所有 token 的完整 `argsort` 结果；
- 当前层 `expert_map` 中记录的专家到缓存槽位映射；
- 用户通过 `--ssd-expert-cache-experts` 指定的每层缓存容量。

上层规划逻辑完成以下工作：

1. 汇总本轮所有 token 的 top-k 专家并取并集；
2. 判断这些专家是否已经驻留在当前层缓存池；
3. 找出发生 cache miss 的专家；
4. 为缺失专家选择可淘汰的驻留专家和目标槽位；
5. 生成原始专家编号到缓存槽位编号的 `mapped_topk`；
6. 将“加载哪个专家、淘汰哪个专家、写入哪个槽位”的计划交给 Vulkan。

当前仍采用原有的简单淘汰规则，没有在本阶段合入其他伙伴新增的复杂缓存策略。若一个 microbatch 所需的专家并集超过每层缓存容量，当前逻辑仍会报错；上游通过限制实际 microbatch 大小来避免这一情况。

### 1.3 Vulkan 当前保留的职责

Vulkan 不再负责解释路由策略，但仍负责所有 backend 专属操作：

- 创建并维护每层的 gate/up/down 热专家缓存张量；
- 根据上层计划从 GGUF/SSD 读取指定专家切片；
- 将专家切片上传到指定 GPU 缓存槽位；
- 更新 Vulkan 内部的专家驻留映射；
- 将 `mapped_topk` 上传到 GPU；
- 根据节点名称和操作类型定位本层 gate/up/down 节点；
- 把这些节点的权重输入和专家编号输入替换为热缓存张量与 `mapped_topk`。

因此，当前架构已经把“选择什么”移到上层，但“如何在设备上应用这个选择”仍然是 Vulkan 专用实现。

### 1.4 不再依赖固定节点相对位置

旧实现先找到 `ffn_moe_weights-<layer>`，再通过 `i - 2`、`i + 6`、`i + 7`、`i + 11` 等固定相对位置寻找 `argsort` 和 gate/up/down 节点。这种方式依赖特定计算图排列，节点融合、模型结构或调度拆图发生变化时容易失效。

当前实现改为：

- callback 通过 `ffn_moe_argsort-<layer>` 和 `GGML_OP_ARGSORT` 找到路由结果；
- Vulkan 应用缓存时通过 `ffn_moe_gate-<layer>`、`ffn_moe_up-<layer>`、`ffn_moe_down-<layer>` 和 `GGML_OP_MUL_MAT_ID` 定位计算节点。

这消除了固定节点偏移，但运行时修改节点 `src` 的做法仍待后续上移。

## 2. 后续需要实现的内容

### 2.1 建立 backend 无关的专家缓存接口

当前接口和类型仍带有 `vk` 前缀，例如 Vulkan provider、Vulkan tensor info 和 Vulkan cache plan。后续需要将模型信息、专家切片读取、缓存规划与设备上传接口拆分为通用结构，避免上层逻辑直接依赖 Vulkan 类型。

目标职责边界应当是：

```text
llama/context：解释模型、读取路由结果、维护缓存状态、制定缓存计划
backend：按计划传输专家权重，并执行后续 MoE 数值计算
```

不同 backend 可以分别实现“将某个专家切片写入某个设备槽位”的操作，而不需要复制缓存决策算法。

### 2.2 将实际缓存状态移出 Vulkan

目前 `expert_map`、热缓存槽位及部分缓存状态仍保存在 Vulkan context 中。后续应将 backend 无关的缓存状态放到上层缓存管理对象中，Vulkan 只保存实际设备 buffer 和完成传输所需的私有资源。

这样做可以让同一套缓存策略用于 Vulkan、CUDA 或其他 backend，也能避免 backend 为了作出缓存决策而了解过多模型结构。

### 2.3 避免 Vulkan 在运行时修改计算图节点输入

当前 Vulkan 在应用缓存计划后仍会修改 gate/up/down 节点：

```text
src[0]：完整专家权重 → 热专家缓存张量
src[2]：原始专家编号 → 缓存槽位编号 mapped_topk
```

最终应在构图阶段直接让 MoE 节点使用热缓存张量和可更新的 `mapped_topk`，或者建立通用的专家缓存算子/接口。这样 backend 只执行已经构建好的计算图，不再负责解释节点名称并修改图结构。

这部分会涉及缓存张量的生命周期、scheduler 分配、backend placement 以及计算图复用，是后续架构改造中影响较大的部分。

### 2.4 后续再合入缓存策略优化

其他分支中新增的访问频率、最近访问时间和 router rank 等淘汰策略，可以在通用缓存管理接口稳定后迁移到上层。策略优化与本阶段的架构上移可以分开进行，避免同时修改执行边界和淘汰规则而增加调试难度。

## 3. Callback 引入后的性能问题与修复

### 3.1 性能下降的原因

为了让 CPU 在每层路由完成后管理缓存，scheduler 会以 `ffn_moe_argsort-<layer>` 为分段边界：先把截至该节点的命令提交给对应 backend，等待执行完成，再调用 callback。

初版 callback 路径在每个分段后调用完整的 `ggml_backend_synchronize()`。对于 Vulkan，完整同步除了提交命令、等待 fence 和复制输出外，还会调用 `ggml_vk_graph_cleanup()`。因此一张包含多层 MoE 的计算图会在每层 callback 后反复清理 Vulkan 临时执行资源。

旧 Vulkan 专家缓存路径同样存在必要的层间同步，但层内主要使用内部 `ggml_vk_synchronize()`，通常不会在每层执行完整 graph cleanup。callback 初版把“等待路由结果”和“整张图执行结束后的清理”错误地绑定在了一起，因此 decode 速度明显降低。

### 3.2 区分中间同步和最终同步

当前新增了两种同步语义：

**中间同步**用于 callback 边界和 scheduler 内部的数据依赖：

- 提交尚未提交的 transfer/compute 命令；
- 等待 GPU 完成相关工作；
- 完成异步输出复制，使 CPU 可以读取 `argsort`；
- 不执行 graph cleanup。

**最终同步**用于一个完整 scheduler graph 执行结束：

- 等待最后剩余的 GPU 工作完成；
- 确保最终输出可用；
- 对本张图累计的 Vulkan 临时执行资源统一 cleanup。

一次 decode token 的执行关系大致为：

```text
第 0 层 argsort → 中间同步 → 缓存管理
第 1 层 argsort → 中间同步 → 缓存管理
...
第 N 层 argsort → 中间同步 → 缓存管理
图尾部计算       → 最终同步 → cleanup 一次
```

prefill 如果被拆成多个实际 ubatch，则每个 ubatch 都有一张独立计算图，并在各自结束时 cleanup 一次。

### 3.3 调度器通用同步的处理

除了 callback 层间同步，scheduler 在以下场景也需要等待 backend：

- 输入张量复制前；
- 覆盖仍可能被设备使用的目标 buffer 前；
- 跨 backend 复制张量前；
- 从设备读取专家编号等张量前；
- 异步复制不可用时的同步回退路径。

这些同步的目的只是保证数据依赖和 buffer 生命周期安全，不代表整张计算图已经完成。因此当前也改为使用中间同步，避免每次通用等待都附带 cleanup。

Vulkan context 使用 `cleanup_pending` 标记当前是否真的积累了待清理的图资源，使后续重复的完整同步不会再次执行空 cleanup。

### 3.4 延后 cleanup 的影响

延后释放的是 Vulkan 的临时执行管理资源，包括 command buffer 上下文、descriptor 使用记录、semaphore/event 和节点依赖记录。不会清除或复制保留以下大对象：

- 模型权重；
- KV cache；
- 专家缓存池；
- `expert_map`；
- 已计算的张量结果。

这些临时资源现在会存活到一个完整 ubatch 或 decode token 的计算图结束，因此峰值管理内存可能略有增加，但不会随整个对话无限增长。计算公式、节点顺序、专家权重内容和缓存策略均未改变，因此该修改本身不应影响模型精度。

需要继续验证的场景包括长时间生成、多轮对话、长 prompt、多 ubatch prefill 以及计算失败后的恢复路径，并观察 RAM/显存是否保持稳定。

### 3.5 当前诊断信息

当前保留了由 `LLAMA_SSD_EXPERT_TIMING` 控制的临时诊断日志，可分别观察：

- scheduler 的分段、提交、同步和 callback 时间；
- context 读取 `argsort` 与应用缓存计划的时间；
- Vulkan 每层的专家加载与 `mapped_topk` 上传时间；
- Vulkan 同步内部的 transfer、fence wait、output copy 和 cleanup 时间。

这些日志用于确认性能瓶颈和同步生命周期。架构与性能稳定后，可以将带有明确诊断标记的代码作为独立部分删除。

## 阶段结论

本阶段已经完成了从“Vulkan 在图执行过程中自行发现路由节点并决定缓存操作”到“scheduler 在 callback 边界暂停、`llama_context` 读取路由结果并制定缓存计划、Vulkan 应用计划”的转换。

当前改造降低了计算图节点相对位置耦合，并把缓存策略移动到了更接近模型语义的上层。同时，通过区分中间同步和最终同步，消除了 callback 分段带来的重复 cleanup 开销，恢复了 decode 性能。

下一阶段的重点不是继续移动同一段函数，而是定义真正 backend 无关的缓存状态和设备操作接口，并逐步取消 Vulkan 对模型信息、缓存决策和运行时节点输入修改的依赖。
