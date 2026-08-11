# Wave-Partitional Prefill 移植说明

本文记录将 llama.cpp PR #25294 的 Wave-Partitional Prefill 思路移植到 `ZHW` 分支的实现、与本分支历史 SSD expert cache 代码的兼容处理，以及实际验证结果。

## 1. 背景

Qwen3.5MoE 的每个 token 会由 router 选出 `n_expert_used` 个 expert。当前分支把完整 expert 权重保存在 SSD，只让 `n_ssd_expert_cache` 个 expert slot 常驻 GPU。

原来的实现要求一个 ubatch 内所有被使用的 expert 同时装入 cache。当长 prompt 的 prefill ubatch 可能触及的 expert 数超过 slot 数时，只能把 `n_ubatch` 限制为：

```text
n_ssd_expert_cache / n_expert_used
```

这会把长 prompt 拆成大量小 ubatch，增加重复读取 expert、图调度和同步开销。

Wave-Partitional Prefill 不再缩小用户设置的 ubatch。它把一次 MoE expert GEMM 分成多个 wave：

1. 每个 wave 只装入并计算一部分 expert。
2. 不属于当前 wave 的 token-expert 对通过 mask 置零。
3. 所有 wave 的输出最终相加。

因此，同一个 ubatch 内每个被触及的 expert 只需要在所属 wave 中参与一次计算。

## 2. 与分支历史提交的关系

本次修改不是直接覆盖现有实现，而是在以下历史功能上继续演进：

- `4f66971f08ef32f54cff72b3e6bae017e788d0ba`（`topk+8`）：decode/small-ubatch 路径会优先用排名 9～16 中已驻留的 expert 替换 top-8 中的冷 expert，以减少 SSD 读取。
- `71c9648ea`：让 `mapped_topk_tensor` 的 token 维度可随 `n_tokens` 调整，解决早期实现只能安全处理 `-ub 1` 的问题。
- `24c39583d`：把 SSD cache planning 从 Vulkan 后端移动到 `llama_context` 的 eval callback，并增加 intermediate synchronize。
- `4b659b3cd`：当前移植前的 `ZHW` 基线，包含 MoE 相邻层 expert 重合率统计。

`24c39583d` 已经删除了 `4f66971f` 的 resident `topk+8` 回退逻辑，而且把每层 direct cache 更新都变成 scheduler callback 观察点。对单 token decode 来说，这会产生大量分段提交和同步。

本次修改恢复了历史 direct 快速路径及 `topk+8` 行为，同时仅把真正需要分 wave 的大 ubatch 留在 callback 路径。

## 3. direct 与 wave 的分流

构图时 router 的实际 argsort 尚未计算，因此无法提前知道精确的不同 expert 数。实现使用以下保守上界：

```text
n_touch_max = min(n_expert, n_tokens * n_expert_used)
```

分流规则为：

```text
n_touch_max <= n_ssd_expert_cache  -> direct 路径
n_touch_max >  n_ssd_expert_cache  -> wave 路径
```

### 3.1 direct 路径

direct 路径用于 decode 和能够由 cache 一次容纳的小 ubatch：

- 保持原有 `ffn_moe_gate/up/down-<layer>` 节点命名。
- 在 Vulkan graph compute 内读取 argsort、更新 `expert_map`、加载缺失 expert，并把 expert id 映射为 slot id。
- 启用已有的 resident `topk+8` 回退，以维持该分支原来的 decode 策略。
- 图结束时同步并释放本次 direct 图的 Vulkan graph-lifetime context。

最后一点用于修复长文本配合 `-ub 1` 时的 `0xC0000409` fail-fast。`24c39583d` 引入的 intermediate synchronize 会保留 graph-lifetime 资源；在一次 `llama_decode()` 内连续处理数百个 single-token ubatch 时，这些资源会持续累积。direct 图完成后显式同步和 cleanup 可以安全结束其生命周期，避免累计越界，同时不影响 wave callback 仍在使用的图资源。

### 3.2 wave 路径

wave 路径用于可能触及的 expert 数超过 slot 数的大 ubatch：

- 不再把 `n_ubatch` 限制为 cache slot 数能够一次容纳的大小。
- 构造连续的 top-k id tensor，并为每个 wave 创建 `ffn_moe_wave_ids_<wave>_<capacity>-<layer>` marker。
- 每个 wave 分别构造 gate、up、down `MUL_MAT_ID` 节点。
- callback 读取一次该层 argsort，随后逐 wave 更新 cache 和 `expert_map`，并上传当前 wave 的 slot 映射。
- mask 根据第一次出现的顺序给 expert 分配 wave；非当前 wave 的 token-expert 输出为零。
- 所有 wave 输出相加后，再进入原有 router weight 和 shared-expert 处理。

当前实现的 wave capacity 为：

```text
(n_ssd_expert_cache - n_expert_used) / 2
```

预留空间用于每个 token 的有效、互不重复 parking slot 以及 wave 切换期间的 cache 安全操作。若 slot 数不足以支持 wave，会在构图时明确终止，而不是产生无效映射。

wave 路径不执行 resident `topk+8` 替换。它计算 router 实际选中的 top-k expert，只通过分 wave 改变执行顺序，避免在长 prefill 中引入额外的近似路由。

## 4. `expert_map` 和节点查找

本次修改继续复用已有的持久化 `expert_map`：

```text
expert id -> GPU cache slot
```

没有增加另一份持久化的 `slot -> expert` 状态。cache planner 只在一次规划期间根据 `expert_map` 临时构造反向表，用于验证 slot 唯一性和完整性。

Vulkan 仍按节点名称查找目标 GEMM：

- direct：`ffn_moe_gate/up/down-<layer>`
- wave：`ffn_moe_gate/up/down_wave_<wave>_<capacity>-<layer>`

节点名称同时区分 layer、wave 和 capacity，是 `llama-graph` 与 Vulkan cache 应用逻辑之间的运行时契约。本次修改没有取消该机制。

## 5. 边界情况

### 5.1 最后一层零输出 token

Qwen3.5MoE 会在中间 prompt ubatch 的最后一个 transformer layer 进行输出裁剪，因此该层可能出现形状为 `[n_expert_used, 0, 1, 1]` 的空 tensor。

空 tensor 不代表 cache 错误。本次实现只对 `n_tokens > 0` 的 MoE 子图启用 SSD wave，并且 eval callback 忽略 `ggml_nbytes(tensor) == 0` 的 argsort/wave marker，避免以下错误：

```text
invalid SSD expert wave tensor ... ne=[8, 0, 1, 1]
```

### 5.2 映射合法性

cache planner 会验证：

- expert id 和 eviction candidate 均在合法范围内；
- 一个 slot 不会同时映射到两个 resident expert；
- 当前 wave 的 required expert 数不超过 cache 容量；
- 同一 token 的 `MUL_MAT_ID` slot id 有效且互不重复；
- 非当前 wave 的 pair 也会获得合法 parking slot，随后由 mask 清零输出。

## 6. 主要修改文件

- `src/llama-graph.cpp`
  - 构造 wave marker、mask、分 wave expert GEMM 和输出累加。
  - 小 ubatch 保持 direct 图。
- `src/llama-context.cpp`
  - cache planning、wave argsort callback、resident `topk+8` 回退和零 token 校验。
  - 移除 SSD cache 对 `n_ubatch` 的强制缩小。
- `src/llama-context.h`
  - 保存当前 wave callback 使用的 argsort layer/backend 状态。
- `src/llama-cparams.h`
  - 把 SSD cache slot 数传递给 graph builder。
- `ggml/include/ggml-vulkan.h`
  - 扩展 cache plan 和 wave cache 应用接口。
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
  - direct/wave cache 应用、动态 mapped top-k、按名称查找节点及 direct 图资源清理。

## 7. 构建环境

初始 PowerShell 环境没有加载 MSVC/Ninja，且环境里同时存在 `Path` 和 `PATH`，直接使用 MSBuild 会触发重复键异常。因此验证使用 Visual Studio Developer Command Prompt 和 Visual Studio 自带 Ninja：

```text
D:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat
D:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
```

构建目录：

```text
build-msvc-vulkan-ninja-check-release
```

`llama-cli` Release 构建成功；Vulkan `MUL_MAT_ID` 后端测试在 wave 初始实现后通过 692/692。

## 8. 实际模型验证

测试模型：

```text
D:\model\Qwen3.5-35B-A3B-Q4_K_M.gguf
```

测试输入为 523 tokens 的 UTF-8 中文长文本。自动测试在用户给出的核心参数之外增加 `-f <prompt-file> -st -n 1 --no-display-prompt -lv 3`，以便非交互运行并记录完整退出状态。

### 8.1 `-ub 1`

```text
.\llama-cli.exe -m "D:\model\Qwen3.5-35B-A3B-Q4_K_M.gguf" -c 4096 -ngl 999 --no-mmap -ub 1 -s 0 --temp 0 --reasoning-budget 0
```

结果：

```text
exit code        = 0
processed tokens = 523 / 523
graph splits     = 2
prompt speed     = 26.32 tokens/s
```

该速度与保留 `topk+8` 的 `71c9648ea` Release 基线实测 26.20 tokens/s 基本一致，说明 direct/decode 路径的性能特征已恢复。

### 8.2 `-ub 512`

```text
.\llama-cli.exe -m "D:\model\Qwen3.5-35B-A3B-Q4_K_M.gguf" -c 4096 -ngl 999 --no-mmap -ub 512 -s 0 --temp 0 --reasoning-budget 0
```

结果：

```text
exit code        = 0
processed tokens = 523 / 523
graph splits     = 482 (bs=512), 2 (bs=1)
prompt speed     = 59.51 tokens/s
```

两个测试均启用默认 warmup，未出现 compute error、零行 wave tensor 错误或 Windows fail-fast。

## 9. 当前适用范围

本次移植只在以下条件下启用：

- 模型架构为 `LLM_ARCH_QWEN35MOE`；
- 使用 Vulkan backend；
- GGUF 中存在本分支 SSD expert 地址信息；
- cache slot 数满足 `n_expert_used <= n_ssd_expert_cache < n_expert`。

其他模型和未启用 SSD expert cache 的运行路径保持原有行为。
