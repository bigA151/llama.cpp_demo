# SSD Expert Cache Top-k Substitution

本文档说明 Vulkan SSD expert cache 的 top-k expert 替换开关，以及替换后门控权重的处理方式。

## 默认行为

默认不启用 expert 替换策略。也就是说，router 选出的原始 top-k expert 会被保留：

```text
top-1 .. top-k
```

如果这些 expert 不在显存热缓存中，就进入 SSD load + eviction 流程，把缺失 expert 从 SSD 读入热缓存。

## 命令行开关

开启替换策略：

```bash
--ssd-expert-cache-substitute
```

配合缓存 expert 数使用示例：

```bash
--ssd-expert-cache-experts 120 --ssd-expert-cache-substitute
```

不传 `--ssd-expert-cache-substitute` 就是关闭替换策略。环境变量也可以开启：

```bash
LLAMA_ARG_SSD_EXPERT_CACHE_SUBSTITUTE=1
```

## 替换策略

开启后，每个 token 会先看原始 top-k expert。如果某些 top-k expert 不在显存热缓存中，则尝试用紧随其后的候选 expert 替换。

当前策略使用的候选范围是：

```text
top-(k+1) .. top-(2k)
```

以 Qwen3.5MoE 的 top-k=8 为例，就是用 rank 9 到 rank 16 中已经在显存的 expert，替换 top 1 到 top 8 中不在显存的 expert。

替换顺序：

1. 先找原始 top-k 里不在显存的 cold expert。
2. 从较低分的 top-k 位置开始替换，例如 top-8 优先于 top-1。
3. 候选 expert 必须已经在显存热缓存中。
4. 替换后，实际 routed expert id 会写回 `argsort_buffer` 的 top-k 区间。

## 门控权重处理

旧逻辑只替换 expert id，没有同步更新门控权重。这样会出现一个问题：实际计算使用的是替换后的 expert，但权重仍然对应被替换前的 expert。

现在开启替换策略时，会额外处理权重：

1. 从 GPU 读取完整 gating scores。这里的 gating score 是门控网络给每个 expert 的分数。
2. 从 GPU 读取当前 top-k weights。weights 是 top-k expert 当前使用的门控权重。
3. 如果某个 token 发生了 expert 替换，就用替换后 expert 对应的 gating score 重写该 token 的 top-k weights。
4. 对该 token 替换后的 top-k weights 重新归一化，使它们的和为 1。
5. 把更新后的 top-k weights 写回 GPU，后续 MoE FFN 计算会使用更新后的权重。

归一化公式：

```text
new_weight[i] = gating_score[selected_expert[i]] / sum(gating_score[selected_expert[0..k-1]])
```

这里的 normalization，中文就是“归一化”：把一组分数按总和缩放，使它们相加等于 1。

## 涉及的代码位置

- 命令行参数：`common/arg.cpp`
- common 参数：`common/common.h`、`common/common.cpp`
- llama context 参数：`include/llama.h`、`src/llama-context.cpp`
- Vulkan provider 开关：`ggml/include/ggml-vulkan.h`
- 替换与权重重写核心：`ggml/src/ggml-vulkan/ggml-vulkan.cpp`

核心函数：

```cpp
hot_expert_offload_process(...)
```

这个函数现在同时负责：

- 根据开关决定是否进行 top-k expert 替换。
- 统计替换后的 required experts。
- 对缺失 expert 执行 SSD load 和 eviction。
- 记录 expert 访问历史，用于 victim 选择。

## 验证

已通过以下编译验证：

```bash
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 -host_arch=x64 && cmake --build build-msvc-vulkan --config Release --target ggml-vulkan"
```

```bash
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 -host_arch=x64 && cmake --build build-msvc-vulkan --config Release --target llama-cli"
```

构建通过，只出现已有的 Vulkan/MSVC warning。

## 注意事项

这个开关会改变实际 routed experts，因此输出可能与原始 top-k 路径不同。建议实验时至少对比两组：

```bash
--ssd-expert-cache-experts 120
```

和：

```bash
--ssd-expert-cache-experts 120 --ssd-expert-cache-substitute
```

重点观察 tokens/s、SSD load 次数、eviction 次数、输出质量变化和后续要添加的 expert hit rate。
