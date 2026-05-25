# Janus-Pro：统一多模态模型的解耦编码器

> 统一多模态模型有一个不可避免的张力。理解需要语义特征——SigLIP 或 DINOv2 输出的富含概念级信息的向量。生成需要适合重建的编码——能组合回清晰像素的 VQ token。这两个目标在单一编码器中不兼容。Janus（DeepSeek, 2024年10月）和 Janus-Pro（DeepSeek, 2025年1月）认为解决办法是停止尝试：解耦两个编码器。在任务间共享 transformer 主体，但理解走 SigLIP 路由，生成走 VQ 分词器路由。在 7B 参数下，Janus-Pro 在 GenEval 上超越 DALL-E 3，同时在 MMMU 上匹配 LLaVA。本课解读为什么两个编码器在一个失败的地方成功。

**类型：** 构建
**语言：** Python（标准库，双编码器路由 + 共享主体信号）
**前置课程：** Phase 12 · 13（Transfusion），Phase 12 · 14（Show-o）
**时间：** ~120 分钟

## 学习目标

- 解释为什么单一共享编码器会在理解或生成质量上妥协。
- 描述 Janus-Pro 的路由：输入侧的 SigLIP 特征用于理解，VQ token 在输入和输出两侧用于生成。
- 追踪使 Janus-Pro 成功而 Janus 未能成功的数据混合缩放。
- 比较解耦（Janus-Pro）、耦合连续（Transfusion）和耦合离散（Show-o）架构。

## 问题所在

统一模型在理解和生成之间共享 transformer 主体。之前的尝试（Chameleon、Show-o、Transfusion）都对两个方向使用一个视觉分词器。分词器是一种折中：

- 为重建优化（生成方向）：VQ-VAE 捕获细粒度像素细节但产出的 token 语义连贯性弱。
- 为语义优化（理解方向）：SigLIP 嵌入将"猫"的图像聚集在"猫"token 附近，但不允许良好的重建。

Show-o 和 Transfusion 为此在某个方向上付出了可见的质量税。Janus-Pro 问道：当两个任务有不同的需求时，为什么要求使用一个分词器？

## 核心概念

### 解耦视觉编码

Janus-Pro 的架构分离了两个编码器：

- 理解路径。输入图像 → SigLIP-SO400m → 2 层 MLP → transformer 主体。
- 生成路径。输入图像（如果以现有图像为条件）→ VQ 分词器 → token ID → transformer 主体。
- 输出生成。Transformer 预测的图像 token → VQ 解码器 → 像素。

Transformer 主体是共享的。主体上下游的一切都是任务特定的。

输入通过提示格式消歧：`<understand>` 标签路由到 SigLIP；`<generate>` 路由到 VQ。或者路由从任务隐式推断。

### 为什么这样有效

理解损失获得 SigLIP 特征，CLIP 风格预训练已将其调优为语义相似性。模型的感知基准超过 Show-o / Transfusion，因为输入特征更适合该任务。

生成损失获得 VQ token，分词器已将其调优为重建。图像质量超过 Show-o，因为 VQ 码能干净地组合回像素。

共享的 transformer 主体看到两种输入分布（SigLIP 和 VQ）并学会与两者协作。主张是：足够的数据 + 足够的参数，主体就能吸收这种切换。

### 数据缩放 —— Janus vs Janus-Pro

Janus（原版，arXiv 2410.13848）引入了解耦但规模小（1.3B 参数，有限数据）。Janus-Pro（arXiv 2501.17811）进行了缩放：

- 7B 参数（vs 1.3B）。
- 阶段 1（对齐）9000 万图文对，从 7200 万增加。
- 阶段 2（统一）7200 万，从 2600 万增加。
- 阶段 3 增加了 20 万图像生成指令样本。

结果：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs ~58），在 GenEval 上超越 DALL-E 3（0.80 vs 0.67）。一个开源模型，在统一光谱两侧都具有竞争力。

### JanusFlow —— 修正流变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径换为修正流 (rectified-flow) 生成路径（连续）。分离变为 SigLIP 用于理解 + 修正流用于生成。质量上限进一步提升。架构仍然是解耦编码器-共享主体。

### 共享主体的职责

Transformer 主体处理统一序列但有两种输入分布。其职责是：

- 理解：消费 SigLIP 特征 + 文本 token → 自回归输出文本。
- 生成：消费文本 token +（可选的图像 VQ token）→ 自回归输出图像 VQ token。

主体在每个块中没有模态特定权重。它就是你在 Qwen 或 Llama 内部找到的文本风格 transformer，加上两个输入适配器。

有趣的是，这意味着 Janus-Pro 的主体可以从预训练 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这个选择很重要：LLM 贡献了纯从头训练的统一模型难以达到的推理能力。

### 与 InternVL-U 的对比

InternVL-U（Lesson 12.10）是 2026 年的后续。它结合了：

- 原生多模态预训练（InternVL3 骨干）。
- 解耦编码器路由（SigLIP 输入，VQ + 扩散头输出）。
- 统一理解 + 生成 + 编辑。

InternVL-U 将 Janus-Pro 的架构选择纳入了更大的框架。解耦编码器的想法现在是大规模统一模型的默认选择。

### 局限性

解耦编码器增加了架构复杂性。两个分词器要训练，两条输入路径要维护，两组失败模式。对于不需要生成的产品，Janus-Pro 过度工程化了——选一个 LLaVA 家族理解模型即可。

对于不需要理解的产品，Janus-Pro 资质过高——选一个 Stable Diffusion 3 / Flux 模型即可。

对于两者都需要的产品，Janus-Pro 现在是参考开源架构。

## 上手实践

`code/main.py` 模拟 Janus-Pro 路由：

- 两个模拟编码器：类 SigLIP（产出 256 维语义向量）和类 VQ（产出整数码）。
- 基于任务标签选择编码器的提示路由器。
- 无论哪个编码器产出 token 序列，共享主体（替代品）都处理它们。
- 从阶段 1（对齐）到阶段 3（指令微调）的加权采样调度切换。

打印 3 个示例的路由路径：图像问答、文生图、图像编辑。

## 交付产出

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定需要在前沿水平同时具备生成 + 理解的产品，在 Janus-Pro、JanusFlow 或 InternVL-U 之间做选择，并给出具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上超越 DALL-E 3。解释为什么一个 7B 开源模型能在生成上匹配前沿专有模型，但在理解上不能。

2. 实现一个路由函数：给定提示文本，分类为 `understand` 或 `generate`。如何处理"描述然后画一幅"这类模糊提示？

3. JanusFlow 将 VQ 路径替换为修正流。Transformer 主体现在输出什么，损失有什么变化？

4. 提出一个 Janus-Pro 架构可以通过再增加一个解耦编码器来处理的第四种任务。例如：图像分割（DINO 风格）、深度估计（MiDaS 风格）。

5. 阅读 Janus-Pro 第 4.2 节关于数据缩放。哪个数据阶段对 T2I 质量提升（相比 Janus）贡献最大？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 解耦编码 (Decoupled encoding) | "两个视觉编码器" | 每个方向使用独立的分词器或编码器：语义用于理解，重建用于生成 |
| 共享主体 (Shared body) | "一个 transformer" | 单个 transformer 处理任一编码器的输出；无模态特定权重 |
| SigLIP 用于理解 | "语义特征" | CLIP 家族视觉塔提供丰富的概念特征但重建能力差 |
| VQ 用于生成 | "重建码" | 能干净地解码回像素的向量量化 token |
| JanusFlow | "修正流变体" | 用连续流匹配生成头替代 VQ 的 Janus-Pro |
| 路由标签 (Routing tag) | "任务标签" | 选择输入编码器的提示标记（`<understand>` / `<generate>`） |

## 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
