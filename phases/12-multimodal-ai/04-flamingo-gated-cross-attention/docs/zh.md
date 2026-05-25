# Flamingo 与用于少样本 VLM 的门控交叉注意力

> DeepMind 的 Flamingo（2022）做了两件前无古人的事。它证明了单个模型可以处理任意交错的图像、视频和文本序列。它还证明了 VLM 可以进行上下文学习——给定一个少样本提示，包含三个示例（图像, 描述）对，模型无需任何梯度更新就能为一张新图像生成描述。其机制是：在冻结 LLM 的现有层之间插入门控交叉注意力层，并使用一个初始值为零的可学习 tanh 门控，从而在初始化时保留 LLM 的文本能力。本课将讲解 Flamingo 的 Perceiver 重采样器和门控交叉注意力架构——这是 Gemini 交错输入和 Idefics2 视觉 token 的鼻祖。

**类型：** 学习
**语言：** Python（标准库，门控交叉注意力 + Perceiver 重采样器演示）
**前置课程：** Phase 12 · 03（BLIP-2 Q-Former）
**时间：** 约 120 分钟

## 学习目标

- 解释门控交叉注意力如何通过 tanh(gate) = 0 在初始化时保留冻结 LLM 的文本能力。
- 逐步讲解 Perceiver 重采样器：N 个图像块通过交叉注意力 → K 个固定的"潜在"查询。
- 描述 Flamingo 如何处理交错图像-文本序列，并使用尊重图像位置关系的因果掩码。
- 复现一个少样本多模态提示结构（3 个图像-描述示例，然后是一张查询图像）。

## 问题

BLIP-2 将 32 个视觉 token 送入冻结 LLM 的输入层。这对于每个提示只有一张图像的情况是可行的。但如果你想送入*多张*与文本交错的图像呢？比如"这是图像 A，为它写描述；这是图像 B，为它写描述；现在这是图像 C，为它写描述"？LLM 的自注意力需要在一个单一序列中同时处理图像 token 和文本 token，而哪些位置可以关注哪些图像的问题就会变得很棘手。

Flamingo 的答案是：完全不去改变 LLM 的输入流。而是在现有的 LLM 块之间插入额外的交叉注意力层。文本 token 仍然像往常一样流经 LLM 的因果自注意力。每隔几个 LLM 块，文本 token 还会通过一个新的门控层对图像特征进行交叉注意力计算。门控（初始化为零）意味着在第零步时，新层是无操作层——模型的行为与预训练 LLM 完全一致。随着训练的进行，门控逐渐打开，视觉信息开始流入。

Flamingo 回答的第二个问题是：如何处理每个提示中可变数量的图像（0 张、1 张或多张）？答案是 Perceiver 重采样器——一个小的交叉注意力模块，它接收任意数量的图像块，并产生固定数量的视觉潜在 token。LLM 的交叉注意力层看到的形状始终相同，无论提示中有多少张图像。

## 概念

### 冻结 LLM

Flamingo 从一个冻结的 Chinchilla 70B LLM 开始。所有 70B 权重保持不变。现有的文本自注意力和 FFN 正常运作。

### Perceiver 重采样器

对于提示中的每张图像，ViT 产生 N 个块 token。Perceiver 重采样器有 K 个固定的可学习潜在向量（Flamingo 使用 K=64）。每个重采样器块包含两个子步骤：

1. 交叉注意力：K 个潜在向量对 N 个块 token 进行注意力计算（Q 来自潜在向量，K/V 来自块）。
2. 潜在向量内部的自注意力 + FFN。

经过 6 个重采样器块后，输出是 K=64 个维度为 1024 的视觉 token，无论 ViT 产生了多少个块。一张 224×224 的图像（196 个块）和一张 480×480 的图像（900 个块）输出都是 64 个重采样器 token。

对于视频，重采样器在时间维度上应用：每一帧的块产生 64 个潜在向量，时间位置编码让模型能够区分 t=0 和 t=N。整个视频变为 T × 64 个视觉 token。

### 门控交叉注意力

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的门控交叉注意力块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个可学习的标量，初始化为零。
- `tanh(0) = 0`，因此在初始化时门控分支的贡献为零。
- 随着 `alpha` 远离零，交叉注意力的贡献平滑增长。
- 残差连接意味着即使门控完全打开，也不会覆盖 LLM 的文本表示；它只是在上面叠加视觉信息。

这是 Flamingo 中最重要的设计选择：视觉条件化是可加的、门控的，且在初始化时为零。第 0 步的 Flamingo 在纯文本输入上就是一个完美的 Chinchilla 70B。

### 交错输入的掩码交叉注意力

在一个提示中，如"<图像 A> 描述 A <图像 B> 描述 B <图像 C> ?"，每个文本 token 只应看到在序列中位于它之前的图像。交叉注意力掩码强制执行：位置 `t` 的文本 token 只关注图像重采样器 token，且这些 token 的图像索引 `i < i_t`，其中 `i_t` 是位置 `t` 之前最近的图像索引。"只看到最后一张前置图像"或"看到所有前置图像"都是有效的选择；Flamingo 选择了前者。

### 上下文少样本学习

Flamingo 提示如下所示：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到补全模式并输出 "bird"（或者 image3 显示的任何内容）。无需梯度更新。冻结 LLM 的上下文学习能力通过门控交叉注意力得以延续——这是该论文的核心亮点，也是它如此重要的原因。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb (M3W)：4300 万个包含交错图像和文本的网页，重建阅读顺序。
2. Image-Text Pairs（ALIGN + LTIP）：44 亿对。
3. Video-Text Pairs (VTP)：2700 万个短视频片段。

OBELICS（2023）是交错网页语料库的开放复现版本，Idefics、Idefics2 以及大多数开源"类 Flamingo"模型都在此之上训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开放复现版本。架构完全相同（在冻结的 LLaMA 或 MPT 上使用 Perceiver 重采样器 + 门控交叉注意力）。提供了 3B、4B、9B 的检查点。由于基础 LLM 更小且数据更少，质量不如 Flamingo。

Otter（2023）在 OpenFlamingo 的基础上进行了 MIMIC-IT（一个多模态指令数据集）上的指令微调，证明了门控交叉注意力同样适用于指令跟随。

### 后继者

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力谱系，逐步简化（Idefics2 去掉了重采样器，改用带自适应池化的直接块 token）。
- Flamingo 到 Chameleon 的过渡：到 2024 年，许多团队转向了早期融合（第 12.11 课）；Flamingo 风格的门控交叉注意力在需要冻结主干的场景中仍然在生产中使用。
- Gemini 的交错输入：在概念上继承了 Flamingo 的交错格式灵活性，尽管具体机制是专有的。

### 与 BLIP-2 的比较

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥接 | Q-Former 在输入层一次完成 | 每 M 层的门控交叉注意力 |
| 视觉 token | 每张图像 32 个 | 每张图像每层交叉注意力 64 个 |
| 冻结 LLM | 是 | 是 |
| 少样本上下文学习 | 弱 | 强——论文的核心亮点 |
| 交错输入 | 不支持 | 是，设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错页面 |
| 训练参数量 | 1.88 亿 | 约 100 亿（交叉注意力层） |
| 计算量 | 8 张 A100 上数天 | 数千张 TPUv4 上数周 |

预算有限时选择 BLIP-2 进行单图像 VQA。需要交错、少样本或多图像推理时选择 Flamingo/Idefics2。

## 使用说明

`code/main.py` 演示了以下内容：

1. 一个 Perceiver 重采样器，对 36 个假 patch token 使用 8 个可学习潜在向量（纯 Python 交叉注意力实现）。
2. 一个门控交叉注意力步骤，其中 `alpha = 0` → 输出等于输入（LLM 不变），然后 `alpha = 2.0` → 视觉贡献被混入。
3. 一个交错掩码构建器，为 "(image 1) (text 1) (image 2) (text 2)" 序列生成二维注意力掩码。

## 交付物

本课生成 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开放 VLM 的配置（是否使用重采样器、交叉注意力频率、门控方案），它会识别出 Flamingo 系谱元素并解释冻结策略。可用于调试为什么微调会降低文本性能（答案：门控打开得太快太宽）。

## 练习

1. 计算 Flamingo-9B 的视觉参数数量：9B LLM + 1.4B 门控交叉注意力层 + 64M 重采样器。训练参数占总参数的比例是多少？

2. 在 PyTorch 中实现门控残差 `y = tanh(alpha) * cross + x`。通过实验证明，当 `alpha=0` 时，初始化时 `y==x` 精确成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390），了解当每个提示包含不同数量的图像时，他们如何处理批次中的多张图像。描述填充策略。

4. 为什么 Flamingo 的交叉注意力掩码允许文本 token 只关注*最近一张*前置图像，而不是所有前置图像？阅读 Flamingo 论文第 2.4 节并解释其中的权衡。

5. 上下文少样本学习：为一个新的 Flamingo 变体构造一个包含 4 个 "图像 → 主要物体颜色" 示例的提示。描述当示例数量从 0 变化到 8 时，预期的准确率变化规律。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| Perceiver resampler | "固定潜在交叉注意力" | 从可变数量的输入 patch 中产生 K 个固定 token 的模块 |
| Gated cross-attention | "Tanh 门控桥接" | 残差层 `y = tanh(alpha)*cross + x`，alpha 可学习，初始化为 0 |
| Interleaved input | "混合序列" | 图像和文本按阅读顺序自由交错的提示格式 |
| Frozen LLM | "不计算 LLM 梯度" | 文本 LLM 的权重不更新；仅训练重采样器和交叉注意力层 |
| Few-shot | "上下文示例" | 在提示中提供少量（图像, 答案）对；模型无需微调即可泛化 |
| OBELICS | "交错网页语料库" | 包含 1.41 亿个按阅读顺序排列的图像和文本网页的开放数据集 |
| Chinchilla | "70B 冻结基座" | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| Gate schedule | "alpha 如何变化" | 训练过程中交叉注意力门控打开的速度 |
| Cross-attn frequency | "每 M 层" | 门控交叉注意力块插入的频率；Flamingo 使用 M=4 |
| OpenFlamingo | "开源复现" | MosaicML/LAION 的 3-9B 开源检查点；架构与 Flamingo 完全相同 |

## 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网页语料库。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 指令微调的 Flamingo 衍生模型。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化版本。
