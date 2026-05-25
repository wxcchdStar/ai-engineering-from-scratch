# 视觉Transformer与Patch令牌原语

> 在接触任何多模态之前，一张图像必须先变成一串 Transformer 能处理的令牌序列。2020 年的 ViT（视觉Transformer）论文用 16×16 像素的 patch（图像块）、一个线性投影和一个位置嵌入回答了这个问题。五年后，2026 年的每一款前沿模型（原生 2576px 的 Claude Opus 4.7、Gemini 3.1 Pro、Qwen3.5-Omni）仍然以此为基础——编码器从 ViT 演进到 DINOv2 再到 SigLIP 2，新增了寄存器令牌，位置编码方案变成了 2D-RoPE，但这个原语始终未变。本课将从头到尾阅读 patch 令牌流水线，并用标准库 Python 实现它，以便 Phase 12 的其余部分对"视觉令牌"有一个具体的思维模型。

**类型：** 学习
**语言：** Python（标准库，patch 分词器 + 几何计算器）
**前置要求：** Phase 7（Transformer）、Phase 4（计算机视觉）
**时间：** 约 120 分钟

## 学习目标

- 将一张 H×W×3 的图像转换为一串带有正确位置编码的 patch 令牌。
- 计算给定（patch 大小、分辨率、隐藏维度、深度）的 ViT 的序列长度、参数量和 FLOPs。
- 说出将 ViT 从 2020 年的研究推向 2026 年生产环境的三个升级：自监督预训练（DINO / MAE）、寄存器令牌和原生分辨率打包。
- 在下游任务中，在 CLS 池化、均值池化和寄存器令牌之间做出选择。

## 问题

Transformer 操作的是向量序列。文本本身就是序列（字节或令牌），而图像是一个带有三个颜色通道的 2D 像素网格——不是序列。如果把每个像素都展开，一张 224×224 的 RGB 图像会变成 150,528 个令牌，这种长度下的自注意力根本无法使用（与序列长度呈二次方关系）。

2020 年之前的方法是在前面接一个 CNN 特征提取器：ResNet 生成一个 7×7 的 2048 维向量特征图，将这 49 个令牌喂给 Transformer。这种方法可行，但继承了 CNN 的偏好（平移等变性、局部感受野），并且丧失了 Transformer 对规模的偏好。

Dosovitskiy 等人（2020）提出了一个直截了当的问题：如果跳过 CNN 呢？将图像分割成固定大小的 patch（比如 16×16 像素），将每个 patch 线性投影为一个向量，加上位置嵌入，然后把序列喂给一个普通的 Transformer。这在当时被认为是异端——没有卷积的视觉。但在足够多的数据（JFT-300M，后来是 LAION）上，它在 ImageNet 上击败了 ResNet，并且持续提升。

到 2026 年，ViT 原语已是无可争议的基础。每一个开源 VLM 的视觉塔都是某种后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题已不再是"是否应该使用 patch？"，而是"用多大的 patch 大小、什么样的分辨率调度、什么样的预训练目标、什么样的位置编码"。

## 概念

### Patch 即令牌

给定一张形状为 `(H, W, 3)` 的图像 `x` 和一个 patch 大小 `P`，你将图像划分为一个 `(H/P) × (W/P)` 的不重叠 patch 网格。每个 patch 是一个 `P × P × 3` 的像素立方体。将每个立方体展开为一个 `3 P^2` 维向量。应用一个形状为 `(3 P^2, D)` 的共享线性投影 `W_E`，将每个 patch 映射到模型的隐藏维度 `D`。

以 ViT-B/16 的典型配置为例：
- 分辨率 224，patch 大小 16 → 网格 14×14 → 196 个 patch 令牌。
- 每个 patch 为 `16 × 16 × 3 = 768` 个像素值，投影到 `D = 768`。
- 添加一个可学习的 `[CLS]` 令牌 → 序列长度 197。

patch 投影在数学上与核大小为 `P`、步长为 `P`、输出通道数为 `D` 的 2D 卷积完全相同。这就是生产代码实际实现的方式——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。"线性投影"的表述是概念层面的，核的表述才是高效的。

### 位置嵌入

Patch 没有固有顺序——Transformer 将它们视为一个集合。早期的 ViT 添加了可学习的 1D 位置嵌入（每个位置一个 768 维向量，共 197 个）。这可行，但将模型与训练分辨率绑定在一起：如果在推理时改变网格，就必须插值位置表。

现代视觉骨干网络使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或分解的 2D 位置编码。2D-RoPE 根据 patch 的（行、列）索引旋转查询和键向量，使模型从旋转角度推断出相对 2D 位置。无需位置表。模型可以在推理时处理任意网格大小。

### [CLS] 令牌、池化输出与寄存器令牌

什么是图像级别的表示？三种选择并存：

1. `[CLS]` 令牌。在 patch 序列前添加一个可学习的向量。经过所有 Transformer 块后，[CLS] 令牌的隐藏状态就是图像表示。继承自 BERT。原始 ViT、CLIP 使用。
2. 均值池化。对 patch 令牌的输出隐藏状态取平均值。SigLIP、DINOv2 以及大多数现代 VLM 使用。
3. 寄存器令牌。Darcet 等人（2023）观察到，在没有显式接收令牌（sink token）的情况下训练的 ViT 会产生高范数的"伪影"patch，劫持自注意力。添加 4–16 个可学习的寄存器令牌可以吸收这种负载，并改善密集预测质量（分割、深度估计）。DINOv2 和 SigLIP 2 都内置了寄存器。

这一选择对下游任务至关重要。[CLS] 适合分类任务。对于将 patch 令牌喂给 LLM 的 VLM，你完全跳过池化——每个 patch 都成为 LLM 的输入令牌。寄存器令牌在交接前被丢弃（它们是脚手架，而非内容）。

### 预训练：有监督、对比、掩码、自蒸馏

2020 年的 ViT 在 JFT-300M 上进行有监督分类预训练。很快被以下方法取代：

- CLIP（2021）：在 4 亿对图像-文本上进行对比学习。第 12.02 课。
- MAE（2021，He 等人）：掩码 75% 的 patch，重建像素。自监督，仅使用纯图像即可工作。
- DINO（2021）/ DINOv2（2023）：使用学生-教师自蒸馏，无需标签、无需标题。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉骨干网络，也是"密集特征"用例的默认选择。
- SigLIP / SigLIP 2（2023、2025）：采用 sigmoid 损失和 NaFlex（原生弹性分辨率）的 CLIP。是 2026 年开源 VLM（Qwen、Idefics2、LLaVA-OneVision）中的主导视觉塔。

预训练方式的选择决定了骨干网络适合什么：CLIP/SigLIP 适合与文本进行语义匹配，DINOv2 适合密集视觉特征，MAE 适合作为下游微调的起点。

### 缩放定律

ViT 缩放定律（Zhai 等人，2022）确立了 ViT 的质量在模型规模、数据规模和计算量方面遵循可预测的规律。在固定计算量下：
- 更大的模型 + 更多数据 → 更好的质量。
- patch 大小是序列长度与保真度之间的调节杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型值）比 patch 16 产生更多每图像令牌；更适合 OCR 和密集任务，但速度更慢。
- 分辨率是另一个重要的调节杠杆。从 224 提升到 384 再到 512 几乎总是有帮助的，但 FLOPs 会呈二次方增长。

ViT-g/14（10 亿参数，patch 14，分辨率 224 → 256 个令牌）和 SigLIP SO400m/14（4 亿参数，patch 14）是 2026 年开源 VLM 的两款主力编码器。

### ViT 的参数量

完整计算见 `code/main.py`。以 ViT-B/16（分辨率 224）为例：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载检查点之前，用这种方式估算每一个 ViT。骨干网络的大小决定了任何下游 VLM 的显存下限。

### 2026 年生产配置

2026 年大多数开源 VLM 搭载的编码器是原生分辨率（NaFlex）下的 SigLIP 2 SO400m/14。它具有：
- 4 亿参数。
- patch 大小 14，默认分辨率 384 → 每图像 729 个 patch 令牌。
- 图像级任务使用均值池化；所有 729 个 patch 流入 LLM 进行 VQA。
- 4 个寄存器令牌，在 LLM 交接前丢弃。
- 带有图像级缩放的 2D-RoPE，支持原生宽高比。

该配置中的每一项决策都可以追溯到一篇你可以阅读的论文。

## 使用指南

`code/main.py` 是一个 patch 分词器与几何计算器。它接收（图像 H, W, patch P, hidden D, depth L）并输出：

- 分块后的网格形状与序列长度。
- 一个合成 8×8 像素玩具图像的 token 序列（遍历 flatten + project 路径）。
- 按 patch embed、position embed、transformer block 和 head 拆分的参数量。
- 目标分辨率下单次前向传播的 FLOPs。
- 跨 ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。将参数量与已发表的数字进行比对。调整 patch 大小和分辨率，感受 token 数量带来的成本变化。

## 产出物

本课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（patch 大小、分辨率、隐藏维度、深度），它会产出 token 数量、参数量和显存估算，并附有推导说明。在为 VLM 选择视觉骨干时使用此 skill——它能避免"token 爆炸导致 LLM 上下文溢出"的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280×720 输入、patch 大小 14 下的 patch-token 序列长度。与仅使用 CLS 的表示相比如何？

2. 一个 1080p 帧（1920×1080）在 patch 14 下产生多少个 token？以 30 FPS 播放 5 分钟视频，总共产生多少视觉 token？哪种方案最省钱：pooling、帧采样还是 token 合并？

3. 用纯 Python 实现 patch token 的均值 pooling。验证对 196 个 DINOv2 输出 token 做 mean-pool 的结果与模型 `forward` 返回的 pooled embedding 一致。

4. 阅读 "Vision Transformers Need Registers" (arXiv:2309.16588) 的第 3 节。用两句话描述 register 吸收了哪些 artifact，以及这对下游密集预测为何重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一个不同分辨率图像的列表，生成单个 packed 序列和块对角注意力掩码。到第 12.06 课时验证你的实现。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| Patch | "16×16 像素方块" | 输入图像中固定大小、不重叠的区域；每个区域变成一个 token |
| Patch embedding | "线性投影" | 一个共享的可学习矩阵（或步长为 P 的 Conv2d），将展平的 patch 像素映射为 D 维向量 |
| CLS token | "分类 token" | 前置的可学习向量，其最终隐藏状态代表整张图像；在 2026 年已非必需 |
| Register token | "sink token" | 额外的可学习 token，用于吸收 ViT 在预训练过程中产生的高范数注意力 artifact |
| Position embedding | "位置信息" | 每个位置对应的向量或旋转编码，使序列具有顺序感知能力；2D-RoPE 是现代默认方案 |
| Grid | "patch 网格" | 给定分辨率和 patch 大小下，大小为 (H/P) × (W/P) 的 2D patch 阵列 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的特性：单个模型无需重新训练即可支持多种宽高比和分辨率 |
| Backbone | "视觉塔" | 预训练的图像编码器，其 patch-token 输出作为 VLM 中 LLM 的输入 |
| Pooling | "图像级摘要" | 将 patch token 转化为单一向量的策略：CLS、均值、注意力池化或基于 register 的方式 |
| Patch 14 vs 16 | "更细 vs 更粗的网格" | Patch 14 每张图像产生更多 token，OCR 保真度更好但速度更慢；patch 16 是经典默认值 |

## 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无需标签。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token 与 artifact 分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年的默认视觉塔。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验性缩放定律。
