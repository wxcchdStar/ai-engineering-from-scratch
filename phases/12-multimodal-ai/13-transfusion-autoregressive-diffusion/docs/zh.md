# Transfusion：自回归文本 + 扩散图像统一在一个 Transformer 中

> Chameleon 和 Emu3 把一切赌注押在离散 token 上。它们可行，但量化瓶颈可见——图像质量停滞在连续空间扩散模型之下。Transfusion（Meta, Zhou et al., 2024年8月）下了相反的赌注：保持图像连续，完全丢弃 VQ-VAE，训练一个带两种损失的 transformer。文本 token 用下一 token 预测 (NTP)。图像 patch 用流匹配/扩散损失 (flow-matching / diffusion loss)。两个目标优化相同的权重。Stable Diffusion 3 底层架构 MMDiT 是其近亲。本课解读 Transfusion 论题，构建一个简易双损失训练器，并追踪让一个 transformer 同时完成两项任务的注意力掩码。

**类型：** 构建
**语言：** Python（标准库，MNIST 级简易双损失训练器）
**前置课程：** Phase 12 · 11（Chameleon），Phase 8（生成式 AI）
**时间：** ~180 分钟

## 学习目标

- 编写一个在同一骨干上运行两种损失（文本 token 上的 NTP、图像 patch 上的扩散 MSE）的 transformer。
- 解释为什么图像 patch 间双向注意力加文本 token 上因果注意力是正确的掩码选择。
- 比较 Transfusion 风格（连续图像，扩散损失）与 Chameleon 风格（离散图像，NTP）在计算量、质量和代码复杂度上的差异。
- 说明 MMDiT 的贡献：每个块中模态特定的权重，残差流上的联合注意力。

## 问题所在

离散 vs 连续图像 token 的争论比 LLM 更古老。连续表示（原始像素、VAE 潜在表示）保留细节。离散 token（VQ 索引）适配 transformer 的原生词表但在量化步骤丢失细节。

Chameleon / Emu3 走了离散路线：一个损失、一个架构，但图像保真度受限于分词器质量。

扩散模型走了连续路线：出色的图像质量，但与 LLM 是独立模型，复杂的噪声调度工程，且无法与文本生成干净集成。

Transfusion 问道：能两者兼得吗？保持图像连续，仍然训练一个模型，将两种损失缝合在一个梯度步中。

## 核心概念

### 双损失架构

一个纯解码器 transformer 处理的序列包含：

- 文本 token（离散，来自 BPE 词表）。
- 图像 patch（连续，16×16 像素块通过线性嵌入投影到隐藏维度——与 ViT 编码器的输入相同）。
- `<image>` 和 `</image>` 标签标记连续 patch 所在位置。

前向传播运行一次。损失按 token 选择两个头之一：

- 对于文本 token：标准交叉熵，在词表 logits 头上。
- 对于图像 patch：连续 patch 上的扩散损失——预测添加到每个 patch 上的噪声。

梯度流过共享的 transformer 主体。两种损失同时改善共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 token 必须是因果的——不能让文本 token 注意到未来的文本，否则教师强制会被破坏。然而，图像 patch 代表一个快照；它们应在同一图像块内双向相互注意。

掩码：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # 文本因果
  OR (i is image and j is image and same_image_block(i, j))   # 图像块内双向
  OR (i is text and j is image and j < i_image_end)   # 文本注意到之前的图像
  OR (i is image and j is text and j < i_image_start)   # 图像注意到之前的文本
```

实现为训练和推理时的块三角形掩码。

### Transformer 内部的扩散损失

扩散损失是标准的：给图像 patch 加噪声，让模型预测噪声（或等价地预测干净 patch）。Transfusion 的版本使用流匹配 (flow matching)——预测从噪声到干净的速度场。

训练时：
1. 对每个图像 patch x0，采样随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（流匹配的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列中的文本 NTP 损失一起反向传播。

推理时的生成：
- 文本 token：标准自回归采样。
- 图像 patch：扩散采样循环（典型 10-30 步），以前面的文本 token 为条件。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser et al., 2024年3月）几乎与 Transfusion 同时发布了 MMDiT（多模态扩散 Transformer）。两者是姊妹架构。

MMDiT 的关键差异：

- 每个块有模态特定权重。每个 transformer 块对文本 token 和图像 patch 有独立的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其他都是模态特定的。
- 修正流训练 (Rectified flow)。一种特定的流匹配变体，具有已知的采样过程和比 DDPM 更简单的数学。
- 规模。MMDiT 是 SD3（2B 和 8B 参数变体）的骨干。Transfusion 论文规模到 7B。

两者收敛于相同的核心思想：一个 transformer 在文本上跑 NTP，在连续图像表示上跑扩散。

### 为什么优于 Chameleon 风格

连续扩散与离散 NTP 在图像生成上的质量差距是可量化的。Transfusion 论文报告：

- 在 7B 参数下，FID 比同规模的 Chameleon 风格模型好 3-5 分。
- 无需训练分词器——图像编码器更简单（线性投影到隐藏维度，与 ViT 输入层相同）。
- 推理可以并行化图像 patch 去噪，不像自回归图像 token 那样串行。

缺点：Transfusion 是双损失模型，训练动态更棘手。损失权重需要调节。NTP 和扩散之间的调度不匹配可能导致一个头主导。

### 后续发展

Janus-Pro（Lesson 12.15）精化了 Transfusion 的想法，将理解和生成的视觉编码器解耦——SigLIP 用于理解，VQ 用于生成——同时共享 transformer 主体。Show-o（Lesson 12.14）将扩散换成离散扩散（掩码预测）。统一生成家族在 Transfusion 之后迅速分支。

2026 年能输出图像的生产 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎可以确定使用了这个家族的某种后代。细节是专有的。

## 上手实践

`code/main.py` 在一个简易的类 MNIST 问题上构建了一个简易 Transfusion：

- 文本描述是描述数字（0-9）的短整数序列。
- 图像是 4×4 字节网格。
- 一对共享权重的线性投影充当 transformer 替代；文本上 NTP 损失，噪声 patch 上 MSE 损失。
- 训练循环交替两种损失，注意力掩码是显式的。
- 生成在一次前向传播中产出文本描述和 4×4 图像。

Transformer 是简易版。双损失管道、注意力掩码构造和推理循环才是真正的产出物。

## 交付产出

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定新的多模态训练任务（文本+图像、文本+音频、文本+视频），它设计双损失调度（损失权重、掩码形状、共享 vs 模态特定块）并标记实现风险。

## 练习

1. 一个 Transfusion 风格模型训练 70% 文本 token 和 30% 图像 patch。图像扩散损失的幅度约为文本 NTP 损失的 10 倍。什么损失权重能平衡它们？

2. 为以下序列实现块三角形掩码：`[T, T, <image>, P, P, P, P, </image>, T]`。标记每个条目为 0 或 1。

3. MMDiT 有模态特定的 QKV 权重。相比 Transfusion 的完全共享 transformer，这增加了多少参数量开销？在 7B 参数下，这值得吗？

4. 生成：给定一个文本提示，模型先跑 50 个 token 的 NTP，然后遇到 `<image>`，然后在 256 个 patch 上跑 20 步去噪的扩散。总共多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述修正流 (rectified flow) 以及为什么它比 DDPM 需要更少的推理步数就能收敛。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 双损失训练 (Two-loss training) | "NTP + 扩散" | 一个 transformer 在同一梯度步中同时优化文本 token 上的交叉熵和连续图像 patch 上的 MSE |
| 流匹配 (Flow matching) | "修正流" | 预测从噪声到干净数据的速度场的扩散变体；数学比 DDPM 更简单 |
| MMDiT | "多模态 DiT" | Stable Diffusion 3 的架构：联合注意力，模态特定的 MLP 和归一化 |
| 块三角形掩码 (Block-triangular mask) | "因果文本 + 双向图像" | 跨文本因果、图像区域内双向的注意力掩码 |
| 连续图像表示 (Continuous image representation) | "无 VQ" | 图像 patch 作为实值向量，而非整数码本索引 |
| 速度预测 (Velocity prediction) | "v 参数化" | 网络输出的是噪声和数据之间的速度场，而非噪声本身 |

## 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
