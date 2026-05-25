# Chameleon 与早期融合纯 Token 多模态模型

> 到目前为止我们看到的每个 VLM 都将图像和文本分开处理。视觉 token 来自视觉编码器，经过投影器，然后在 LLM 内部与文本相遇。视觉和文本词表从不重叠。Chameleon（Meta, 2024年5月）问道：如果让它们重叠呢？训练一个 VQ-VAE，将图像转化为来自共享词表 (shared vocabulary) 的离散 token 序列。每个多模态文档现在都是一个序列——文本 token 和图像 token 交织，单一自回归损失。附带效果：模型可以生成混合模态输出——在单次推理调用中交替文本和图像 token。本课解读早期融合论题，并端到端构建一个简易版本。

**类型：** 构建
**语言：** Python（标准库，VQ-VAE 分词器 + 交织解码器）
**前置课程：** Phase 12 · 05，Phase 8（生成式 AI）
**时间：** ~180 分钟

## 学习目标

- 解释为什么共享词表 + 单一损失改变了模型能做的事情。
- 描述 VQ-VAE 如何将图像分词为与 transformer 下一 token 预测目标兼容的离散序列。
- 列出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 位置、LayerNorm 顺序。
- 比较 Chameleon 和 BLIP-2 的 Q-Former 方案，描述各自适用的场景。

## 问题所在

基于适配器的 VLM（LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两种不同的东西。文本 token 通过 `embed(text_token)` 处理；图像则通过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，在中途合并。

三个后果：

1. LLM 只能消费图像，不能输出图像。输出仅限文本。
2. 混合模态文档（如文章中交替出现的段落和图像）处理起来很别扭——你要么在模型外解析多模态输入，要么链式生成。
3. 分布不匹配。视觉 token 和文本 token 存在于隐藏空间的不同区域，产生微妙的对齐问题。

Chameleon 拒绝这个前提：图像就是来自共享词表的离散 token 序列。在交织文档上训练，一个损失，一个自回归解码器，免费解锁混合模态生成。

## 核心概念

### VQ-VAE 作为图像分词器

分词器是向量量化变分自编码器 (Vector-Quantized Variational Autoencoder)。架构如下：

- 编码器：CNN + ViT，将图像映射到空间特征图，例如 32×32 个 256 维特征。
- 码本 (Codebook)：K 个学习到的向量（Chameleon 使用 8192 个），同样 256 维。
- 量化：对每个空间特征，通过 L2 距离查找最近的码本条目。将连续特征替换为整数索引。
- 解码器：CNN，将量化后的特征还原为像素。

训练：VAE 重建损失 + 承诺损失 (commitment loss) + 码本损失 (codebook loss)。码本索引构成图像的离散字母表。

对于 Chameleon：一张图像变为 32×32 = 1024 个 token，取自 8192 大小的词表。与文本 token（来自 LLM 的 BPE 词表，假设 32000）拼接。最终词表：40192。Transformer 看到的是一个序列、一个损失。

### 共享词表

Chameleon 的词表结合了文本 token、图像 token 和模态分隔符。每个 token 有一个唯一 ID。输入嵌入层将每个 ID 映射到 D 维隐藏向量。输出投影将隐藏向量映射回词表 logits。Softmax 选出下一个 token，无论什么模态。

分隔符很重要：`<image>` 和 `</image>` 标签括住图像 token 序列。在生成时，如果模型输出 `<image>`，下游软件就知道接下来的 1024 个 token 是 VQ 索引，需要送入解码器进行像素渲染。

### 混合模态生成

推理就是在共享词表中做下一 token 预测。示例提示："画一只猫并描述它。"Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 个图像 token) </image>
The cat is orange, sitting on a windowsill...
```

模型自主选择顺序——可能先产生图像再产生文本，先文本再图像，或者交替产生。同一个解码器，同一个损失。

与只能文本输出的适配器 VLM 相比，Chameleon 重新打开了模型输出模态的问题。

### 训练稳定性 —— QK-Norm、dropout、LayerNorm 顺序

大规模的早期融合训练是不稳定的。Chameleon 论文记录了三个技巧：

- QK-Norm。在注意力内部对 query 和 key 投影应用 LayerNorm，在点积之前执行。防止深层处 logit 幅度爆炸。被 2024 年后的多个大模型采用。
- Dropout 位置。在每个残差连接之后都加 dropout，不仅仅在 attention 和 MLP 之后。当图像 token 的梯度可能主导时需要更多正则化。
- LayerNorm 顺序。残差分支上用 Pre-LN（标准做法），加上最后一个块的跳跃连接上额外一个 LN。稳定最终层的梯度流。

没有这些技巧，34B 参数的 Chameleon 训练在多个检查点发散。有了它们，才能收敛。训练方案与架构本身同样是贡献。

### 分词器的重建上限

VQ-VAE 是有损的。使用 8192 个码本条目和每张 512×512 图像 1024 个 token，重建 PSNR 大约在 26-28 dB。这对可辨认的图像生成足够，但明显不如连续空间扩散模型（Stable Diffusion 3 达到 32+ dB）。

分词器是瓶颈。更好的分词器（MAGVIT-v2、IBQ、SBER-MoVQGAN）抬高了上限。Emu3（Lesson 12.12）仅通过更好的分词器就实现了 SDXL 级质量的生成。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（早期融合，共享词表）：
- 一个损失，一个解码器。
- 生成混合模态输出。
- 分词器是质量上限。
- 昂贵：推理路径中每张生成图像都需要 VQ-VAE 解码器。

BLIP-2 / LLaVA（后期融合，分离塔）：
- 视觉输入，仅文本输出。
- 复用预训练 LLM。
- 理解方面无分词器瓶颈。
- 便宜：单次前向传播。

按任务选择。如果需要图像生成，选 Chameleon 家族。如果只需要理解，适配器 VLM 更简单且复用更多预训练计算。

### Fuyu 和 AnyGPT

Fuyu（Adept, 2023）是一种相关方案：完全跳过单独的视觉编码器，将原始图像 patch 通过 LLM 的输入投影层当作 token 输入，无需分词器。比 Chameleon 更简单，但失去了共享词表的输出生成能力。

AnyGPT（Zhan et al., 2024）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种模态使用相同的 VQ-VAE 技巧，共享 transformer。任意到任意生成。在 Lesson 12.16 中有更多介绍。

## 上手实践

`code/main.py` 构建一个简易的端到端早期融合模型：

- 一个小型 VQ-VAE 风格量化器，将 8×8 patch 映射到码本索引（K=16）。
- 共享词表：（文本 ID 0..31）+（图像 ID 32..47）+（分隔符 48, 49）。
- 一个简易自回归解码器（bigram 表），在合成描述 + 图像 token 序列上训练。
- 采样循环，给定提示输出交替的文本 + 图像 token。

代码刻意保持 transformer 极小（bigram），这样你可以端到端追踪信号流。

## 交付产出

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品规格（仅理解 vs 理解+生成、所需图像质量、成本预算），它在 Chameleon 家族（早期融合）和 LLaVA 家族（后期融合）之间做选择，并用量化经验法则给出理由。

## 练习

1. Chameleon 使用 K=8192 码本条目和每张 512×512 图像 1024 个 token。估算相对于 24 位 RGB 图像的压缩比。有损吗？有多大损失？

2. 一张 4K 图像（3840×2160）在相同 VQ-VAE 密度下产生多少图像 token？Chameleon 风格的模型能在一次推理调用中生成 4K 图像吗？什么先崩溃——上下文窗口、分词器质量还是 KV cache？

3. 用纯 Python 实现 QK-Norm。给定 64 维的 query 和 key，展示 LayerNorm 前后的点积。为什么幅度控制在深层很重要？

4. 阅读 Chameleon 第 2.3 节关于训练稳定性的内容。描述论文在 34B 无 QK-Norm 时观察到的确切失败模式。"范数爆炸"的特征是什么？

5. 扩展简易解码器，使其在给定纯文本提示时输出混合模态响应。测量在训练数据分布 60% 先文本 / 40% 先图像的情况下，模型多频繁地选择先图像 vs 先文本。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 早期融合 (Early fusion) | "统一 token" | 图像转换为离散 token，从第一步起就共享 transformer 的词表 |
| VQ-VAE | "图像分词器" | CNN + ViT + 码本，将图像映射为 transformer 可预测的整数索引 |
| 共享词表 (Shared vocabulary) | "同一本字典" | 一个 token ID 空间覆盖文本 + 图像 + 模态分隔符 |
| QK-Norm | "注意力稳定器" | 在点积之前对 query 和 key 应用 LayerNorm，防止范数爆炸 |
| 混合模态生成 (Mixed-modality generation) | "文本 + 图像输出" | 推理过程在单次传递中自主产出交织的文本和图像 token |
| 码本大小 (Codebook size) | "K 个条目" | VQ-VAE 可量化到的离散向量数量；在压缩和保真度之间权衡 |
| 分词器上限 (Tokenizer ceiling) | "重建极限" | 解码 VQ token 可达到的最佳 PSNR；决定了模型的图像质量上界 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
