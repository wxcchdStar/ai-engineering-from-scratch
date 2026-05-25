# Emu3：用下一 Token 预测实现图像和视频生成

> BAAI 的 Emu3（Wang et al., 2024年9月）是 2024 年本应终结扩散 vs 自回归争论的成果。一个单一的 Llama 风格纯解码器 transformer，仅通过下一 token 预测 (next-token prediction) 目标进行训练，跨文本 + VQ 图像 token + 3D VQ 视频 token 的统一词表，在图像生成上超越 SDXL，在感知上超越 LLaVA-1.6。无 CLIP 损失。无扩散调度。推理时使用无分类器引导 (classifier-free guidance) 来提升质量，但核心训练目标就是带教师强制 (teacher forcing) 的下一 token 预测。发表于 Nature。本课解读 Emu3 的论题——为什么更好的分词器加上规模就是你所需的一切——并与扩散方法做对比。

**类型：** 学习
**语言：** Python（标准库，3D 视频分词器数学 + 自回归采样器骨架）
**前置课程：** Phase 12 · 11（Chameleon）
**时间：** ~120 分钟

## 学习目标

- 解释为什么 Emu3 的单损失下一 token 目标能够奏效，尽管长期以来人们认为图像质量需要扩散。
- 描述 3D 视频分词器：时空 VQ 码本 (spatiotemporal VQ codebook) 是什么样的，为什么 patch 跨越时间。
- 在（训练计算量、推理成本、质量上限）维度上比较 Emu3 与 Stable Diffusion XL。
- 列出同一个 Emu3 模型扮演的三种角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题所在

2024 年之前的传统观点：图像生成需要扩散。论据是：离散图像 token 丢失太多信息无法重建细节，自回归采样在数千个 token 上累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种形式的扩散。Chameleon（Lesson 12.11）在小规模上部分反驳了这一点，但未能在质量上匹配 SDXL。

Emu3 正面进攻这个论点。其主张：更好的视觉分词器 + 足够的规模 + 下一 token 损失 = 在同一个模型中实现超越扩散的图像生成，同时该模型还能做感知。

这个赌注在发表时存在争议。两年后，开源的统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）已成为研究的默认路径；前沿生产模型似乎使用了某种变体。

## 核心概念

### Emu3 分词器

关键成分是视觉分词器。Emu3 训练了一个自定义的 IBQ 类分词器（逆瓶颈量化器 (Inverse Bottleneck Quantizer)，SBER-MoVQGAN 家族），每个 token 对应 8×8 的分辨率缩减。一张 512×512 的图像变为 64×64 = 4096 个 token，码本大小 32768。

这比 Chameleon 的每张 512×512 图像 1024 token（K=8192）多，但每 token 更便宜（更小的码本查找、更简单的编解码器）。关键指标：重建 PSNR 为 30.5 dB，与 Stable Diffusion 连续潜在空间的 32 dB 竞争。

对于视频：3D VQ 分词器将一个时空 patch（4×4×4 像素）编码为一个整数。一个 4 秒片段以 8 FPS 有 32 帧；在 256×256 分辨率配合 4× 空间缩减和 4× 时间缩减下，token 数为 (256/4) × (256/4) × (32/4) = 64 × 64 × 8 = 32,768 个 token。

分词器质量就是上限。Emu3 的贡献部分在于"我们训练了一个非常好的分词器"。

### 单损失训练

Emu3 使用一个目标：跨文本 token、2D 图像 token 和 3D 视频 token 共享词表的下一 token 预测。训练时通过模态特定因子乘以权重来平衡贡献，但损失函数是相同的。

训练数据混合包含：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似。
- 纯文本：标准 NTP。

模型从数据分布中学习何时输出图像 token 何时输出文本 token。生成行为从模型在 `<image>` 标签后预测图像 token 中涌现。

### 无分类器引导和温度

自回归图像生成通过推理时的无分类器引导 (CFG) 获得显著提升。Emu3 使用了它：生成两次，一次用完整描述，一次用空描述，以引导权重（典型 3.0-7.0）混合 logits。这与扩散使用的 CFG 技巧相同，被借用到自回归设置中。

温度很重要：过高产生伪影，过低导致模式坍缩。Emu3 推荐的温度：感知任务 1.0，图像生成 0.8。

### 三种角色，一个模型

Emu3 以三个功能不同的 API 形式发布，但底层只有一套权重：

- Emu3-Gen。图像生成。输入文本，输出图像 token。
- Emu3-Chat。VQA 和描述。输入图像（token），输出文本。
- Emu3-Stage2。视频生成和视频 VQA。输入文本或视频，输出文本或视频。

无任务特定头。只是不同的提示模板。同一个检查点。

### 基准测试

来自 Emu3 论文（2024年9月）：

- 图像生成：在 MJHQ-30K FID 上超越 SDXL（5.4 vs 5.6），GenEval 总分（0.54 vs 0.55 —— 统计平手），Deep-Eval 综合分持平。
- 图像感知：在 VQAv2 上超越 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致持平。
- 视频生成：4 秒片段质量在 FVD 上与 Sora 时代公开基准测试的模型竞争。

数字并非总是赢——Emu3 在某处赢一分在另处丢一分——但"下一 token 预测就是你所需的一切"这一主张在跨模态上是站得住脚的。

### 计算成本

Emu3 在约 3000 亿多模态 token 上训练了一个 7B 参数的模型。GPU 小时大致相当于 Llama-2-7B 预训练（在 A100 级硅片上 2k-4k GPU 年）。Stable Diffusion 3 等扩散模型训练预算类似，但需要单独的文本编码器和更复杂的流水线。

在推理时，Emu3 每张图比 SDXL 慢：4096 个图像 token 以 30 tok/s 生成需要约 2 分钟生成一张 512×512 图像，而 SDXL 只需 2-5 秒。推测性解码 (speculative decoding) 和 KV cache 优化缩小了差距但未消除。自回归图像生成是计算密集型的；这是持续存在的权衡。

### 为什么重要

Emu3 的深层贡献是概念性的。如果下一 token 预测在图像生成上可以缩放到匹配扩散，那么统一模型路径（一个损失、一个骨干、任何模态）就是可行的。未来的模型不需要单独的文本编码器、单独的扩散调度器、单独的 VAE。一个 transformer，每种模态一个分词器，然后缩放。

Show-o、Janus-Pro 和 InternVL-U 都在这一论题的基础上构建或挑战它。中国实验室（BAAI、DeepSeek）在这个方向上比美国实验室发表得更积极（截至 2025 年）。

## 上手实践

`code/main.py` 构建两个简易组件：

- 2D vs 3D VQ 分词器 token 数量计算器：给定（分辨率、patch、片段长度、FPS），计算图像 vs 视频的 token 数。
- 带无分类器引导和温度的自回归图像 token 采样器。

CFG 实现匹配 Emu3 的方案——以引导权重混合条件和无条件 logits。

## 交付产出

本课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定生成产品规格（图像或视频、目标分辨率、质量层级、延迟预算），它计算 token 数、推理成本，并在 Emu3 家族和扩散之间做选择。

## 练习

1. Emu3 在 8×8 缩减下每张 512×512 图像产生 4096 个 token。计算 1024×1024 和 2048×2048 的等价值。推理延迟会怎样？

2. 阅读 Emu3 第 3.3 节关于视频分词器。描述 3D VQ patch 的形状以及为什么是 4×4×4 而非 8×8×1。

3. 无分类器引导权重 5.0 vs 3.0：视觉效果有何不同？追踪 `code/main.py` 中的数学过程。

4. 计算 Emu3-7B 在 300B token 上的训练 FLOPs 并与 Stable Diffusion 3 比较。哪个训练更贵？

5. Emu3 在 FID 上超越 SDXL 但在 VQAv2 上不敌专门的 VLM。解释为什么统一损失方法在不同基准测试上对比专家模型表现出不同的优势。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 下一 token 预测 (Next-token prediction) | "NTP" | 标准自回归损失：给定 token[0..i] 预测 token[i+1]；分词后适用于所有模态 |
| IBQ 分词器 (IBQ tokenizer) | "逆瓶颈量化器" | 一类具有更大码本（32768+）和比 Chameleon 更好重建质量的 VQ-VAE |
| 3D VQ | "时空量化器" | 以 (时间, 行, 列) 索引的码本；一个 token 覆盖一个 4×4×4 像素立方体 |
| 无分类器引导 (Classifier-free guidance) | "CFG" | 以权重 gamma 混合条件和无条件 logits；推理时提升图像质量 |
| 统一词表 (Unified vocabulary) | "共享 token" | 文本 + 图像 + 视频都从同一个整数空间取值；模型预测接下来的任何模态 |
| MJHQ-30K | "图像生成基准" | 含 30k 提示的 Midjourney 质量基准测试；Emu3 在此报告 FID |

## 延伸阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
