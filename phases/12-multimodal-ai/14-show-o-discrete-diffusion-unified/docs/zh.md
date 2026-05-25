# Show-o 与离散扩散统一模型

> Transfusion 混合了连续和离散表示。Show-o（Xie et al., 2024年8月）走了另一条路：文本 token 使用因果下一 token 预测，图像 token 使用 MaskGIT 精神的掩码离散扩散 (masked discrete diffusion)。两者坐在同一个 transformer 中，共用混合注意力掩码。结果是在一个骨干、每种模态一个分词器、一种损失公式（将下一 token 预测扩展为掩码预测）上统一了 VQA、文生图、图像修补和混合模态生成。本课解读 Show-o 的设计——为什么掩码离散扩散是一种并行的少步图像生成器——并与 Transfusion 和 Emu3 做对比。

**类型：** 学习
**语言：** Python（标准库，掩码离散扩散采样器）
**前置课程：** Phase 12 · 13（Transfusion）
**时间：** ~120 分钟

## 学习目标

- 解释掩码离散扩散：均匀掩盖 token 然后让 transformer 恢复它们的调度。
- 比较并行图像解码（Show-o, MaskGIT）与自回归图像解码（Chameleon, Emu3）在速度和质量上的差异。
- 列出 Show-o 在一个检查点中处理的三种任务：文生图 (T2I)、VQA、图像修补 (inpainting)。
- 选择一种掩码调度（余弦、线性、截断）并推理其对样本质量的影响。

## 问题所在

Transfusion 的双损失训练可行，但动态更棘手——连续扩散损失与离散 NTP 损失在数值尺度上不同。平衡损失权重是一项超参数搜索。架构有效但复杂。

Show-o 的答案：保持两种模态都是离散的（像 Chameleon），但通过掩码离散扩散并行生成图像而非顺序生成。训练目标变为单一的掩码 token 预测，自然地泛化了下一 token 预测。

## 核心概念

### 掩码离散扩散（MaskGIT）

Chang et al.（2022）的原始 MaskGIT 技巧很优雅。从一张完全掩码的图像开始（每个 token 都是特殊的 `<MASK>` ID）。每一步并行预测所有被掩码的 token，然后保留置信度最高的 top-K 预测，重新掩盖其余的。经过约 8-16 次迭代，所有 token 被填充。每步揭示多少 token 的调度需要调优——余弦调度效果好。

训练很简单：从 [0, 1] 均匀采样一个掩码比例，应用到图像的 VQ token 上，训练 transformer 恢复被掩码的 token。完全就是 BERT 对文本做的事，缩放到图像生成。

### Show-o：一个 transformer，混合掩码

Show-o 将 MaskGIT 放入因果语言模型 transformer 中。注意力掩码是：

- 文本 token：因果（标准 LLM）。
- 图像 token：图像块内完全双向（这样被掩码的 token 在预测时能看到其他所有图像 token）。
- 文本到图像：文本注意到之前的图像，图像注意到之前的文本。

训练在以下之间交替：
1. 文本序列上的标准 NTP。
2. 文生图样本：文本 → 图像，图像 token 被掩码，掩码 token 预测损失。
3. VQA 样本：图像 → 文本，文本 token 被掩码（实际上就是 NTP）。

统一损失是 `<MASK>` token 上的交叉熵，它涵盖了文本 NTP（只有最后一个 token 被"掩码"）和图像掩码扩散（随机子集被掩码）。

### 并行采样

Show-o 用约 16 步生成一张图像，而不是约 1000 步（逐 token 自回归）或约 20 步（扩散）。每步并行预测所有被掩码的 token；提交置信度最高的 top-K；重复。

对比：
- Chameleon / Emu3（逐 token 自回归）：N_tokens 次前向传播，通常每张图 1024-4096 次。
- Transfusion（连续扩散）：约 20 步，每步一次完整 transformer 传播。
- Show-o（掩码离散扩散）：约 16 步，每步一次完整 transformer 传播。

Show-o 在相似规模模型上比 Chameleon 更快，步数与 Transfusion 大致匹配，但每步成本更低（离散词表 logits vs 连续 MSE 损失）。

### 一个检查点中的多任务

Show-o 在推理时通过提示格式选择支持四种任务：

- 文本生成：标准自回归文本输出。
- VQA：图像输入，文本输出。
- 文生图 (T2I)：文本输入，通过掩码离散扩散输出图像。
- 图像修补 (Inpainting)：图像中部分 token 被掩码，填充补全。

修补能力从掩码预测训练中免费获得。掩盖 VQ token 网格的一个区域，输入其余部分加文本提示，预测被掩码的 token。

### 掩码调度

每步揭示多少 token 的调度决定了质量。Show-o 推荐余弦调度：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

第 0 步，所有 token 被掩码（比例 1.0）。第 T 步，无掩码。余弦将质量集中在中间比例范围，此时预测最具信息量。线性调度也可以但收敛更快到平台期。

### Show-o2

Show-o2（2025 年后续，arXiv 2506.15564）缩放了 Show-o：更大的 LLM 基座，更好的分词器，改进的掩码调度。相同的架构模式。

### Show-o 的位置

在 2026 年的分类体系中：

- 离散 token + NTP：Chameleon, Emu3。简单但推理慢。
- 离散 token + 掩码扩散：Show-o, MaskGIT, LlamaGen, Muse。并行采样，仍受限于分词器的有损性。
- 连续 + 扩散：Transfusion, MMDiT, DiT。最高质量，训练更复杂。
- 连续 + VLM 中的流匹配：JanusFlow, InternVL-U。最新的。

按任务选择：当你需要在一个开源模型中以合理速度实现 T2I + 修补 + VQA 时选 Show-o；当质量至上且你能承受双损失管道时选 Transfusion。

## 上手实践

`code/main.py` 模拟 Show-o 采样：

- 一个 16 个 VQ token 的简易网格。
- 一个模拟"transformer"，根据提示和当前未掩码的 token 预测 logits。
- 8 步余弦调度的并行掩码采样。
- 打印中间状态（掩码模式演变）和最终 token。

运行它，观察掩码逐步消融。

## 交付产出

本课产出 `outputs/skill-unified-gen-model-picker.md`。给定需要同时具备理解（VQA、描述）和生成（T2I、修补）且有开源权重约束的产品，在 Show-o 家族、Transfusion/MMDiT 家族和 Emu3/Chameleon 家族之间做选择，给出具体的权衡。

## 练习

1. 掩码离散扩散采样约 16 步。为什么不是 1 步？如果在第 0 步就揭示所有 token 会怎样？

2. 修补在掩码扩散中是免费的。提出一个产品使用场景（真实或假设的），其中 Show-o 的修补优于专家模型。

3. 余弦调度 vs 线性调度：追踪 T=8 时每步的未掩码 token 数。哪个更均衡？

4. 一张 512×512 的 Show-o 图像是 1024 个 token。在词表 K=16384 时，模型输出 1024 × log2(16384) = 14,336 bit（约 1.75 KiB）的数据。Stable Diffusion 输出 512×512×24 bit = 6,291,456 bit（约 768 KiB）的原始像素。压缩比是多少，换来了什么质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的类条件自回归图像模型与 Show-o 的掩码方法有何不同？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 掩码离散扩散 (Masked discrete diffusion) | "MaskGIT 风格" | 训练预测被掩码的 token；推理时迭代地揭示置信度最高的预测 |
| 余弦调度 (Cosine schedule) | "揭示调度" | 推理步数上掩码比例的衰减；将置信度增长集中在中间范围 |
| 并行解码 (Parallel decoding) | "所有 token 同时" | 每步在一次前向传播中预测所有被掩码 token 的完整序列，然后提交 top-K |
| 混合注意力 (Hybrid attention) | "因果 + 双向" | 对文本 token 因果、对图像块内双向的掩码 |
| 图像修补 (Inpainting) | "填充生成" | 以部分 token 被掩码的图像为条件，预测缺失部分；从训练目标中免费获得 |
| 提交率 (Commitment rate) | "每步 top-K" | 每次迭代中声明"完成"的 token 数量；控制推理速度 vs 质量的权衡 |

## 延伸阅读

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
