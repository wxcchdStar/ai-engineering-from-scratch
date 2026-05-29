# 视觉 Transformer（Vision Transformers, ViT）

> 将图像切成块（Patch），将每个块视为一个词，运行标准 Transformer。不要回头。

**类型：** 构建（Build）
**语言：** Python
**前置知识：** 第 7 阶段第 02 课（自注意力）、第 4 阶段第 04 课（图像分类）
**时间：** 约 45 分钟

## 学习目标

- 从零开始实现块嵌入（Patch Embedding）、可学习位置嵌入（Positional Embedding）、类别 token（Class Token）和 Transformer 编码器块，构建一个最小 ViT
- 解释为什么 ViT 被认为需要大规模预训练数据，直到 DeiT 和 MAE 证明并非如此
- 比较 ViT、Swin 和 ConvNeXt 的架构先验（无、局部窗口注意力、卷积骨干网络）
- 使用 `timm` 和标准的线性探测（Linear Probe）/ 微调方案在小型数据集上微调预训练 ViT

## 问题

十年来，卷积是计算机视觉的同义词。CNN 具有强大的归纳偏置（Inductive Bias）— 局部性（Locality）、平移等变性（Translation Equivariance）— 没有人认为你可以替换它们。然后 Dosovitskiy et al.（2020）展示了应用于展平图像块的纯 Transformer，完全没有卷积机制，可以在大规模上匹敌或击败最好的 CNN。

问题在于"在大规模上"。ViT 在 ImageNet-1k 上输给了 ResNet。在 ImageNet-21k 或 JFT-300M 上预训练然后在 ImageNet-1k 上微调的 ViT 击败了它。结论是 Transformer 缺乏有用的先验，但可以从足够的数据中学习它们。后续工作（DeiT、MAE、DINO）表明，通过正确的训练方案 — 强数据增强、自监督预训练、蒸馏 — ViT 在小数据上也能很好地训练。

到 2026 年，纯 CNN 在边缘设备上仍然有竞争力（ConvNeXt 是最强的），但 Transformer 主导了其他一切：分割（Mask2Former、SegFormer）、检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）、视频（VideoMAE、VJEPA）。ViT 块结构是需要了解的那个。

## 概念

### 流水线

```mermaid
flowchart LR
    IMG["图像<br/>（3, 224, 224）"] --> PATCH["块嵌入<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["展平为<br/>（196, 768）个 token"]
    FLAT --> CAT["前置<br/>[CLS] token"]
    CAT --> POS["添加可学习<br/>位置嵌入"]
    POS --> ENC["N 个 Transformer<br/>编码器块"]
    ENC --> CLS["取 [CLS]<br/>token 输出"]
    CLS --> HEAD["MLP 分类器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七个步骤。块 -> token -> 注意力 -> 分类器。每个变体（DeiT、Swin、ConvNeXt、MAE 预训练）改变其中一两个步骤，其余保持不变。

### 块嵌入

第一个卷积是秘密。卷积核大小 16，步长 16，因此 224x224 图像变成 14x14 的 16x16 块网格，每个块投影到 768 维嵌入。这个单一卷积同时完成分块和线性投影。

```
输入： (3, 224, 224)
Conv (3 -> 768, k=16, s=16, 无填充):
输出： (768, 14, 14)
展平空间维度： (196, 768)
```

196 个块 = 196 个 token。每个 token 的特征维度是 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### 类别 token

一个单一的可学习向量前置到序列中：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   形状 (197, 768)
```

经过 N 个 Transformer 块后，`[CLS]` 输出是全局图像表示。分类头只读取这一个向量。

### 位置嵌入

Transformer 没有内置的空间位置概念。为每个 token 添加一个可学习向量：

```
tokens = tokens + learned_pos_embedding   （形状也是 (197, 768)）
```

嵌入是模型的一个参数；基于梯度的训练使其适应 2D 图像结构。正弦 2D 替代方案存在但实践中很少使用。

### Transformer 编码器块

标准。多头自注意力（Multi-Head Self-Attention）、MLP、残差连接（Residual Connection）、前置 LayerNorm（Pre-LN）。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP 是两层带 GELU：Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 堆叠 12 个这样的块，每个块有 12 个注意力头，总计 86M 参数。

### 为什么是 Pre-LN

早期 Transformer 使用 Post-LN（`x = LN(x + sublayer(x))`），在没有预热的情况下难以训练超过 6-8 层。Pre-LN（`x = x + sublayer(LN(x))`）无需预热即可稳定训练更深的网络。每个 ViT 和每个现代 LLM 都使用 Pre-LN。

### 块大小权衡

- 16x16 块 -> 196 个 token，标准。
- 32x32 块 -> 49 个 token，更快但分辨率更低。
- 8x8 块 -> 784 个 token，更精细但 O(n^2) 注意力成本严重恶化。

更大的块 = 更少的 token = 更快但空间细节更少。SwinV2 在分层窗口中使用 4x4 块。

### DeiT 在 ImageNet-1k 上训练 ViT 的方案

原始 ViT 需要 JFT-300M 才能击败 CNN。DeiT（Touvron et al., 2020）仅用四个改变就在 ImageNet-1k 上将 ViT-B 训练到 81.8% top-1：

1. 强数据增强：RandAugment、Mixup、CutMix、Random Erasing。
2. 随机深度（Stochastic Depth）（训练时随机丢弃整个块）。
3. 重复增强（Repeated Augmentation）（同一图像每批次采样 3 次）。
4. 从 CNN 教师模型蒸馏（可选，进一步提升准确率）。

每个现代 ViT 训练方案都源自 DeiT。

### Swin vs ConvNeXt

- **Swin**（Liu et al., 2021）— 基于窗口的注意力。每个块在局部窗口内进行注意力计算；交替的块移动窗口以跨窗口混合信息。在保持注意力算子的同时带回类似 CNN 的局部性先验。
- **ConvNeXt**（Liu et al., 2022）— 重新设计的 CNN，匹配 Swin 的架构选择（深度可分离卷积、LayerNorm、GELU、倒置瓶颈）。表明差距不是"注意力 vs 卷积"而是"现代训练方案 + 架构"。

在 2026 年，ConvNeXt-V2 和 Swin-V2 都是生产级的；正确的选择取决于你的推理技术栈（ConvNeXt 对边缘设备编译更好）和预训练语料库。

### MAE 预训练

掩码自编码器（Masked Autoencoder, He et al., 2022）：随机掩码 75% 的块，训练编码器仅处理可见的 25%，训练一个小型解码器从编码器输出重建被掩码的块。预训练后，丢弃解码器并微调编码器。

MAE 使 ViT 可以仅在 ImageNet-1k 上训练，达到 SOTA，是当前默认的自监督方案。

## 构建它

### 步骤 1：块嵌入

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一个卷积，一次展平，一次转置。这就是整个图像到 token 的步骤。

### 步骤 2：Transformer 块

Pre-LN，多头自注意力，带 GELU 的 MLP，残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` 处理拆分为头、缩放点积和输出投影。`batch_first=True` 使形状为 `(N, seq, dim)`。

### 步骤 3：ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

约 2.8M 参数 — 一个可在 CPU 上处理的小型 ViT。真正的 ViT-B 是 86M；相同的类定义，使用 `dim=768, depth=12, num_heads=12`。

### 步骤 4：健全性检查 — 单图像推理

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

应该无错误运行。概率和为 1。

## 使用它

`timm` 提供每个 ViT 变体及其 ImageNet 预训练权重。一行代码：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm` 是 2026 年视觉 Transformer 的生产默认选择。在同一 API 下支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 等数十种模型。

对于多模态工作（图像 + 文本），`transformers` 提供 CLIP、SigLIP、BLIP-2、LLaVA。所有这些中的图像编码器都是 ViT 变体。

## 交付它

本课产出：

- `outputs/prompt-vit-vs-cnn-picker.md` — 一个提示词，根据数据集大小、计算资源和推理技术栈在 ViT、ConvNeXt 或 Swin 之间做出选择。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` — 一个技能，验证 ViT 的块嵌入和位置嵌入形状是否与模型预期的序列长度匹配，捕获最常见的移植错误。

## 练习

1. **（简单）** 打印上述小型 ViT 前向传播中每个中间张量的形状。确认：输入 `(N, 3, 64, 64)` -> 块 `(N, 16, 192)` -> 带 CLS `(N, 17, 192)` -> 分类器输入 `(N, 192)` -> 输出 `(N, num_classes)`。
2. **（中等）** 在第 4 课的合成 CIFAR 数据集上微调预训练的 `timm` ViT-S/16。与相同数据上的 ResNet-18 微调进行比较。报告训练时间和最终准确率。
3. **（困难）** 为小型 ViT 实现 MAE 预训练：掩码 75% 的块，训练编码器 + 小型解码器重建被掩码的块。评估预训练前后合成数据上的线性探测准确率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 块嵌入（Patch Embedding） | "第一个卷积" | 卷积核大小 = 步长 = 块大小的卷积；将图像变成 token 嵌入网格 |
| 类别 token（Class Token） | "[CLS]" | 前置到 token 序列的可学习向量；其最终输出是全局图像表示 |
| 位置嵌入（Positional Embedding） | "可学习位置" | 添加到每个 token 的可学习向量，使 Transformer 知道每个块来自哪里 |
| Pre-LN | "子层前的 LayerNorm" | 稳定的 Transformer 变体：`x + sublayer(LN(x))` 而非 `LN(x + sublayer(x))` |
| 多头注意力（Multi-Head Attention） | "并行注意力" | 标准 Transformer 注意力，拆分为 num_heads 个独立子空间，之后拼接 |
| ViT-B/16 | "Base, patch 16" | 规范尺寸：dim=768, depth=12, heads=12, patch_size=16, image=224；约 86M 参数 |
| DeiT | "数据高效 ViT" | 仅用 ImageNet-1k 配合强增强训练的 ViT；证明大规模预训练数据集并非严格必需 |
| MAE | "掩码自编码器" | 自监督预训练：掩码 75% 的块，重建；主流的 ViT 预训练方案 |

## 扩展阅读

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT 论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) — 如何仅在 ImageNet-1k 上训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) — MAE 预训练
- [timm 文档](https://huggingface.co/docs/timm) — 你在生产中使用的每个视觉 Transformer 的参考文档
