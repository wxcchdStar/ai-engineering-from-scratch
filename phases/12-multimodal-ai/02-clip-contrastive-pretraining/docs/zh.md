# CLIP 与对比式视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足以支撑未来五年的简单想法：仅使用嘈杂的网络图文对和对比损失，将图像编码器与文本编码器对齐到同一向量空间中。零监督标签。4 亿对数据。由此产生的嵌入空间能够进行零样本分类、图文检索，并作为视觉塔接入每一个 2026 年的 VLM。SigLIP 2（2025）用 sigmoid 替代了 softmax，以更低成本超越了 CLIP 的规模。本课将从 InfoNCE 到 sigmoid 成对损失的数学推导出发，并用标准库 Python 实现训练步骤。

**类型：** 构建
**语言：** Python（标准库，InfoNCE + sigmoid 损失实现）
**前置知识：** 第 12 阶段 · 01（ViT patches），第 7 阶段（Transformers）
**时间：** 约 180 分钟

## 学习目标

- 从互信息推导 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释为什么 sigmoid 成对损失（SigLIP）能够扩展到 batch 32768+，而无需 softmax 所需的 all-gather 开销。
- 通过构造文本模板（`a photo of a {class}`）并取余弦相似度的 argmax，运行零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练提供的四个控制杠杆：batch size、温度、prompt 模板、数据质量。

## 问题

CLIP 之前的视觉是监督式的。收集标注数据集（ImageNet：120 万张图片，1000 个类别），训练一个 CNN，交付使用。标签昂贵，标签偏向于标注者能达成共识的内容，且标签不经微调就无法迁移到新任务。

互联网上的图文对拥有超过十亿个松散的免费标注。一张金毛犬的图片配上 alt 文本 "my dog Max in the park" 就携带着监督信号——文本描述了图像。问题是：你能把它变成有用的训练吗？

CLIP 的回答：把图文对当作匹配任务。给定一个 batch 包含 N 张图片和 N 条描述，学习将每张图片与其自己的描述匹配，对抗 N-1 个干扰项。监督信号是"这两者属于一起；其他 N-1 对不是"。没有类别标签。没有人工标注。只有一个对比损失。

由此产生的嵌入空间能做的不止 CLIP 训练时的目标。ImageNet 零样本之所以有效，是因为 "a photo of a cat" 嵌入到了那些从未被显式标注为"猫"的猫图片附近。这就是催生了每一个 2026 年 VLM 的赌注。

## 核心概念

### 双塔编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图片输出一个 D 维向量。
- 文本编码器 `g`：小型 transformer，每条描述输出一个 D 维向量。

两个塔都将其输出归一化为单位长度。由于两者都是单位范数，相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`。

对于包含 N 个（图片，描述）对的 batch，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是可学习的温度（CLIP 初始化为 0.07；在 log 空间中学习）。

### InfoNCE 损失

CLIP 使用对行和列的对称交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图片的正样本是其自己的描述
loss_t2i = CE(S^T, labels=identity)   # 每条描述的正样本是其自己的图片
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 强制每张图片与其描述比 batch 中的其他任何描述更匹配。"负样本"是 batch 中的所有其他项。更大的 batch = 更多负样本 = 更强的信号。CLIP 以 batch 32k 训练；规模很重要。

### 温度

`tau` 控制 softmax 的锐度。低 tau → 尖锐分布，难负样本挖掘效果。高 tau → 柔和，所有样本都有贡献。CLIP 学习 log(1/tau)，加以裁剪以防止崩塌。SigLIP 2 固定初始 tau 并改用可学习的偏置。

### 为什么 sigmoid 扩展性更好（SigLIP）

Softmax 需要整个相似度矩阵同步。在分布式训练中，你必须将每个嵌入 all-gather 到每个副本，然后做 softmax。这在通信上对 world size 是二次的。

SigLIP 用逐元素的 sigmoid 替代 softmax：对于每对 `(i, j)`，损失是一个二分类问题——"这是匹配对吗？"正类标签在对角线上，其他都是负类。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 如果 `i == j`，否则为 0。每对损失是独立的。不需要 all-gather。每个 GPU 计算其本地块并求和。SigLIP 2 可以廉价地扩展到 batch 32k-512k，而 CLIP 则需要成比例的更多通信。

### 零样本分类

给定 N 个类别名称，为每个类别构造一个文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图片。取余弦相似度的 argmax = 预测类别。无需在目标类别上训练。

Prompt 模板很重要。CLIP 原始论文对每个类别使用 80 个模板（普通的、艺术的、照片、绘画等）并平均嵌入。ImageNet 上提升 3 个百分点。现代用法通常选择一两个模板。

### 线性探针与微调

零样本是基线。线性探针（在冻结的 CLIP 特征上为你的目标类别训练一个线性层）在域内任务上优于零样本。全量微调在域内优于线性探针，但可能损害零样本迁移能力。三种方案，三种权衡。

### SigLIP 2：NaFlex 与稠密特征

SigLIP 2（2025）新增：
- NaFlex（原生弹性分辨率）：单一模型处理可变宽高比和分辨率。
- 更好的稠密特征用于分割和深度估计，目标是作为 VLM 中的冻结骨干网络使用。
- 多语言：在 100+ 种语言上训练，而 CLIP 仅支持英语。
- 1B 参数规模，而 CLIP 最高仅 400M。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认的视觉塔。CLIP 仍然是纯图文检索的默认选择，当特定的 LAION-2B 训练分布与你的查询模式匹配时。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的思路，18 亿对规模，90% 噪声。证明了噪声数据可以规模化。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开源复现，多尺度，是首选的开源检查点。EVA-CLIP：从掩码图像建模初始化；是 VLM 的强骨干。BASIC：Google 的 CLIP+ALIGN 混合体。所有这些都属于同一家族，只是数据和调参不同。

### 零样本天花板

CLIP 类模型在 ImageNet 零样本上的上限约为 76%（CLIP-G、OpenCLIP-G）。要超越这一水平，要么需要更大的数据（SigLIP 2 达到 80%+），要么需要架构变更（监督头、更多参数）。基准测试正在趋于饱和；真正的价值在于下游 VLM 所使用的嵌入空间。

## 使用方法

`code/main.py` 实现了：

1. 一个玩具级双编码器（基于哈希的图像特征，文本字符特征），让你无需 numpy 也能直观理解 InfoNCE 的结构。
2. 纯 Python 实现的 InfoNCE 损失函数（通过 log-sum-exp 保证数值稳定性）。
3. Sigmoid 成对损失用于对比。
4. 零样本分类例程：计算与一组文本提示的余弦相似度，取 argmax 作为预测结果。

运行它并观察损失曲线。绝对值是玩具级别的，但曲线形态与真实 CLIP 训练器输出的一致。

## 交付物

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和一个目标类别列表，它使用 CLIP 模板构建文本提示，用指定的检查点（如 `openai/clip-vit-large-patch14`）对两侧进行嵌入，并返回 top-1 / top-5 预测结果及相似度分数。该 skill 不会对未出现在提示列表中的类别做出任何断言。

## 练习

1. 手工为 4 对样本实现 InfoNCE。构建 4×4 的相似度矩阵，执行 softmax，取出对角线，计算交叉熵。用你的 Python 实现验证手工计算结果。

2. SigLIP 除了温度参数外，还使用了一个偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当批次中存在严重的类别不平衡（每行负样本远多于正样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 构建一个猫 vs 狗的零样本分类器。尝试两种提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上衡量准确率。模板集成是否优于单个模板？

4. 计算 softmax InfoNCE 与 sigmoid 成对损失在 512 GPU、批次大小 32k 时的通信开销。哪个的复杂度是 O(N)，哪个是 O(N²)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 缩放定律论文（arXiv:2212.07143，Cherti 等）。从图表中复现他们关于数据缩放的结论：在模型大小固定的情况下，ImageNet 零样本准确率与训练数据量之间的对数线性关系是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| InfoNCE | "对比损失" | 在批次的相似度矩阵上做交叉熵；每个样本的正例是其配对项，负例是其他所有项 |
| Sigmoid loss | "SigLIP 损失" | 逐对的二元交叉熵；无需 softmax，无需 all-gather，在分布式训练中扩展成本低 |
| Temperature | "tau" | 在 softmax/sigmoid 之前缩放 logits 的标量；控制分布的锐度 |
| Zero-shot | "无需微调的分类" | 用文本提示构建类别嵌入，通过余弦相似度分类；无需在目标类别上训练 |
| Prompt template | "a photo of a ..." | 围绕类别名称的文本框架；可影响零样本准确率 1-5 个百分点 |
| Dual encoder | "双塔" | 一个图像编码器 + 一个文本编码器，输出在共享的 D 维空间中 |
| Hard negative | "难区分的干扰项" | 与正例足够相似的负例，模型需要下功夫才能将它们分开 |
| Linear probe | "冻结 + 一层" | 仅在冻结的特征之上训练一个线性分类器；用于衡量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的能力，可以以任意宽高比和分辨率输入图像，无需调整大小 |
| Temperature scaling | "对数参数化的 tau" | CLIP 对 `log(1/tau)` 进行参数化以改善梯度行为；裁剪以防止 tau 塌缩到接近零 |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 利用噪声网络数据扩展规模。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 缩放定律。
