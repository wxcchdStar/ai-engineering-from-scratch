# 用于文本的 CNN 与 RNN

> 卷积（Convolution）学习 n-gram。循环（Recurrence）负责记忆。两者都已被注意力机制（Attention）取代，但在受限硬件上两者仍然重要。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 3 · 11（PyTorch 入门），阶段 5 · 03（词嵌入（Word Embeddings）），阶段 4 · 02（从零实现卷积（Convolutions from Scratch））
**时间：** 约 75 分钟

## 问题所在

TF-IDF 和 Word2Vec 产生的是忽略词序的平坦向量。基于它们构建的分类器无法区分 `dog bites man` 和 `man bites dog`。词序有时承载着关键信号。

在 Transformer 出现之前，有两类架构填补了这一空白。

**用于文本的卷积网络（TextCNN）。** 在词嵌入序列上应用一维卷积（1D Convolution）。宽度为 3 的滤波器（Filter）是一个可学习的三元组检测器：它跨越三个词并输出一个分数。堆叠不同宽度（2、3、4、5）的滤波器来检测多尺度模式。通过最大池化（Max-Pooling）得到固定大小的表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 逐个处理词元（Token），维护一个将信息向前传递的隐藏状态（Hidden State）。顺序执行、携带记忆、支持可变输入长度。从 2014 年到 2017 年主导了序列建模（Sequence Modeling），然后注意力机制出现了。

本课将构建这两种架构，然后指出促使注意力机制诞生的失败之处。

## 核心概念

**TextCNN**（Kim, 2014）。词元被嵌入。一个宽度为 `k` 的一维卷积在连续的 `k`-gram 嵌入上滑动滤波器，生成特征图（Feature Map）。对该特征图进行全局最大池化（Global Max-Pooling），选出最强的激活值。将来自多个滤波器宽度的最大池化输出拼接起来，送入分类器头（Classifier Head）。

为什么有效。一个滤波器就是一个可学习的 n-gram。最大池化是位置无关的，因此 "not good" 无论出现在评论的开头还是中间，都会触发相同的特征。三种滤波器宽度各配 100 个滤波器，就能得到 300 个学习到的 n-gram 检测器。训练是并行的，没有顺序依赖。

**RNN。** 在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b` 在时间上共享。时间 `T` 处的隐藏状态是整个前缀的摘要。对于分类任务，对 `h_1 ... h_T` 进行池化（最大值、均值或最后一个）。

普通 RNN 存在梯度消失（Vanishing Gradient）问题。**LSTM** 添加了门控（Gate），决定遗忘什么、存储什么、输出什么，从而在长序列中稳定梯度。**GRU** 将 LSTM 简化为两个门，参数更少但性能相近。

**双向 RNN（Bidirectional RNN）** 同时运行一个前向 RNN 和一个后向 RNN，将隐藏状态拼接起来。每个词元的表示都能看到左右两侧的上下文。对于标注任务至关重要。

## 动手构建

### 步骤 1：在 PyTorch 中实现 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道（Channel）。池化后的输出大小是固定的，与输入长度无关。

### 步骤 2：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列做最大池化，而不是取最后一个状态。对于分类任务，最大池化通常优于取最后一个隐藏状态，因为长序列末尾的信息往往会主导最后一个状态。

### 步骤 3：梯度消失演示（直观理解）

没有门控的普通 RNN 无法学习长距离依赖（Long-Range Dependency）。考虑一个玩具任务：预测词元 `A` 是否在序列中的任何位置出现过。如果 `A` 在位置 1，而序列长度为 100 个词元，那么来自损失的梯度必须通过 99 次循环权重（Recurrent Weight）的乘法反向传播。如果权重小于 1，梯度会消失；如果大于 1，梯度会爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 权重为 0.9，经过 100 步：
#   0.9 ^ 100 ≈ 2.7e-5
# 从第 100 步到第 1 步的梯度实际上为零。
```

LSTM 通过**细胞状态（Cell State）**来解决这个问题，细胞状态以仅包含加法交互的方式贯穿网络（遗忘门（Forget Gate）以乘法方式对其进行缩放，但梯度仍然沿着这条"高速公路"流动）。GRU 用更少的参数做了类似的事情。两者都能让你在 100 步以上的序列中稳定训练。

### 步骤 4：为什么这仍然不够

即使有了 LSTM，仍然存在三个问题。

1. **顺序瓶颈（Sequential Bottleneck）。** 在长度为 1000 的序列上训练 RNN 需要 1000 次串行的前向/反向步骤，无法在时间维度上并行化。
2. **编码器-解码器（Encoder-Decoder）架构中的固定大小上下文向量。** 解码器（Decoder）只能看到编码器（Encoder）的最终隐藏状态，整个输入被压缩到其中。长输入会丢失细节。第 09 课将直接讨论这一点。
3. **远距离依赖的准确率天花板。** LSTM 优于普通 RNN，但仍然难以在 200 步以上的距离上传播特定信息。

注意力机制解决了所有三个问题。Transformer 完全抛弃了循环。第 10 课是转折点。

## 实际使用

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 是生产就绪的。训练代码是标准的。

Hugging Face 提供了预训练嵌入（Pretrained Embeddings），你可以将其作为输入层插入：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

适用场景检查清单。

- **边缘/端侧推理（Edge / On-Device Inference）。** 使用 GloVe 嵌入的 TextCNN 比 Transformer 小 10 到 100 倍。如果你的部署目标是手机，这就是你的技术栈。
- **流式/在线分类（Streaming / Online Classification）。** RNN 一次处理一个词元；Transformer 需要完整序列。对于实时传入的文本，LSTM 仍然胜出。
- **用于基线的微型模型。** 在新任务上快速迭代。在 CPU 上用 5 分钟训练一个 TextCNN。
- **数据有限的序列标注（Sequence Labeling）。** BiLSTM-CRF（第 06 课）对于 1k-10k 标注句子仍然是生产级的命名实体识别（NER）架构。

其他所有场景都用 Transformer。

## 交付物

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **简单。** 在一个 3 分类的玩具数据集上训练一个 TextCNN（数据由你自己构造）。验证滤波器宽度 (2, 3, 4) 在平均 F1 分数上优于单一宽度 (3)。
2. **中等。** 为 LSTM 分类器实现最大池化、均值池化和最后状态池化。在一个小数据集上进行比较，记录哪种池化方式胜出并推测原因。
3. **困难。** 构建一个 BiLSTM-CRF 命名实体识别标注器（结合第 06 课和本课内容）。在 CoNLL-2003 上训练。与第 06 课的纯 CRF 基线以及 BERT 微调进行比较。报告训练时间、内存和 F1 分数。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| TextCNN | 用于文本的 CNN | 在词嵌入上堆叠一维卷积并进行全局最大池化。Kim (2014)。 |
| RNN | 循环网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 带门控的 RNN | 添加输入门（Input Gate）/ 遗忘门（Forget Gate）/ 输出门（Output Gate）+ 细胞状态。在长序列中稳定训练。 |
| GRU | 更简单的 LSTM | 两个门而非三个。准确率相近，参数更少。 |
| 双向（Bidirectional） | 两个方向 | 前向 + 后向 RNN 拼接。每个词元都能看到其上下文的左右两侧。 |
| 梯度消失（Vanishing Gradient） | 训练信号消失 | 普通 RNN 中反复乘以小于 1 的权重，使早期步骤的梯度实际上为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文。八页。可读性强。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文。出乎意料地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 对所有人都变得易懂的图解。
