# 注意力机制（Attention Mechanism）—— 突破性进展

> 解码器不再眯着眼看一个压缩后的摘要，而是开始审视整个源序列。此后的一切，都是注意力机制加上工程实践。

**类型：** 构建
**语言：** Python
**前置课程：** 第 5 阶段 · 09（序列到序列模型）
**预计时间：** 约 45 分钟

## 问题所在

第 09 课以一个可量化的失败收尾。一个在玩具复制任务上训练的 GRU 编码器-解码器（Encoder-Decoder），在长度为 5 时准确率达到 89%，但在长度为 80 时几乎跌至随机猜测水平。原因在于结构，而非训练错误：编码器提取的每一点信息都必须塞进一个固定大小的隐藏状态（Hidden State）中，而解码器永远看不到其他任何东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行代码的修复方案。与其只给解码器最终的编码器状态，不如保留每一个编码器状态。在每个解码步骤中，计算编码器状态的加权平均（Weighted Average），其中权重表示"解码器此刻需要关注编码器位置 `i` 的程度"。这个加权平均就是上下文向量（Context Vector），它在每个解码步骤中都会变化。

这就是全部的核心思想。Transformer 对其进行了扩展。自注意力（Self-Attention）将其应用于单个序列。多头注意力（Multi-Head Attention）将其并行运行。但 2014 年的版本已经打破了瓶颈，一旦你掌握了它，转向 Transformer 就只是工程问题，而非概念问题。

## 核心概念

![Bahdanau 注意力：解码器查询所有编码器状态](../assets/attention.svg)

在每个解码步骤 `t`：

1. 使用上一个解码器隐藏状态 `s_{t-1}` 作为**查询（Query）**。
2. 将其与每个编码器隐藏状态 `h_1, ..., h_T` 进行评分。每个编码器位置得到一个标量。
3. 对评分进行 Softmax 归一化，得到注意力权重（Attention Weights）`α_{t,1}, ..., α_{t,T}`，其和为 1。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`。即编码器状态的加权平均。
5. 解码器接收 `c_t` 加上前一个输出词元（Token），生成下一个词元。

加权平均是关键所在。当解码器需要将 "Je" 翻译为 "I" 时，它会将 "Je" 对应的编码器状态权重调高，其他调低。当需要翻译 "not" 时，它会将 "pas" 的权重调高。上下文向量在每个步骤中都会重新调整。

## 张量形状（Shape）—— 最容易出错的地方

这是每个注意力机制实现第一次都会出错的地方。请仔细阅读。

| 对象 | 形状 | 说明 |
|-------|-------|-------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果是 BiLSTM，则 `d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力评分 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 做 Softmax 之后 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态形状相同 |

**Bahdanau（加性，Additive）评分。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 的形状为 `(d_s,)`，`h_i` 的形状为 `(d_h,)`。
- `W_a` 的形状为 `(d_attn, d_s)`。`U_a` 的形状为 `(d_attn, d_h)`。
- 它们在 tanh 内部的和的形状为 `(d_attn,)`。
- `v_α` 的形状为 `(d_attn,)`。与 `v_α` 的内积将其压缩为一个标量。**这就是 `v_α` 的作用。** 它不是什么魔法，而是一个投影（Projection），将注意力维度的向量转化为一个标量评分。

**Luong（乘性，Multiplicative）评分。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。这是一个硬性约束。如果你的编码器是双向的，请跳过此变体。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` 的形状为 `(d_s, d_h)`。消除了维度必须相等的约束。
- `concat`：本质上就是 Bahdanau 的形式。由于前两种更高效，很少使用。

**一个值得指出的 Bahdanau / Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（生成当前词*之前*的解码器状态）。Luong 使用 `s_t`（生成当前词*之后*的状态）。混淆二者会产生微妙的错误梯度，极难调试。选定一篇论文，严格遵循其约定。

## 动手构建

### 步骤 1：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上表检查你的张量形状。`encoder_states` 的形状为 `(T_enc, d_h)`。`projected_enc` 的形状为 `(T_enc, d_attn)`。`projected_dec` 的形状为 `(d_attn,)`，并通过广播（Broadcast）相加。`combined` 的形状为 `(T_enc, d_attn)`。`scores` 的形状为 `(T_enc,)`。`weights` 的形状为 `(T_enc,)`。`context` 的形状为 `(d_h,)`。搞定。

### 步骤 2：Luong dot 和 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

各三行代码。这就是 Luong 论文能迅速普及的原因。在大多数任务上达到相同的准确率，代码却少得多。

### 步骤 3：一个数值计算示例

给定三个编码器状态（大致对应 "cat"、"sat"、"mat"）和一个与第一个状态最对齐的解码器状态，注意力分布会集中在位置 0。如果解码器状态偏移到与最后一个状态对齐，注意力就会转移到位置 2。上下文向量随之变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后将解码器状态移近第三个编码器状态，观察权重如何转移。就是这样。注意力就是显式的对齐（Alignment）。

### 步骤 4：为什么这是通向 Transformer 的桥梁

将上面的语言翻译成 Q/K/V：

- **查询（Query）** = 解码器状态 `s_{t-1}`
- **键（Key）** = 编码器状态（我们用来评分的对象）
- **值（Value）** = 编码器状态（我们加权求和的对象）

在经典注意力中，键和值是同一个东西。自注意力（Self-Attention）将它们分开：你可以用不同的可学习投影来查询一个序列自身，分别得到 K 和 V。多头注意力（Multi-Head Attention）用不同的可学习投影并行运行注意力。Transformer 将整个阶段堆叠多次，并去掉了 RNN。

数学原理是一样的。张量形状是一样的。从 Bahdanau 注意力到缩放点积注意力（Scaled Dot-Product Attention）的教学跳跃，主要只是符号上的变化。

## 实际使用

PyTorch 和 TensorFlow 直接内置了注意力机制。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 Transformer 注意力层。查询批次有 5 个位置，键/值批次有 10 个位置，每个 128 维，8 个头。`output` 是新的上下文增强后的查询。`weights` 是 5x10 的对齐矩阵，你可以将其可视化。

### 经典注意力仍然重要的场景

- 教学。单头、单层、基于 RNN 的版本让每个概念都清晰可见。
- 设备端序列任务，Transformer 放不下的场景。
- 任何 2014-2017 年的论文。不了解 Bahdanau 的约定，你会读错它们。
- 机器翻译（MT）中的细粒度对齐分析。即使在 Transformer 模型上，原始注意力权重也是一种可解释性工具，而读懂它们需要知道它们是什么。

### "注意力权重即解释"的陷阱

注意力权重看起来是可解释的。它们是跨位置求和为 1 的权重；你可以把它们画出来；高权重意味着"模型关注了这里"。审稿人喜欢它们。

但它们并不像看起来那样可解释。Jain 和 Wallace（2019）证明，对于某些任务，注意力分布可以被置换并替换为任意替代分布，而不会改变模型预测。在没有消融实验（Ablation）或反事实检验（Counterfactual Check）的情况下，永远不要将注意力权重作为推理证据来报告。

## 交付成果

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **简单。** 实现 `softmax` 掩码（Masking），使编码器中的填充（Padding）词元获得零注意力权重。在一个包含变长序列的批次上进行测试。
2. **中等。** 为 Luong `general` 形式添加多头注意力。将 `d_h` 分成 `n_heads` 组，每组独立运行注意力，然后拼接（Concatenate）。验证单头情况与你之前的实现一致。
3. **困难。** 在第 09 课的玩具复制任务上，训练一个带 Bahdanau 注意力的 GRU 编码器-解码器。绘制准确率随序列长度变化的曲线。与无注意力的基线进行对比。你应该会看到随着长度增加，差距逐渐拉大，这证实了注意力机制确实打破了瓶颈。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 注意力（Attention） | 关注某些东西 | 值序列的加权平均，权重由查询-键相似度计算得出。 |
| 查询、键、值（Query, Key, Value） | QKV | 三个投影：Q 负责提问，K 是匹配目标，V 是返回内容。 |
| 加性注意力（Additive Attention） | Bahdanau | 前馈评分：`v^T tanh(W q + U k)`。 |
| 乘性注意力（Multiplicative Attention） | Luong dot / general | 评分为 `q^T k` 或 `q^T W k`。更高效，在大多数任务上准确率相同。 |
| 对齐矩阵（Alignment Matrix） | 那张漂亮的图 | 注意力权重构成的 `(T_dec, T_enc)` 网格。阅读它可以了解模型关注了什么。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原始论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种评分变体及其对比。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 关于可解释性的警示。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 带 PyTorch 的可运行教程。
