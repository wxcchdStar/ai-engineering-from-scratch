# 序列到序列模型（Sequence-to-Sequence Models）

> 两个 RNN 假装自己是翻译器。它们遇到的信息瓶颈正是注意力机制（Attention）存在的原因。

**类型（Type）：** 构建（Build）
**语言（Languages）：** Python
**前置课程（Prerequisites）：** 阶段 5 · 08（用于文本的 CNN 与 RNN），阶段 3 · 11（PyTorch 入门）
**时间（Time）：** 约 75 分钟

## 问题（The Problem）

分类任务将变长序列映射到单个标签。翻译任务则将变长序列映射到另一个变长序列。输入和输出使用不同的词汇表，可能属于不同的语言，且长度没有保证一致。

seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一个刻意简洁的方案解决了这个问题。两个 RNN。一个读取源语句并生成一个固定大小的上下文向量（Context Vector）。另一个读取该向量并逐词元（token）生成目标语句。代码与你在第 08 课中写的相同，只是拼接方式不同。

这值得学习，原因有二。第一，上下文向量瓶颈是 NLP 中最具教学价值的失败案例。它解释了注意力机制和 Transformer 擅长解决的一切问题的动机。第二，其训练方法（教师强制（Teacher Forcing）、计划采样（Scheduled Sampling）、推理时的束搜索（Beam Search））至今仍适用于包括大语言模型（LLM）在内的所有现代生成系统。

## 概念（The Concept）

**编码器（Encoder）。** 一个读取源语句的 RNN。其最终隐藏状态就是**上下文向量**——整个输入的固定大小摘要。理论上，除了源信息本身，什么都不丢失。

**解码器（Decoder）。** 另一个从上下文向量初始化的 RNN。每一步，它将上一个生成的词元作为输入，并生成目标词汇表上的概率分布。通过采样或 argmax 选择下一个词元。将其反馈回去。重复此过程，直到生成 `<EOS>` 词元或达到最大长度。

**训练：** 每个解码步骤的交叉熵损失（Cross-Entropy Loss），在整个序列上求和。通过标准的通过时间的反向传播（Backprop Through Time）在两个网络上进行。

**教师强制。** 在训练期间，解码器在第 `t` 步的输入是位置 `t-1` 处的*真实*词元，而不是解码器自己之前的预测。这能稳定训练；没有它，早期的错误会级联放大，模型永远学不会。在推理时，你必须使用模型自己的预测，因此始终存在训练/推理分布差距。这个差距被称为**曝光偏差（Exposure Bias）**。

**瓶颈。** 编码器学到的关于源信息的所有内容都必须被压缩到那一个上下文向量中。长句会丢失细节。罕见词会被模糊化。语序重排（如 "chat noir" 与 "black cat"）必须被记忆，而非被计算。

注意力机制（第 10 课）通过让解码器查看*每一个*编码器隐藏状态（而不仅仅是最后一个）来解决这个问题。这就是它的全部卖点。

## 构建（Build It）

### 步骤 1：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状为 `[batch, seq_len, hidden_dim]`——每个输入位置对应一个隐藏状态。`hidden` 的形状为 `[1, batch, hidden_dim]`——最后一步的状态。第 08 课说的是"对输出做池化以进行分类"。这里我们保留最后一个隐藏状态作为上下文向量，并忽略每个步骤的输出。

### 步骤 2：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器每次被调用一步。输入：一批单个词元和当前的隐藏状态。输出：下一个词元的词汇表 logits 和更新后的隐藏状态。

### 步骤 3：带教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

有两个值得命名的旋钮。`ignore_index=0` 跳过填充（padding）词元上的损失。`teacher_forcing_ratio` 是每一步使用真实词元而非模型预测的概率。从 1.0（完全教师强制）开始，在训练过程中逐步退火到约 0.5，以缩小曝光偏差差距。

### 步骤 4：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码（Greedy Decoding）在每一步选择概率最高的词元。它可能会走偏：一旦你选定了一个词元，就无法收回。**束搜索**保留前 `k` 个部分序列，并在最后选择得分最高的完整序列。束宽度 3-5 是标准配置。

### 步骤 5：瓶颈演示

在玩具复制任务上训练模型：源序列 `[a, b, c, d, e]`，目标序列 `[a, b, c, d, e]`。增加序列长度。观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损地记忆 40 个词元的输入。信息在每个编码器步骤都存在，但解码器只能看到最后一个状态。注意力机制直接解决了这个问题。

## 使用（Use It）

PyTorch 提供了 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供了完整的编码器-解码器模型（BART、T5、mBART、NLLB），这些模型在数十亿词元上训练而成。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器已经用 Transformer 取代了 RNN。高层结构（编码器、解码器、逐词元生成）与 2014 年的 seq2seq 论文完全相同。每个模块内部的机制不同。

### 何时仍然选择基于 RNN 的 seq2seq

对于新项目，几乎从不。特定的例外情况：

- 流式翻译，你需要以有界内存逐词元消费输入。
- 设备端文本生成，Transformer 的内存开销过高。
- 教学目的。理解编码器-解码器瓶颈是理解 Transformer 为何胜出的最快路径。

### 曝光偏差及其缓解方法

- **计划采样。** 在训练过程中逐步退火教师强制比率，使模型学会从自己的错误中恢复。
- **最小风险训练（Minimum Risk Training）。** 在句子级别的 BLEU 分数上训练，而不是在词元级别的交叉熵上训练。更接近你真正想要的目标。
- **强化学习微调。** 用一个指标奖励序列生成器。用于现代 LLM 的 RLHF 中。

这三种方法同样适用于基于 Transformer 的生成。

## 交付（Ship It）

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习（Exercises）

1. **简单。** 实现玩具复制任务。在输入-输出对（目标等于源）上训练一个 GRU seq2seq。测量长度 5、10、20 时的准确率。复现瓶颈现象。
2. **中等。** 添加束宽度为 3 的束搜索解码。在一个小型平行语料库上，对比贪心解码测量 BLEU 分数。记录束搜索在哪些地方胜出（通常是最后几个词元）以及在哪些地方没有区别。
3. **困难。** 在一个 10k 对的释义（paraphrase）数据集上微调 `facebook/bart-base`。将微调后模型的 beam-4 输出与基础模型在留出输入上的输出进行比较。报告 BLEU 分数并挑选 10 个定性示例。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 实际含义（What it actually means） |
|------|-----------------|-----------------------|
| 编码器（Encoder） | 输入 RNN | 读取源信息。生成每个步骤的隐藏状态和一个最终的上下文向量。 |
| 解码器（Decoder） | 输出 RNN | 从上下文向量初始化。逐词元生成目标词元。 |
| 上下文向量（Context vector） | 摘要 | 编码器的最终隐藏状态。固定大小。注意力机制解决的瓶颈所在。 |
| 教师强制（Teacher forcing） | 使用真实词元 | 在训练时将真实的前一个词元输入模型。稳定学习过程。 |
| 曝光偏差（Exposure bias） | 训练/测试差距 | 在真实词元上训练的模型从未练习过从自己的错误中恢复。 |
| 束搜索（Beam search） | 更好的解码方式 | 在每一步保留前 k 个部分序列，而不是贪心地确定选择。 |

## 延伸阅读（Further Reading）

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始的 seq2seq 论文。共四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入了 GRU 和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力机制论文。学完本课后立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可构建的 seq2seq + 注意力机制代码。
