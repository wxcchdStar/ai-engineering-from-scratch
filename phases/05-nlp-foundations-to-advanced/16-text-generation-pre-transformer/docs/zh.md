# 文本生成在 Transformer 之前 — N-gram 语言模型（N-gram Language Models）

> 如果一个词令人意外，说明模型不好。困惑度（Perplexity）将意外量化为数字。平滑（Smoothing）使其保持有限。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 01（文本处理），第 2 阶段 · 14（朴素贝叶斯）
**时间：** 约 45 分钟

## 问题

在 transformer 架构（transformers）、循环神经网络（RNNs）、词嵌入（Word Embeddings）出现之前，语言模型（Language Model）通过统计一个词在前面 `n-1` 个词之后出现的次数来预测下一个词。统计 "the cat" → "sat" 出现 47 次，"the cat" → "jumped" 出现 12 次，"the cat" → "refrigerator" 出现 0 次。归一化后得到一个概率分布（Probability Distribution）。

这就是 n-gram 语言模型。从 1980 年到 2015 年，它驱动了每一个语音识别器（Speech Recognizer）、每一个拼写检查器（Spell Checker）以及每一个基于短语的机器翻译（Phrase-Based Machine Translation）系统。当你需要在设备端进行低成本的语言建模时，它仍然在运行。

有趣的问题在于如何处理未见过的 n-gram。基于原始计数的模型会为任何未见过的内容分配零概率，这是灾难性的，因为句子很长，而且几乎每个长句子都至少包含一个未见过的序列。五十年的平滑研究解决了这个问题。Kneser-Ney 平滑（Kneser-Ney Smoothing）就是其成果，而现代深度学习继承了它的经验传统。

## 概念

![N-gram 模型：计数、平滑、生成](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（三元语法 trigram 通常取 3，四元语法 4-gram 取 4）。通过计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何在训练中未见过的 n-gram 都会得到零概率。2007 年一项针对 Brown 语料库（Brown Corpus）的研究发现，即使是 4-gram 模型，在留出数据中也有 30% 的 4-gram 在训练中从未出现。如果不进行平滑处理，你无法在任何真实文本上进行评估。

**平滑方法，按复杂程度递增排列：**

1. **拉普拉斯平滑/加一平滑（Laplace/Add-One）。** 每个计数加 1。简单，但对稀有事件效果很差。
2. **Good-Turing 平滑。** 基于频率的频率，将概率质量从高频事件重新分配给未见事件。
3. **插值（Interpolation）。** 用可调权重组合 n-gram、(n-1)-gram 等估计值。
4. **回退（Backoff）。** 如果 n-gram 计数为零，回退到 (n-1)-gram。Katz 回退（Katz Backoff）对此进行了规范化。
5. **绝对折扣（Absolute Discounting）。** 从所有计数中减去一个固定折扣 `D`，重新分配给未见事件。
6. **Kneser-Ney 平滑。** 绝对折扣加上对低阶模型的巧妙选择：使用*延续概率（Continuation Probability）*（一个词出现在多少种上下文中）而不是原始频率。

Kneser-Ney 的洞察非常深刻。"San Francisco" 是一个常见的 bigram（二元语法）。unigram（一元语法）"Francisco" 大多出现在 "San" 之后。朴素的绝对折扣会给 "Francisco" 很高的 unigram 概率（因为计数很高）。Kneser-Ney 注意到 "Francisco" 只出现在一种上下文中，因此相应地降低了它的延续概率。结果：以 "Francisco" 结尾的新颖 bigram 会得到适当的低概率。

**评估：困惑度（Perplexity）。** 在留出测试集上，每个词的平均负对数似然的指数。越低越好。困惑度为 100 意味着模型就像在 100 个词中均匀选择一样困惑。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 构建它

### 步骤 1：trigram 计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是一个分词后的句子列表。输出是 n-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界标记。

### 步骤 2：拉普拉斯平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

每个计数加 1。实现了平滑，但将过多的概率质量分配给了未见事件，同时也损害了已知稀有事件。

### 步骤 3：Kneser-Ney（bigram，插值版本）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个关键部分。`continuation_prob` 捕获了"这个词出现在多少种不同的上下文中？"（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放出的概率质量，用于加权回退项。最终概率是折扣后的主项加上加权的延续项。

### 步骤 4：使用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。每次使用不同的种子（seed）会得到不同的输出。要获得类似束搜索（Beam Search）的输出，可以在每一步选择 argmax（贪心/Greedy），并添加一个小的随机性调节参数（温度/Temperature）。

### 步骤 5：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于 Brown 语料库，一个调优良好的 4-gram KN 模型的困惑度约为 140。而 transformer 语言模型在相同测试集上的困惑度为 15-30。差距大约是 10 倍。这个差距就是该领域向前发展的原因。

## 使用它

- **经典 NLP 教学。** 你能获得的最清晰的平滑、最大似然估计（MLE）和困惑度入门。
- **KenLM。** 生产级 n-gram 库。在语音和机器翻译系统中用作重打分器（rescorer），适用于低延迟场景。
- **设备端自动补全。** 键盘中的 trigram 模型。至今仍在使用。
- **基线（Baselines）。** 在宣称你的神经语言模型表现良好之前，始终先计算一个 n-gram LM 的困惑度。如果你的 transformer 没有大幅超越 KN，那一定有问题。

## 交付它

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **简单。** 在 1000 句莎士比亚语料库上训练一个 trigram LM。生成 20 个句子。它们在局部上看起来合理，但在全局上是不连贯的。这是经典的演示。
2. **中等。** 在留出的莎士比亚数据集上为你的 KN 模型实现困惑度计算。与拉普拉斯平滑进行比较。你应该看到 KN 的困惑度低 30-50%。
3. **困难。** 构建一个 trigram 拼写校正器：给定一个拼写错误的词及其上下文，生成候选校正词，并根据 LM 下的上下文概率进行排序。在 Birkbeck 拼写语料库（公开可用）上进行评估。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| N-gram | 词序列 | `n` 个连续 token 组成的序列。 |
| 平滑（Smoothing） | 避免零概率 | 重新分配概率质量，使未见事件获得非零概率。 |
| 困惑度（Perplexity） | LM 质量指标 | 留出数据上的 `exp(-平均对数概率)`。越低越好。 |
| 回退（Backoff） | 回退到更短的上下文 | 如果 trigram 计数为零，则使用 bigram。Katz 回退对此进行了规范化。 |
| Kneser-Ney | n-gram 的最佳平滑方法 | 绝对折扣 + 低阶模型的延续概率。 |
| 延续概率（Continuation Probability） | KN 特有 | `P(w)` 按 `w` 出现的上下文数量加权，而不是按原始计数。 |

## 进一步阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LM 和平滑的权威论述。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立了 Kneser-Ney 为最佳 n-gram 平滑方法的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始的 KN 论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速的生产级 n-gram LM，到 2026 年仍用于延迟敏感的应用场景。
