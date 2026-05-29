# 文本摘要（Text Summarization）

> 抽取式（Extractive）系统告诉你文档说了什么。生成式（Abstractive）系统告诉你作者想表达什么。不同的任务，不同的陷阱。

**类型：** 构建（Build）
**语言：** Python
**前置要求：** 阶段 5 · 02（词袋模型与 TF-IDF），阶段 5 · 11（机器翻译）
**时间：** 约 75 分钟

## 问题

一篇 2000 词的新闻文章出现在你的信息流中。你需要用 120 个词来概括它。你可以从文章中挑选最重要的三个句子（抽取式），或者用自己的话重写内容（生成式）。两者都叫摘要（Summarization），但它们是完全不同的问题。

抽取式摘要是一个排序问题。对每个句子打分，返回得分最高的 `k` 个句子。输出总是语法正确的，因为它是逐字提取的。风险在于会遗漏分散在文章各处的信息。

生成式摘要是一个生成问题。一个 Transformer 模型根据输入生成新的文本。输出流畅且具有压缩性，但可能会产生源文本中不存在的事实幻觉（Hallucination）。风险在于自信地编造内容。

本课将构建这两种方法，并展示各自固有的失败模式。

## 概念

![抽取式 TextRank vs 生成式 Transformer](../assets/summarization.svg)

**抽取式。** 将文章视为一个图（Graph），其中节点是句子，边是相似度。在图上面运行 PageRank（或类似算法），根据句子与其他所有内容的连接程度来打分。得分最高的句子构成摘要。经典实现是 **TextRank**（Mihalcea 和 Tarau，2004）。

**生成式。** 在文档-摘要对（document-summary pairs）上微调一个 Transformer 编码器-解码器（Encoder-Decoder）模型（BART、T5、Pegasus）。推理时，模型读取文档并通过交叉注意力（Cross-Attention）逐 token 生成摘要。Pegasus 特别使用了间隔句子预训练（Gap-Sentence Pretraining）目标，使其在无需大量微调的情况下就能出色地完成摘要任务。

使用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation，面向召回率的摘要评估替代指标）进行评估。ROUGE-1 和 ROUGE-2 分别计算一元组（Unigram）和二元组（Bigram）的重叠度。ROUGE-L 计算最长公共子序列（Longest Common Subsequence）。分数越高越好，但 ROUGE-L 达到 40 就算"好"，达到 50 就算"出色"。每篇论文都会报告这三项指标。使用 `rouge-score` 包。

## 构建

### 步骤 1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两点值得说明。相似度函数使用了对数归一化的词重叠度（Log-Normalized Word Overlap），这是原始 TextRank 的变体。TF-IDF 向量的余弦相似度（Cosine Similarity）也可以使用。阻尼因子（Damping Factor）0.85 和迭代次数是 PageRank 的默认值。

### 步骤 2：使用 BART 进行生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上进行了微调。它开箱即用地生成新闻风格的摘要。对于其他领域（科学论文、对话、法律），请使用相应的 Pegasus 检查点（Checkpoint），或在你的目标数据上进行微调。

### 步骤 3：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取（Stemming）。如果不使用，"running" 和 "run" 会被视为不同的词，导致 ROUGE 低估分数。

### 超越 ROUGE（2026 年摘要评估）

ROUGE 作为主导的摘要评估指标已有二十年，但在 2026 年，仅靠它是不够的。一项针对 NLG 论文的大规模元分析（Meta-Analysis）显示：

- **BERTScore**（上下文嵌入相似度，Contextual Embedding Similarity）在 2023 年之前逐渐普及，现在大多数摘要论文中与 ROUGE 一起报告。
- **BARTScore** 将评估视为生成任务：根据预训练的 BART 模型在给定源文本的情况下为摘要分配的概率来评分。
- **MoverScore**（基于上下文嵌入的推土机距离，Earth Mover's Distance over Contextual Embeddings）在 2025 年的摘要基准测试中名列前茅，因为它比 ROUGE 更好地捕捉了语义重叠。
- **FactCC** 和**基于问答的事实性评估（QA-based Faithfulness）**在 2021-2023 年间很常见，现在通常被 **G-Eval**（一个 GPT-4 提示链，通过思维链推理对连贯性、一致性、流畅性和相关性进行评分）所取代。
- **G-Eval** 和类似的 LLM 评判方法在评分标准设计良好的情况下，与人类判断的一致性约为 80%。

生产环境建议：报告 ROUGE-L 用于与历史结果对比，报告 BERTScore 用于语义重叠评估，报告 G-Eval 用于连贯性和事实性评估。用 50-100 条人工标注的摘要进行校准。

### 步骤 4：事实性问题

生成式摘要容易产生幻觉。抽取式摘要的幻觉风险要低得多，因为输出是从源文本中逐字提取的，但如果源句子被去语境化、过时或引用顺序错乱，它们仍然可能产生误导。这是生产系统在涉及合规相关内容时仍然偏好抽取式方法的最重要原因。

需要了解的幻觉类型：

- **实体替换（Entity Swap）。** 源文本说 "John Smith"。摘要说 "John Brown"。
- **数字漂移（Number Drift）。** 源文本说 "25,000"。摘要说 "25 million"。
- **极性翻转（Polarity Flip）。** 源文本说 "rejected the offer"。摘要说 "accepted the offer"。
- **事实编造（Fact Invention）。** 源文本未提及 CEO。摘要却说 CEO 批准了。

有效的评估方法：

- **FactCC。** 一个在源句子和摘要句子之间的蕴含关系（Entailment）上训练的二分类器。预测事实性/非事实性。
- **基于问答的事实性评估。** 向问答模型提问，问题的答案在源文本中。如果摘要支持不同的答案，则标记。
- **实体级别 F1。** 比较源文本和摘要中的命名实体（Named Entities）。仅出现在摘要中的实体是可疑的。

对于任何面向用户且事实性至关重要的场景（新闻、医疗、法律、金融），抽取式是更安全的默认选择。生成式需要在流程中加入事实性检查。

## 使用

2026 年的技术栈：

| 使用场景 | 推荐方案 |
|---------|-------------|
| 新闻，3-5 句摘要，英文 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或微调后的 T5 |
| 多文档、长文本 | 任何具有 32k+ 上下文的 LLM，通过提示词驱动 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，从结构上降低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

在 2026 年，当计算资源不是限制因素时，具有长上下文的 LLM 通常优于专用模型。权衡在于成本和可复现性；专用模型能提供更一致的输出。

## 交付

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. **简单。** 在 5 篇新闻文章上运行 TextRank。将得分最高的 3 个句子与参考摘要进行比较。测量 ROUGE-L。在 CNN/DailyMail 风格的文章上，你应该能看到 30-45 的 ROUGE-L 分数。
2. **中等。** 实现实体级别的事实性评估：从源文本和摘要中提取命名实体（使用 spaCy），计算源实体在摘要中的召回率（Recall）以及摘要实体相对于源文本的精确率（Precision）。高精确率和低召回率意味着安全但简略；低精确率意味着存在幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 和 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（通过实体 F1）以及每条摘要的成本。记录各自在哪些方面胜出。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Extractive | 挑选句子 | 从源文本中逐字返回句子。永远不会产生幻觉。 |
| Abstractive | 重写 | 基于源文本生成新文本。可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考摘要之间的 N-gram / LCS 重叠度。 |
| TextRank | 基于图的抽取式方法 | 在句子相似度图上运行 PageRank。 |
| Factuality | 是否正确 | 摘要中的断言是否有源文本支持。 |
| Hallucination | 编造的内容 | 摘要中源文本不支持的內容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式摘要的经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 及其间隔句子预训练目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 事实性全景论文。
