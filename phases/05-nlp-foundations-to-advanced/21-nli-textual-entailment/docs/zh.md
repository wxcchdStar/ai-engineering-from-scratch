# 自然语言推理 — 文本蕴含（Textual Entailment）

> "t entails h" 意味着一个人类读者在阅读 t 后会得出 h 为真的结论。自然语言推理（NLI）是预测蕴含（entailment）/ 矛盾（contradiction）/ 中立（neutral）的任务。表面上枯燥，但在生产环境中承担着关键作用。

**类型：** 学习
**语言：** Python
**前置要求：** 第 5 阶段 · 05（情感分析 Sentiment Analysis），第 5 阶段 · 13（问答 Question Answering）
**时间：** 约 60 分钟

## 问题

你构建了一个摘要生成器。它生成了一篇摘要。你如何知道该摘要不包含幻觉（hallucination）？

你构建了一个聊天机器人。它回答了"是"。你如何知道该回答是否得到了检索到的段落（retrieved passage）的支持？

你需要按主题对 10,000 篇新闻文章进行分类。你没有训练标签。你能复用一个模型吗？

这三个问题都可以归结为自然语言推理（Natural Language Inference，NLI）。NLI 的问题是：给定一个前提（premise）`t` 和一个假设（hypothesis）`h`，`h` 是被 `t` 蕴含（entailed）、矛盾（contradicted），还是中立（neutral，即无关）？

- **幻觉检查：** `t` = 源文档，`h` = 摘要中的声明。不蕴含 = 幻觉。
- **有依据的问答（Grounded QA）：** `t` = 检索到的段落，`h` = 生成的答案。不蕴含 = 编造。
- **零样本分类（Zero-shot classification）：** `t` = 文档，`h` = 语言化标签（"这是关于体育的"）。蕴含 = 预测标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架在底层都内置了一个 NLI 模型。

## 概念

![NLI：三分类，前提 vs 假设](../assets/nli.svg)

**三种标签。**

- **蕴含（Entailment）。** `t` → `h`。"猫在垫子上"蕴含"有一只猫"。
- **矛盾（Contradiction）。** `t` → ¬`h`。"猫在垫子上"与"没有猫"矛盾。
- **中立（Neutral）。** 无法推断。"猫在垫子上"与"猫饿了"中立。

**不是逻辑蕴含。** NLI 是*自然*语言推理——一个典型的人类读者会推断出什么，而不是严格的逻辑推理。"John walked his dog"在 NLI 中蕴含"John has a dog"，但严格的一阶逻辑（first-order logic）只有在将所有权公理化后才会承认这一点。

**数据集。**

- **SNLI**（2015）。57 万条人工标注的句子对，以图片说明作为前提。领域较窄。
- **MultiNLI**（2017）。43.3 万条句子对，涵盖 10 个领域。2026 年的标准训练语料库。
- **ANLI**（2019）。对抗性 NLI（Adversarial NLI）。人类专门编写了旨在攻破现有模型的示例。难度更高。
- **DocNLI、ConTRoL**（2020–21）。文档级前提。测试多跳（multi-hop）和长距离推理。

**架构。** 一个 Transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表示输入一个 3 路 softmax。在 MNLI 上训练，在留出的基准上评估，在分布内句子对上达到 90% 以上的准确率。

**通过 NLI 实现零样本分类。** 给定一个文档和候选标签，将每个标签转化为一个假设（"This text is about sports"）。计算每个假设的蕴含概率。选择最大值。这就是 Hugging Face 的 `zero-shot-classification` 流水线背后的机制。

## 动手构建

### 步骤 1：运行预训练的 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

对于生产环境中的 NLI，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是开源默认选择。DeBERTa-v3 在排行榜上名列前茅。

### 步骤 2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是 "This example is about {label}."。可以通过 `hypothesis_template` 自定义。无需训练数据。无需微调（fine-tuning）。开箱即用。

### 步骤 3：RAG 的忠实性（faithfulness）检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实性评估的核心。将生成的答案拆分为原子声明（atomic claims）。将每个声明与检索到的上下文进行比对。报告蕴含的比例。

### 步骤 4：手写 NLI 分类器（概念性）

参见 `code/main.py` 中一个仅使用标准库的玩具实现：通过词汇重叠（lexical overlap）+ 否定检测（negation detection）来比较前提和假设。无法与 Transformer 模型竞争——但它展示了任务的基本形态：两个文本输入，3 路标签输出，损失函数 = 在 `{entail, contradict, neutral}` 上的交叉熵（cross-entropy）。

## 常见陷阱

- **仅依赖假设的捷径（Hypothesis-only shortcuts）。** 模型仅凭假设就能在 SNLI 上达到约 60% 的准确率，因为"not"、"nobody"、"never"等词与矛盾相关。这是检测标签泄露（label leakage）的强基线。
- **词汇重叠启发式（Lexical overlap heuristic）。** 子序列启发式（"每个子序列都被蕴含"）能通过 SNLI，但在 HANS/ANLI 上失败。应使用对抗性基准（adversarial benchmarks）。
- **文档级退化。** 单句 NLI 模型在文档级前提上 F1 下降 20 以上。对长上下文应使用 DocNLI 训练的模型。
- **零样本模板敏感性。** "This example is about {label}" vs "{label}" vs "The topic is {label}" 可能导致准确率波动 10 个点以上。需要调优模板。
- **领域不匹配。** MNLI 在通用英语上训练。法律、医学和科学文本需要领域特定的 NLI 模型（例如 SciNLI、MedNLI）。

## 使用指南

2026 年的技术栈：

| 用例 | 模型 |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速 / 边缘设备 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量级） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中的幻觉检测 | RAGAS / DeepEval 内部的 NLI 层 |

2026 年的元模式：NLI 是文本理解的万能胶带。每当你需要判断"A 是否支持 B？"或"A 是否与 B 矛盾？"——在调用另一个 LLM 之前，先考虑使用 NLI。

## 交付物

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习

1. **简单。** 在 20 个手工构建的（前提，假设，标签）三元组上运行 `facebook/bart-large-mnli`，覆盖所有三个类别。测量准确率。添加对抗性"子序列启发式"陷阱（"I did not eat the cake" vs "I ate the cake"），观察它是否被攻破。
2. **中等。** 在 100 条 AG News 标题上比较零样本模板 `"This text is about {label}"`、`"The topic is {label}"` 和 `"{label}"`。报告准确率波动。
3. **困难。** 构建一个 RAG 忠实性检查器：原子声明分解 + 逐声明 NLI。在 50 个带有金标准上下文的 RAG 生成答案上进行评估。测量与人工标签相比的假阳性率和假阴性率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | 自然语言推理（Natural Language Inference） | 前提-假设关系的 3 路分类。 |
| RTE | 识别文本蕴含（Recognizing Textual Entailment） | NLI 的旧称；同一任务。 |
| 蕴含（Entailment） | "t 蕴含 h" | 一个典型读者在给定 t 的情况下会得出 h 为真的结论。 |
| 矛盾（Contradiction） | "t 排除 h" | 一个典型读者在给定 t 的情况下会得出 h 为假的结论。 |
| 中立（Neutral） | "无法判定" | 从 t 到 h 无法做出任何推断。 |
| 零样本分类（Zero-shot classification） | 将 NLI 用作分类器 | 将标签语言化为假设，选择蕴含概率最大的。 |
| 忠实性（Faithfulness） | 答案是否有依据？ | 对（检索到的上下文，生成的答案）进行 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI 基准。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI 作为分类器。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 年的 NLI 主力模型。
