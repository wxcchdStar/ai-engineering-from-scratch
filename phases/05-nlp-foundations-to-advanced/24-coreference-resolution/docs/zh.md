# 指代消解（Coreference Resolution）

> "She called him. He did not answer. The doctor was at lunch." 三处引用指向两个人，且没有人被指名道姓。指代消解（Coreference Resolution）的任务就是弄清楚谁是谁。

**类型（Type）：** Learn
**语言（Languages）：** Python
**前置要求（Prerequisites）：** Phase 5 · 06（NER），Phase 5 · 07（POS 与句法分析）
**时间（Time）：** 约 60 分钟

## 问题（The Problem）

从一篇 300 词的文章中提取 Apple Inc. 的每一次提及（mention）。当文章中出现 "Apple" 时很简单，但当出现 "the company"、"they"、"Cupertino's technology giant" 或 "Jobs's firm" 时就很难了。如果不将这些提及消解到同一实体（entity），你的 NER 流水线将遗漏 60%–80% 的提及。

指代消解将所有指向同一现实世界实体的表达式链接到一个簇（cluster）中。它是表层 NLP（NER、句法分析）与下游语义任务（信息抽取 IE、问答 QA、摘要、知识图谱 KG）之间的粘合剂。

为什么在 2026 年它很重要：

- 摘要（Summarization）："The CEO announced..." 与 "Tim Cook announced..."——摘要应当明确指出 CEO 是谁。
- 问答（Question answering）："Who did she call?" 需要先消解 "she"。
- 信息抽取（Information extraction）：一个知识图谱中同时存在 "PER1 founded Apple" 和 "Jobs founded Apple" 两条独立条目是错误的。
- 多文档信息抽取（Multi-document IE）：跨文章合并关于同一事件的提及，属于跨文档指代消解（cross-document coreference）。

## 概念（The Concept）

![指代消解聚类：提及 → 实体](../assets/coref.svg)

**任务定义。** 输入：一篇文档。输出：提及（span）的聚类结果，其中每个簇对应一个实体。

**提及类型（Mention types）。**

- **命名实体（Named entity）。** "Tim Cook"
- **名词性（Nominal）。** "the CEO"、"the company"
- **代词性（Pronominal）。** "he"、"she"、"they"、"it"
- **同位语（Appositive）。** "Tim Cook, Apple's CEO,"

**架构（Architectures）。**

1. **基于规则（Rule-based，Hobbs, 1978）。** 基于句法树的代词消解，使用语法规则。是一个不错的基线（baseline）。在代词消解上出人意料地难以被超越。
2. **提及对分类器（Mention-pair classifier）。** 对每一对提及 (m_i, m_j)，预测它们是否共指（corefer）。通过传递闭包（transitive closure）进行聚类。2016 年之前的标准做法。
3. **提及排序（Mention-ranking）。** 对每个提及，对候选先行语（antecedent）进行排序（包括"无先行语"）。选择得分最高的。
4. **基于片段（Span）的端到端方法（Lee et al., 2017）。** Transformer 编码器。枚举所有候选片段（有长度上限）。预测提及得分。为每个片段预测先行语概率。贪心聚类。现代默认方案。
5. **生成式（Generative，2024+）。** 向 LLM 发送提示："列出本文中每个代词及其先行语。"在简单场景下效果不错，但在长文档和罕见指代对象上表现不佳。

**评估指标（Evaluation metrics）。** 有五种标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有任何单一指标能完整衡量聚类质量。通常以前三种指标的平均值作为 CoNLL F1 进行报告。2026 年在 CoNLL-2012 上的最先进水平（state-of-the-art）：约 83 F1。

**已知的困难场景。**

- 定指描述（Definite descriptions）指向前文多页之前引入的实体。
- 桥接回指（Bridging anaphora）（"the wheels" → 前文提到的某辆车）。
- 中文和日语等语言中的零回指（Zero anaphora）。
- 后指（Cataphora，代词出现在指代对象之前）："When **she** walked in, Mary smiled."

## 动手构建（Build It）

### 步骤 1：预训练神经指代消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长的文档上，你会得到类似以下的结果：
- 簇 1：[Apple, The company, they]
- 簇 2：[new products]

### 步骤 2：基于规则的代词消解器（教学用途）

参见 `code/main.py` 中的纯标准库实现：

1. 提取提及：命名实体（大写片段）、代词（字典查找）、定指描述（"the X"）。
2. 对每个代词，查看前 K 个提及并按以下规则打分：
   - 性别/数的一致性（启发式）
   - 近因性（越近得分越高）
   - 句法角色（优先选择主语）
3. 链接得分最高的先行语。

虽然无法与神经模型竞争，但它展示了搜索空间以及端到端模型必须做出的决策。

### 步骤 3：使用 LLM 进行指代消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

需要注意两种失败模式。第一，LLM 会过度合并（over-merge）（将指向两个不同人的 "him" 和 "her" 合并）。第二，LLM 在长文档中会静默丢弃提及。务必通过片段偏移量检查（span-offset checks）进行验证。

### 步骤 4：评估（Evaluation）

标准的 conll-2012 脚本会计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，可以先从标注测试集上的片段级精确率（precision）和召回率（recall）开始，然后加入提及链接 F1。

## 常见陷阱（Pitfalls）

- **单例爆炸（Singleton explosion）。** 某些系统会将每个提及都报告为独立的簇。B³ 对此较为宽容，而 MUC 会严厉惩罚。务必同时检查所有三个指标。
- **长上下文中的代词。** 在超过 2,000 个 token 的文档上，性能会下降约 15 F1。需要谨慎地进行分块（chunk）。
- **性别假设。** 硬编码的性别规则在非二元性别指代对象、组织、动物上会失效。应使用学习型模型或中性评分。
- **LLM 在长文档上的漂移。** 单次 API 调用无法可靠地在 50 段以上的文档中聚类提及。应使用滑动窗口（sliding-window）+ 合并（merge）策略。

## 实际使用（Use It）

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 英文，单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神经指代消解 |
| 多语言 | 在 OntoNotes 或多语言 CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件指代消解 | 专用端到端模型（2025–26 最先进水平） |
| 快速 LLM 基线 | GPT-4o / Claude 配合结构化输出指代消解提示 |
| 生产环境对话系统 | 基于规则的回退（fallback）+ 神经模型主力 + 关键槽位人工审核 |

2026 年上线的集成模式：先运行 NER，再运行指代消解，将指代消解簇合并到 NER 实体中。下游任务看到的是每个簇对应一个实体，而非每个提及对应一个实体。

## 交付（Ship It）

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习（Exercises）

1. **简单。** 在 5 段手工编写的段落上运行 `code/main.py` 中的基于规则的消解器。对照真实标注（ground truth）衡量提及链接准确率。
2. **中等。** 在一篇新闻文章上使用预训练神经指代消解模型。将聚类结果与你自己的手工标注进行对比。它在哪些地方失败了？
3. **困难。** 构建一个指代消解增强的 NER 流水线：先做 NER，然后通过指代消解簇进行合并。在 100 篇文章上衡量相比纯 NER 的实体覆盖率提升。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Mention（提及） | 一个引用 | 指向某个实体的一段文本（名称、代词、名词短语）。 |
| Antecedent（先行语） | "it" 所指的对象 | 后文提及与之共指的更早出现的提及。 |
| Cluster（簇） | 实体的提及集合 | 全部指向同一现实世界实体的提及集合。 |
| Anaphora（回指） | 向后引用 | 后文提及指向前文（"he" → "John"）。 |
| Cataphora（后指） | 向前引用 | 前文提及指向后文（"When he arrived, John..."）。 |
| Bridging（桥接） | 隐式引用 | "I bought a car. The wheels were bad."（那辆车的轮子。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1 得分的平均值。 |

## 延伸阅读（Further Reading）

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 经典教材章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 基于片段的端到端方法。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 改善指代消解的预训练方法。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 基准评测。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典方法。
