# 实体链接与消歧（Entity Linking & Disambiguation）

> 命名实体识别（NER）找到了"Paris"。实体链接（Entity Linking）来决定：是法国巴黎（Paris, France）？帕丽斯·希尔顿（Paris Hilton）？德克萨斯州巴黎市（Paris, Texas）？还是特洛伊王子帕里斯（Paris, the Trojan prince）？没有链接，你的知识图谱（Knowledge Graph）将始终存在歧义。

**类型（Type）：** 构建（Build）
**语言（Languages）：** Python
**前置课程（Prerequisites）：** 阶段 5 · 06（NER），阶段 5 · 24（共指消解 Coreference Resolution）
**时间（Time）：** 约 60 分钟

## 问题（The Problem）

一个句子写道："Jordan beat the press." 你的 NER 将 "Jordan" 标记为 PERSON。很好。但它是*哪个* Jordan？

- 迈克尔·乔丹（Michael Jordan，篮球运动员）？
- 迈克尔·B·乔丹（Michael B. Jordan，演员）？
- 迈克尔·I·乔丹（Michael I. Jordan，伯克利机器学习教授——没错，这种混淆在 ML 论文中真实存在）？
- 约旦（Jordan，国家）？
- Jordan（希伯来语名字）？

实体链接（Entity Linking，EL）将每个提及（mention）解析到知识库（Knowledge Base，KB）中的唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域知识库。包含两个子任务：

1. **候选生成（Candidate generation）。** 给定 "Jordan"，哪些 KB 条目是合理的候选？
2. **消歧（Disambiguation）。** 给定上下文，哪个候选是正确的？

这两个步骤都是可学习的，也都有基准测试。整个流水线（pipeline）已经稳定了十年——变化的是消歧器（disambiguator）的质量。

## 概念（The Concept）

![实体链接流水线：提及 → 候选 → 消歧后的实体](../assets/entity-linking.svg)

**候选生成（Candidate generation）。** 给定提及的表面形式（surface form，"Jordan"），在别名索引（alias index）中查找候选。Wikipedia 别名词典覆盖了大多数命名实体："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK 机场、JFK（电影）。典型的索引每个提及返回 10-30 个候选。

**消歧（Disambiguation）：三种方法。**

1. **先验 + 上下文（Prior + context，Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好、速度快、无需训练。
2. **基于嵌入（Embedding-based，ESS / REL / Blink）。** 编码提及 + 上下文。编码每个候选的描述。选择最大余弦相似度。这是 2020-2024 年的默认方案。
3. **生成式（Generative，GENRE, 2021；基于 LLM, 2023+）。** 逐 token 解码实体的规范名称。通过有效实体名称的前缀树（trie）进行约束解码（constrained decoding），确保输出一定是有效的 KB ID。

**端到端（End-to-end） vs 流水线（Pipeline）。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一次前向传播中完成 NER + 候选生成 + 消歧。流水线系统在生产环境中仍然占主导地位，因为你可以替换其中的组件。

### 两个衡量指标

- **提及召回率（Mention recall，候选生成）。** 正确 KB 条目出现在候选列表中的金标准提及占比。这是整个流水线的下限。
- **消歧准确率 / F1（Disambiguation accuracy / F1）。** 给定正确的候选，top-1 正确的频率。

始终同时报告这两个指标。一个在 80% 候选召回率上达到 99% 消歧准确率的系统，实际上是一个 80% 的流水线。

## 构建它（Build It）

### 步骤 1：从 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据：约 1800 万对 (别名, 实体)。从 Wikidata 转储（dump）下载。存储为倒排索引（inverted index）。

### 步骤 2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠（Jaccard overlap）只是一个玩具示例。请替换为基于嵌入的余弦相似度（参见 `code/main.py` 中 step-2 的 transformer 版本）。

### 步骤 3：基于嵌入（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

在索引时，对每个 KB 实体进行一次嵌入。在查询时，对提及 + 上下文进行一次嵌入，与候选池做点积（dot-product），选择最大值。

### 步骤 4：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码（参见第 20 课）确保只能输出有效的标题。与 KB 支持的前缀树紧密集成。其现代后继者是 REL-GEN 和带有结构化输出的 LLM 提示式实体链接（LLM-prompted EL）。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是 2026 年最简单的可交付 EL 流水线。

### 步骤 5：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准的 EL 基准测试：1,393 篇路透社（Reuters）文章，34k 个提及，Wikipedia 实体。报告 KB 内准确率（`P@1`）和 KB 外 NIL 检测率。

## 常见陷阱（Pitfalls）

- **NIL 处理。** 某些提及不在 KB 中（新兴实体、冷门人物）。系统必须预测 NIL，而不是猜测错误的实体。需要单独衡量。
- **提及边界错误（Mention boundary errors）。** 上游 NER 遗漏了部分片段（"Bank of America" 只被标记为 "Bank"）。EL 召回率因此下降。
- **流行度偏差（Popularity bias）。** 训练后的系统过度预测高频实体。一篇 ML 论文中提到的 "Michael I. Jordan" 经常被链接到篮球运动员乔丹。
- **跨语言实体链接（Cross-lingual EL）。** 将中文文本中的提及映射到英文 Wikipedia 实体。需要多语言编码器（multilingual encoder）或翻译步骤。
- **KB 陈旧性（KB staleness）。** 新公司、事件、人物不在去年的 Wikipedia 转储中。生产流水线需要一个刷新循环。

## 使用它（Use It）

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英文 + Wikipedia | BLINK 或 REL |
| 跨语言，KB = Wikipedia | mGENRE |
| LLM 友好，每天少量提及 | 用候选列表 + 约束 JSON 提示 Claude/GPT-4 |
| 领域特定 KB（医疗、法律） | 自定义 BERT + KB 感知检索 + 在领域 AIDA 风格数据集上微调 |
| 极低延迟 | 仅精确匹配先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年可交付的生产模式：NER → 共指消解（coref）→ 对每个提及进行 EL → 将簇（cluster）折叠为每个簇一个规范实体。输出：文档中每个实体一个 KB ID，而不是每个提及一个。

## 交付它（Ship It）

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习（Exercises）

1. **简单（Easy）。** 在 `code/main.py` 中对 10 个歧义提及（Paris、Jordan、Apple）实现先验 + 上下文消歧器。手动标注正确实体。衡量准确率。
2. **中等（Medium）。** 用 sentence transformer 编码 50 个歧义提及。嵌入每个候选的描述。比较基于嵌入的消歧与 Jaccard 上下文重叠。
3. **困难（Hard）。** 构建一个 1k 实体的领域 KB（例如你公司的员工 + 产品）。实现端到端的 NER + EL。在 100 个留出句子上衡量精确率（precision）和召回率（recall）。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 实体链接（Entity linking, EL） | 链接到 Wikipedia | 将提及映射到唯一的 KB 条目。 |
| 候选生成（Candidate generation） | 可能是谁？ | 为提及返回一个合理的 KB 条目短列表。 |
| 消歧（Disambiguation） | 选出正确的那个 | 使用上下文对候选打分，选出胜者。 |
| 别名索引（Alias index） | 查找表 | 从表面形式映射到候选实体。 |
| NIL | 不在 KB 中 | 明确预测没有匹配的 KB 条目。 |
| KB（Knowledge base） | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域知识库。 |
| AIDA-CoNLL | 基准测试 | 1,393 篇路透社文章，带有金标准实体链接。 |

## 延伸阅读（Further Reading）

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 奠基性的先验 + 上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入的主力方案。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带约束解码的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准测试论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开源生产级技术栈。
