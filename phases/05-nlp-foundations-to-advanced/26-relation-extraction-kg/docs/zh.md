# 关系抽取与知识图谱构建（Relation Extraction & Knowledge Graph Construction）

> 命名实体识别（NER）找到了实体。实体链接（Entity Linking）锚定了它们。关系抽取（Relation Extraction）找到它们之间的边。知识图谱（Knowledge Graph）是节点、边及其来源的总和。

**类型（Type）：** Build
**语言（Languages）：** Python
**前置条件（Prerequisites）：** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**时间（Time）：** ~60 minutes

## 问题（The Problem）

一位分析师读到："Tim Cook became CEO of Apple in 2011." 其中包含四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（Relation Extraction，RE）将自由文本转化为结构化三元组（triple）`(subject, relation, object)`。在语料库中聚合这些三元组，你就得到了一个知识图谱。聚合并查询，你就得到了用于 RAG、分析或合规审计的推理基础。

2026 年的问题：大语言模型（LLM）抽取关系时过于热情——过于热情。它们会幻觉出源文本并不支持的三元组。没有来源追溯（provenance），你无法区分真实的三元组和看似合理的虚构。2026 年的答案是 AEVS 风格的锚定-验证流水线。

## 概念（The Concept）

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系可以来自封闭本体（closed ontology，如 Wikidata 属性、FIBO、UMLS），也可以来自开放集合（OpenIE 风格，任意关系均可）。

**三种抽取方法。**

1. **基于规则/模式（Rule / pattern-based）。** Hearst 模式："X such as Y" → `(Y, isA, X)`。外加手工编写的正则表达式。脆弱、精确、可解释。
2. **有监督分类器（Supervised classifier）。** 给定句子中的两个实体提及，从固定集合中预测关系。在 TACRED、ACE、KBP 上训练。2015–2022 年的标准方法。
3. **生成式大语言模型（Generative LLM）。** 提示模型输出三元组。开箱即用。需要来源追溯，否则会幻觉出看似合理的垃圾数据。

**AEVS（Anchor-Extraction-Verification-Supplement，锚定-抽取-验证-补充，2026）。** 当前缓解幻觉的框架：

- **锚定（Anchor）。** 用精确位置标识每个实体片段和关系短语片段。
- **抽取（Extract）。** 生成链接到锚定片段的三元组。
- **验证（Verify）。** 将每个三元组元素匹配回源文本；拒绝任何无依据的内容。
- **补充（Supplement）。** 覆盖度检查确保没有锚定片段被遗漏。

幻觉率大幅下降。需要更多计算资源，但可审计。

**开放与封闭的权衡。**

- **封闭本体（Closed ontology）。** 固定的属性列表（例如 Wikidata 的 11,000+ 属性）。可预测。可查询。难以编造。
- **开放信息抽取（Open IE）。** 任何动词短语都可以成为关系。高召回率。低精确率。查询混乱。

生产环境中的知识图谱通常混合使用：用开放信息抽取进行发现，然后在合并到主图谱之前将关系规范化到封闭本体上。

## 动手构建（Build It）

### 步骤 1：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整的玩具抽取器见 `code/main.py`。Hearst 模式至今仍在特定领域的流水线中使用，因为它们可调试。

### 步骤 2：有监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个序列到序列（seq2seq）关系抽取器：文本输入，三元组输出，已经使用 Wikidata 属性 ID。在远程监督（distant supervision）数据上微调。标准的开放权重基线模型。

### 步骤 3：带锚定的 LLM 提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

将每个返回的片段与源文本进行验证。拒绝任何 `text[start:end] != triple_entity` 的情况。这就是 AEVS 中"验证"步骤的最简形式。

### 步骤 4：规范化到封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化（Canonicalization）通常占工程工作量的 60-80%。请为此预留时间。

### 步骤 5：构建一个小型图谱并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个基于知识图谱的 RAG（KG-RAG）系统的最小单元。可以使用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图存储来扩展它。

## 常见陷阱（Pitfalls）

- **关系抽取前先做共指消解。** "He founded Apple" — 关系抽取需要知道 "he" 是谁。先运行共指消解（coref，第 24 课）。
- **实体规范化。** "Apple Inc" 和 "Apple" 必须解析为同一个节点。先做实体链接（第 25 课）。
- **幻觉三元组。** LLM 会输出文本不支持的三元组。强制执行片段验证。
- **关系规范化漂移。** 开放信息抽取的关系不一致（"was born in," "came from," "is a native of"）。将其折叠为规范 ID，否则图谱无法查询。
- **时间错误。** "Tim Cook is CEO of Apple" — 现在为真，2005 年为假。许多关系是有时间边界的。使用限定符（Wikidata 中的 `P580` 开始时间、`P582` 结束时间）。
- **领域不匹配。** REBEL 在 Wikipedia 上训练。法律、医学和科学文本通常需要领域微调的关系抽取模型。

## 使用指南（Use It）

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 快速生产环境，通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 特定领域（生物医学、法律） | SciREX 风格的领域微调 + 自定义本体 |
| LLM 提示抽取，可审计输出 | AEVS 流水线：锚定 → 抽取 → 验证 → 补充 |
| 高容量新闻信息抽取 | 基于模式 + 有监督混合方案 |
| 从零构建知识图谱 | 开放信息抽取 + 手动规范化处理 |
| 时序知识图谱 | 带限定符抽取（开始/结束时间、时间点） |

集成模式：NER → 共指消解 → 实体链接 → 关系抽取 → 本体映射 → 图谱加载。每个阶段都是一个潜在的质量关卡。

## 交付物（Ship It）

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习（Exercises）

1. **简单。** 在 5 句新闻文章句子上运行 `code/main.py` 中的模式抽取器。手工检查精确率。
2. **中等。** 在相同句子上使用 REBEL（或一个小型 LLM）。比较三元组。哪个抽取器的精确率更高？召回率更高？
3. **困难。** 构建 AEVS 流水线：用 LLM 抽取 + 对照源文本验证片段。在 50 句 Wikipedia 风格的句子上，测量验证步骤前后的幻觉率。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Triple（三元组） | 主语-关系-宾语 | `(s, r, o)` 元组，是知识图谱的原子单元。 |
| Open IE（开放信息抽取） | 抽取任何内容 | 开放词汇的关系短语；高召回率，低精确率。 |
| Closed ontology（封闭本体） | 固定模式 | 有界的关系类型集合（Wikidata、UMLS、FIBO）。 |
| Canonicalization（规范化） | 标准化一切 | 将表面名称/关系映射到规范 ID。 |
| AEVS | 有依据的抽取 | 锚定-抽取-验证-补充流水线（2026）。 |
| Provenance（来源追溯） | 真实来源链接 | 每个三元组携带文档 ID + 字符片段指向其来源。 |
| Distant supervision（远程监督） | 廉价标签 | 将文本与现有知识图谱对齐以创建训练数据。 |

## 延伸阅读（Further Reading）

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq 关系抽取主力模型。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合信息抽取。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 年缓解幻觉的设计方案。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图谱查询教程。
