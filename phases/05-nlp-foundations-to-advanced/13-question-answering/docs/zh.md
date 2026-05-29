# 问答系统（Question Answering Systems）

> 三种系统塑造了现代问答（QA）。抽取式（Extractive）找到答案片段。检索增强（Retrieval-Augmented）将答案锚定在文档中。生成式（Generative）直接产出答案。每一个现代 AI 助手都是这三者的混合体。

**类型：** 构建（Build）
**语言：** Python
**前置课程：** 第 5 阶段 · 11（机器翻译），第 5 阶段 · 10（注意力机制）
**时间：** 约 75 分钟

## 问题

用户输入"When did the first iPhone launch?"，期望得到"June 29, 2007"。而不是"Apple 的历史悠久且丰富多彩"。也不是孤零零的"2007"，没有完整的句子。用户要的是一个直接、有依据、正确的答案。

在过去十年中，三种架构主导了问答领域。

- **抽取式问答（Extractive QA）。** 给定一个问题和一个已知包含答案的段落，在段落中找到答案片段的起始和结束索引。SQuAD 是经典的基准数据集。
- **开放域问答（Open-domain QA）。** 不提供段落。先检索相关段落，然后抽取或生成答案。这是当今每一个 RAG 流水线的基础。
- **生成式 / 闭卷问答（Generative / Closed-book QA）。** 一个大语言模型（LLM）从其参数化记忆（Parametric Memory）中回答问题。无需检索。推理速度最快，但在事实方面最不可靠。

2026 年的趋势是混合模式：检索最佳的几个段落，然后提示一个生成式模型基于这些段落来回答。这就是 RAG（Retrieval-Augmented Generation，检索增强生成），第 14 课将深入讲解检索部分。本课构建的是问答部分。

## 概念

![QA 架构：抽取式、检索增强、生成式](../assets/qa.svg)

**抽取式（Extractive）。** 使用 Transformer（BERT 家族）将问题和段落一起编码。训练两个头（head），分别预测答案的起始和结束 token 索引。损失函数是有效位置上的交叉熵（Cross-Entropy）。输出是段落中的一个片段。从不产生幻觉（Hallucination）（由架构保证），也从不处理段落无法回答的问题（由架构保证）。

**检索增强（RAG）。** 分为两个阶段。首先，检索器（Retriever）从语料库中找到 top-`k` 个段落。然后，阅读器（Reader）（抽取式或生成式）使用这些段落生成答案。检索器与阅读器的分离使得两者可以独立训练和评估。现代 RAG 通常会在两者之间加入一个重排序器（Reranker）。

**生成式（Generative）。** 一个仅解码器（Decoder-only）的 LLM（如 GPT、Claude、Llama）从学习到的权重中回答问题。没有检索步骤。对常识性知识表现优异，对罕见或近期事实则可能灾难性失败。幻觉率与预训练数据中事实的出现频率呈负相关。

## 构建

### 步骤 1：使用预训练模型进行抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，该数据集包含不可回答的问题。默认情况下，`question-answering` 流水线会返回得分最高的片段，即使模型的空答案（null）得分更高——它*不会*自动返回空答案。要获得显式的"无答案"行为，请在调用流水线时传入 `handle_impossible_answer=True`：此时流水线仅在空答案得分超过所有片段得分时才返回空答案。无论哪种方式，都应始终检查 `score` 字段。

### 步骤 2：检索增强流水线（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段流水线。稠密检索器（Dense Retriever，Sentence-BERT）通过语义相似度找到相关段落。抽取式阅读器（RoBERTa-SQuAD）从合并后的 top 段落中提取答案片段。适用于小型语料库。对于百万级文档的语料库，请使用 FAISS 或向量数据库（Vector Database）。

### 步骤 3：结合 RAG 的生成式问答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示词模式至关重要。明确告诉模型要基于上下文回答，并在上下文不足时返回"I don't know"，与简单提示相比，可以将幻觉率降低 40-60%。更精细的模式还会添加引用、置信度分数和结构化提取。

### 步骤 4：反映真实世界的评估

SQuAD 使用**精确匹配（Exact Match, EM）**和**token 级 F1**。EM 是经过规范化（小写、去除标点、移除冠词）后的严格匹配——要么预测完全匹配，要么得分为 0。F1 基于预测与参考答案之间的 token 重叠计算，给予部分得分。两者都对释义性答案评分偏低："June 29, 2007"与"June 29th, 2007"通常 EM 得分为 0（序数词破坏了规范化），但由于 token 重叠，仍能获得可观的 F1 分数。

对于生产环境的问答系统：

- **答案准确度（Answer Accuracy）**（由 LLM 或人工评判，因为自动化指标无法捕捉语义等价）。
- **引用准确度（Citation Accuracy）。** 引用的段落是否确实支持该答案？通过生成引用与检索段落之间的字符串匹配即可轻松自动检查。
- **拒绝校准（Refusal Calibration）。** 当检索到的段落中没有答案时，系统是否正确地说"I don't know"？衡量错误自信率。
- **检索召回率（Retrieval Recall）。** 在评估阅读器之前，先衡量检索器是否将正确的段落放入了 top-`k`。阅读器无法弥补缺失的段落。

### RAGAS：2026 年的生产环境评估框架

`RAGAS` 专为 RAG 系统构建，是 2026 年的默认发布标准。它在不需要黄金参考答案的情况下对四个维度进行评分：

- **忠实度（Faithfulness）。** 答案中的每个断言是否都来自检索到的上下文？通过基于 NLI（Natural Language Inference，自然语言推理）的蕴含关系来衡量。这是你的主要幻觉指标。
- **答案相关性（Answer Relevance）。** 答案是否回应了问题？通过从答案生成假设性问题并与真实问题进行比较来衡量。
- **上下文精确度（Context Precision）。** 在检索到的文本块中，有多少比例是真正相关的？低精确度 = 提示词中存在噪声。
- **上下文召回率（Context Recall）。** 检索到的集合是否包含了所有必要信息？低召回率 = 阅读器无法成功。

无参考答案的评分方式让你可以在没有精心标注的黄金答案的情况下，对线上生产流量进行评估。对于开放式问题（精确匹配指标无效的场景），可以在其上叠加 LLM-as-judge。

`pip install ragas`。接入你的检索器 + 阅读器。每个查询获得四个标量值。对指标退化设置告警。

## 使用

2026 年的技术栈。

| 用例 | 推荐方案 |
|---------|-------------|
| 给定段落，找到答案片段 | `deepset/roberta-base-squad2` |
| 在固定语料库上，不接受闭卷回答 | RAG：稠密检索器 + LLM 阅读器 |
| 对文档存储进行实时查询 | RAG 配合混合检索器（Hybrid Retriever，BM25 + 稠密）+ 重排序器（第 14 课） |
| 对话式问答（Conversational QA，含追问） | LLM 配合对话历史 + 每轮 RAG |
| 高事实性要求的受监管领域 | 基于权威语料库的抽取式问答；绝不单独使用生成式 |

抽取式问答在 2026 年已不再流行，因为配合 LLM 的 RAG 能处理更多场景。但它仍然在需要逐字引用的场景中交付：法律研究、合规监管、审计工具。

## 交付

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在 10 个维基百科段落上搭建上述 SQuAD 抽取式流水线。手工编写 10 个问题。衡量答案正确的频率。如果段落和问题都清晰，你应该能看到 7-9 个正确答案。
2. **中等。** 添加一个拒绝分类器。当最高检索得分低于某个阈值（例如余弦相似度 0.3）时，返回"I don't know"而不是调用阅读器。在留出集上调优该阈值。
3. **困难。** 在你选择的 10,000 篇文档语料库上构建一个 RAG 流水线。实现混合检索（Hybrid Retrieval，BM25 + 稠密）并使用 RRF（Reciprocal Rank Fusion，倒数排名融合）进行融合（参见第 14 课）。衡量有和没有混合步骤时的答案准确度。记录哪些问题类型受益最大。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式问答（Extractive QA） | 找到答案片段 | 在给定段落中预测答案的起始和结束索引。 |
| 开放域问答（Open-domain QA） | 在语料库上进行问答 | 不提供段落；必须先检索再回答。 |
| RAG | 先检索再生成 | 检索增强生成（Retrieval-Augmented Generation）。检索器 + 阅读器流水线。 |
| SQuAD | 经典基准数据集 | 斯坦福问答数据集（Stanford Question Answering Dataset）。使用 EM + F1 指标。 |
| 幻觉（Hallucination） | 编造的答案 | 阅读器输出的内容不被检索到的上下文所支持。 |
| 拒绝校准（Refusal Calibration） | 知道何时闭嘴 | 系统在无法回答时正确地说"I don't know"。 |

## 扩展阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR（Dense Passage Retrieval），问答领域经典的稠密检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG 综述。
