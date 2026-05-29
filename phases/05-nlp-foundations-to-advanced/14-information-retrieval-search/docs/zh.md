# 信息检索与搜索（Information Retrieval and Search）

> BM25 精确但脆弱。稠密检索（Dense）覆盖面广但会遗漏关键词。混合检索（Hybrid）是 2026 年的默认方案。其余的都是调优。

**类型（Type）：** Build
**语言（Languages）：** Python
**前置条件（Prerequisites）：** Phase 5 · 02（词袋模型（BoW）+ TF-IDF），Phase 5 · 04（GloVe、FastText、子词（Subword））
**时间（Time）：** ~75 分钟

## 问题（The Problem）

用户输入"如果有人撒谎骗钱会怎样"，期望找到真正涵盖该情形的法条："印度刑法典第 420 条（Section 420 IPC）"。关键词搜索完全找不到（没有共享词汇）。如果嵌入（Embeddings）没有在法律文本上训练过，语义搜索（Semantic Search）也会漏掉。真正的搜索必须同时处理这两种情况。

信息检索（Information Retrieval，IR）是每个 RAG 系统、每个搜索栏、每个文档站点的模糊查找背后的流水线。2026 年在生产环境中行之有效的架构不是单一方法，而是一系列互补方法的链条，每种方法都能捕获前一种方法的失败案例。

本课将构建每个组件，并指出每种方法能捕获哪些失败。

## 概念（The Concept）

![混合检索：BM25 + 稠密检索 + RRF + 交叉编码器重排序](../assets/retrieval.svg)

四个层次。按需选择。

1. **稀疏检索（Sparse retrieval，BM25）。** 速度快，精确匹配准确，语义理解差。基于倒排索引（Inverted Index）运行。在数百万文档上每次查询不到 10 毫秒。能正确处理法规引用、产品代码、错误消息、命名实体。
2. **稠密检索（Dense retrieval）。** 将查询和文档编码为向量。最近邻搜索。能捕获释义和语义相似性。但会遗漏仅差一个字符的精确关键词匹配。使用 FAISS 或向量数据库（Vector DB）时每次查询 50-200 毫秒。
3. **融合（Fusion）。** 合并稀疏检索和稠密检索的排序列表。倒数排名融合（Reciprocal Rank Fusion，RRF）是简单的默认方案，因为它忽略原始分数（这些分数处于不同的量级），只使用排名位置。当你知道某个信号在你的领域中占主导地位时，加权融合（Weighted Fusion）是一个可选方案。
4. **交叉编码器重排序（Cross-encoder rerank）。** 从融合结果中取前 30 条。运行交叉编码器（Cross-encoder）（查询 + 文档一起输入，对每对进行评分）。保留前 5 条。交叉编码器每对的推理速度比双编码器（Bi-encoder）慢，但准确度远高于后者。通过仅对前 30 条运行来分摊成本。

在 2026 年的基准测试中，三路检索（Three-way retrieval，BM25 + 稠密检索 + 学习型稀疏检索如 SPLADE）优于两路检索，但需要支持学习型稀疏索引（Learned-sparse Indexes）的基础设施。对大多数团队而言，两路检索加交叉编码器重排序是最佳平衡点。

## 构建（Build It）

### 步骤 1：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

有两个值得了解的参数。`k1=1.5` 控制词频饱和（Term-frequency Saturation）；值越高，词重复的权重越大。`b=0.75` 控制长度归一化（Length Normalization）；0 表示忽略文档长度，1 表示完全归一化。这些默认值是 Robertson 在原论文中的推荐值，很少需要调整。

### 步骤 2：使用双编码器进行稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入进行 L2 归一化，使点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维的，速度快，对大多数英文检索来说足够强大。对于多语言任务，使用 `paraphrase-multilingual-MiniLM-L12-v2`。对于最高准确度，使用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 步骤 3：倒数排名融合（Reciprocal Rank Fusion）

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 这个常量来自原始的 RRF 论文。`k` 值越大，排名差异的贡献越平坦；`k` 值越小，顶部排名越占主导。60 是已发表的默认值，很少需要调整。

### 步骤 4：混合搜索（Hybrid Search）+ 重排序

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合在一起。BM25 找到词汇匹配。稠密检索找到语义匹配。RRF 合并两个排序结果，无需分数校准。交叉编码器使用查询-文档对一起对前 30 条重新评分，捕获双编码器遗漏的细粒度相关性。保留前 5 条。

### 步骤 5：评估（Evaluation）

| 指标（Metric） | 含义（Meaning） |
|--------|---------|
| Recall@k | 在正确答案存在的查询中，正确答案出现在前 k 条中的频率是多少？ |
| MRR（平均倒数排名，Mean Reciprocal Rank） | 第一个相关文档排名的倒数的平均值。 |
| nDCG@k | 考虑相关性等级，而不仅仅是二元的"相关/不相关"。 |

对于 RAG 而言，检索器的 **Recall@k** 是最重要的指标。如果正确的段落不在检索结果集中，你的阅读器（Reader）就无法回答。

调试技巧：对于失败的查询，对比稀疏检索和稠密检索的排序结果。如果其中一个找到了正确的文档而另一个没有，那么你面临的是词汇不匹配（Vocabulary Mismatch）（修复方法：补充缺失的那一半）或语义歧义（Semantic Ambiguity）（修复方法：更好的嵌入或重排序器）。

## 使用（Use It）

2026 年的技术栈：

| 规模（Scale） | 技术栈（Stack） |
|-------|-------|
| 1k-100k 文档 | 内存 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。无需独立数据库。 |
| 100k-10M 文档 | FAISS 或 pgvector 用于稠密检索 + Elasticsearch / OpenSearch 用于 BM25。并行运行。 |
| 10M+ 文档 | Qdrant / Weaviate / Vespa / Milvus，支持混合检索。对前 30 条进行交叉编码器重排序。 |
| 最佳质量前沿 | 三路检索（BM25 + 稠密检索 + SPLADE）+ ColBERT 后期交互重排序（Late-interaction Reranking） |

无论你选择什么，都要为评估留出预算。在评估端到端 RAG 准确度之前，先对检索召回率（Retrieval Recall）进行基准测试。阅读器无法修复检索器遗漏的内容。

### 2026 年生产环境 RAG 的宝贵经验

- **80% 的 RAG 失败可追溯到数据摄入（Ingestion）和分块（Chunking），而非模型。** 团队花费数周时间更换 LLM 和调整提示词，而检索器却默默地在每三次查询中就返回一次错误的上下文。先修复分块。
- **分块策略（Chunking Strategy）比分块大小更重要。** 固定大小的切分会破坏表格、代码和嵌套标题。句子感知（Sentence-aware）分块是默认方案；对于技术文档和产品手册，语义分块（Semantic Chunking）或基于 LLM 的分块（LLM-based Chunking）效果更好。
- **父文档模式（Parent-doc Pattern）。** 检索小的"子"块以提高精度。当来自同一父章节的多个子块出现时，替换为父块以保留上下文。这能持续提升答案质量，无需重新训练。
- **k_rerank=3 通常是最优的。** 超过这个数量的每个额外块都会增加 token 成本和生成延迟，而不会提升答案质量。如果 k=8 对你来说仍然比 k=3 更好，说明重排序器表现不佳。
- **HyDE / 查询扩展（Query Expansion）。** 从查询生成一个假设答案，将其嵌入，然后检索。弥合短问题与长文档之间的措辞差距。无需训练即可免费提升精度。
- **上下文预算控制在 8K token 以下。** 如果在该限制下持续命中，说明重排序器阈值太宽松。
- **对所有内容进行版本管理。** 提示词、分块规则、嵌入模型、重排序器。任何漂移都会悄无声息地破坏答案质量。在忠实度（Faithfulness）、上下文精度（Context Precision）和未回答问题率（Unanswered-question Rate）上设置 CI 门禁，在用户看到之前阻止回归。
- **三路检索（BM25 + 稠密检索 + 学习型稀疏检索如 SPLADE）优于两路检索**，在 2026 年的基准测试中，尤其是对于混合专有名词和语义的查询。当基础设施支持 SPLADE 索引时就部署它。

根据 2026 年的行业测量，合理的检索设计可将幻觉（Hallucinations）减少 70-90%。大多数 RAG 性能提升来自更好的检索，而非模型微调。

## 交付（Ship It）

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习（Exercises）

1. **简单。** 在 500 篇文档的语料库上实现上述 `hybrid_search`。测试 20 个查询。比较 BM25-only、dense-only 和 hybrid 在 Recall@5 上的表现。
2. **中等。** 添加 MRR 计算。对于每个已知正确答案的测试查询，找出正确答案在 BM25、稠密检索和混合检索排序中的排名。报告每种方法的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域上微调稠密编码器。从 500 个查询-文档对构建训练集。比较微调前后的召回率。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 实际含义（What it actually means） |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索（Keyword search） | Okapi BM25。通过词频、IDF 和长度对文档进行评分。 |
| 稠密检索（Dense retrieval） | 向量搜索（Vector search） | 将查询和文档编码为向量，找到最近邻。 |
| 双编码器（Bi-encoder） | 嵌入模型（Embedding model） | 独立编码查询和文档。查询时速度快。 |
| 交叉编码器（Cross-encoder） | 重排序模型（Reranker model） | 将查询和文档一起编码。速度慢但准确。 |
| RRF | 排名融合（Rank fusion） | 通过求和 `1/(k + rank)` 来合并两个排序。 |
| Recall@k | 检索指标（Retrieval metric） | 相关文档出现在前 k 条中的查询比例。 |

## 延伸阅读（Further Reading）

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的权威论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，经典的双编码器方案。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 学习型稀疏检索器，缩小了与稠密检索的差距。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 后期交互检索（Late-interaction Retrieval）。
