# 嵌入模型（Embedding Models）—— 2026 深度解析

> Word2Vec 为每个词给出一个向量。现代嵌入模型为每个段落给出一个向量，支持跨语言，提供稀疏（sparse）、稠密（dense）和多向量（multi-vector）视图，尺寸可适配你的索引。选错了，你的 RAG 就会检索到错误的内容。

**类型（Type）：** Learn
**语言（Languages）：** Python
**前置要求（Prerequisites）：** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**时间（Time）：** ~60 分钟

## 问题所在（The Problem）

你的 RAG 系统有 40% 的概率检索到错误的段落。罪魁祸首很少是向量数据库或提示词，而是嵌入模型。

在 2026 年选择嵌入模型，需要在五个维度上做出权衡：

1. **稠密 vs 稀疏 vs 多向量。** 每个段落一个向量，还是每个词元（token）一个向量，抑或是一个稀疏加权的词袋（bag of words）。
2. **语言覆盖范围。** 单语英语模型在纯英语任务上仍然胜出。多语言模型在语料混合时胜出。
3. **上下文长度（Context length）。** 512 个词元 vs 8,192 vs 32,768 —— 而实际有效容量通常只有标称最大值的 60-70%。
4. **维度预算。** 全精度下 3,072 个浮点数 = 每个向量 12 KB。在 1 亿个向量的规模下，存储成本为 $1,300/月。套娃（Matryoshka）截断可将此削减 4 倍。
5. **开源 vs 托管。** 开源权重意味着你掌控技术栈和数据。托管意味着你用控制权换取始终最新的模型。

本课将逐一阐明这些权衡，让你能基于证据做出选择，而不是跟风上一季度的热门模型。

## 核心概念（The Concept）

![稠密、稀疏和多向量嵌入](../assets/embedding-modes.svg)

**稠密嵌入（Dense embeddings）。** 每个段落一个向量（通常 384-3,072 维）。余弦相似度（cosine similarity）按语义接近程度对段落进行排序。OpenAI `text-embedding-3-large`、BGE-M3 稠密模式、Voyage-3。默认选择。

**稀疏嵌入（Sparse embeddings）。** SPLADE 风格。一个 Transformer 为每个词汇表词元预测一个权重，然后将其中大部分置零。结果是一个大小为 |vocab| 的稀疏向量。能捕捉词汇匹配（类似 BM25），但带有学习到的词项权重。在关键词密集的查询上表现强劲。

**多向量（晚期交互，late interaction）。** ColBERTv2、Jina-ColBERT。每个词元一个向量。使用 MaxSim 进行评分：对每个查询词元，找到最相似的文档词元，将分数求和。存储和评分成本更高，但在长查询和领域特定语料上胜出。

**BGE-M3：三者合一。** 单个模型同时输出稠密、稀疏和多向量表示。每种表示可以独立查询；分数通过加权求和进行融合。当你想从一个检查点（checkpoint）获得灵活性时，这是 2026 年的默认选择。

**套娃表示学习（Matryoshka Representation Learning）。** 训练方式使得向量的前 N 维本身就构成一个可用的独立嵌入。将一个 1,536 维的向量截断到 256 维，以约 1% 的精度损失换取 6 倍的存储节省。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 均支持此特性。

### MTEB 排行榜只讲了故事的一部分

大规模文本嵌入基准（Massive Text Embedding Benchmark）—— 发布时（2022 年）涵盖 8 种任务类型共 56 个任务，MTEB v2 扩展到 100+ 个任务。2026 年初，Gemini Embedding 2 在检索任务上位居榜首（67.71 MTEB-R）。Cohere embed-v4 在通用任务上领先（65.2 MTEB）。BGE-M3 在开源多语言模型上领先（63.0）。排行榜是必要的，但并不充分 —— 务必在你的领域上进行基准测试。

### 三层模式（The three-tier pattern）

| 使用场景 | 模式 |
|----------|---------|
| 快速初筛（Fast first-pass） | 稠密双编码器（bi-encoder）（BGE-M3, text-3-small） |
| 召回率提升（Recall boost） | 稀疏（SPLADE, BGE-M3 sparse）+ RRF 融合 |
| Top-50 精确排序（Precision on top-50） | 多向量（ColBERTv2）或交叉编码器重排序器（cross-encoder reranker） |

大多数生产环境的技术栈三者都用。

## 动手实践（Build It）

### 步骤 1：基线 —— 使用 Sentence-BERT 的稠密嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使点积等于余弦相似度。务必始终设置它。

### 步骤 2：套娃截断（Matryoshka truncation）

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后需要重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 经过训练，使得前几个截断级别几乎无损。非套娃模型（原始 Sentence-BERT）在截断时性能会急剧下降。

### 步骤 3：BGE-M3 多功能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引，一次推理调用。分数融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调优这些权重。

### 步骤 4：在自定义任务上进行 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在一个*有代表性*的子集上运行你的候选模型。不要只相信排行榜排名 —— 你的领域才是关键。

### 步骤 5：从零手写余弦相似度

参见 `code/main.py`。平均哈希技巧嵌入（Averaged Hashing Trick embeddings，仅使用标准库）。虽然无法与 Transformer 嵌入竞争，但展示了基本形态：分词（tokenize）→ 向量化（vector）→ 归一化（normalize）→ 点积（dot product）。

## 常见陷阱（Pitfalls）

- **查询和文档使用相同的模型。** 某些模型（Voyage、Jina-ColBERT）使用非对称编码（asymmetric encoding）—— 查询和文档经过不同的路径。务必查看模型卡片（model card）。
- **缺少前缀（prefix）。** `bge-*` 模型需要在查询前添加 `"Represent this sentence for searching relevant passages: "`。如果忘记，召回率会下降 3-5 个百分点。
- **过度截断套娃维度。** 1,536 → 256 通常是安全的。1,536 → 64 则不安全。在你的评估集上验证。
- **上下文截断。** 大多数模型会静默截断超过其最大长度的输入。长文档需要分块（chunking）（参见第 23 课）。
- **忽略延迟尾部。** MTEB 分数隐藏了 p99 延迟。一个 600M 参数的模型可能比 335M 参数的模型高出 2 分，但每次查询的成本是 3 倍。

## 使用指南（Use It）

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 仅英语、快速、API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重、英语 | `BAAI/bge-large-en-v1.5` |
| 开源权重、多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（137M 参数，MoE） |
| 存储受限 | 套娃截断 + int8 量化 |
| 关键词密集的查询 | 添加 SPLADE 稀疏，与稠密进行 RRF 融合 |

2026 年模式：从 BGE-M3 或 text-3-large 开始，用 MTEB 在你的领域上评估，如果某个领域特定模型领先超过 3 分就切换。

## 交付成果（Ship It）

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习（Exercises）

1. **简单。** 用 `bge-small-en-v1.5` 对 100 个句子进行编码，先使用完整维度（384），再使用套娃 128 维。测量 10 个查询上的 MRR 下降幅度。
2. **中等。** 在你所在领域的 500 个段落上比较 BGE-M3 的稠密、稀疏和 ColBERT 模式。哪种模式在 recall@10 上胜出？RRF 融合是否优于最佳单一模式？
3. **困难。** 在你排名前 2 的领域任务上对三个候选模型运行 MTEB。报告 MTEB 分数、100 条查询批次的 p99 延迟以及每百万查询的成本（$/1M queries）。选出帕累托最优（Pareto-optimal）的那个。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 稠密嵌入（Dense embedding） | 那个向量 | 每个文本一个固定大小的向量。用余弦相似度排序。 |
| 稀疏嵌入（Sparse embedding） | 学习版的 BM25 | 每个词汇表词元一个权重；大部分为零；端到端训练。 |
| 多向量（Multi-vector） | ColBERT 风格 | 每个词元一个向量；MaxSim 评分；索引更大，召回更好。 |
| 套娃（Matryoshka） | 俄罗斯套娃技巧 | 前 N 维本身就是一个有效的更小嵌入。 |
| MTEB | 那个基准 | 大规模文本嵌入基准 —— 发布时 56 个任务，v2 中 100+ 个。 |
| BEIR | 那个检索基准 | 18 个零样本检索任务；常被引用于跨领域鲁棒性评估。 |
| 非对称编码（Asymmetric encoding） | 查询路径 ≠ 文档路径 | 模型对查询和文档使用不同的投影。 |

## 延伸阅读（Further Reading）

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — 双编码器论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 统一三模式模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 生产环境中的晚期交互。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — 实时排名。
