# 词袋模型、TF-IDF 与文本表示

> 先计数，后思考。2026 年，TF-IDF 在定义明确的任务上仍然击败嵌入。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 5 · 01（文本处理）、Phase 2 · 02（从零实现线性回归）
**时间：** 约 75 分钟

## 问题

模型需要数字。你只有字符串。

每个 NLP 流水线都必须回答同一个问题。我们如何将变长的 token 流转换为分类器可以消费的固定大小向量。该领域找到的第一个答案是能用的最笨的那个。数词。做一个向量。

这个向量承载了比任何嵌入模型都多的生产 NLP。垃圾邮件过滤器、主题分类器、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、第一个十年的学术 NLP 基准。2026 年的实践者在狭窄的分类任务上仍然首先使用它。它快速、可解释，并且在词出现就是关键的任务上，通常与 4 亿参数的嵌入模型无法区分。

本课从零构建词袋模型，然后是 TF-IDF。然后展示 scikit-learn 用三行代码做同样的事。然后指出让你转向嵌入的失败模式。

## 概念

**词袋模型（Bag of Words, BoW）** 丢弃顺序。对于每个文档，计算每个词汇词出现的次数。向量长度是词汇量大小。位置 `i` 是词 `i` 的计数。

**TF-IDF** 重新加权 BoW。出现在每个文档中的词没有信息量，所以降低它的权重。在整个语料库中罕见但在单个文档中频繁出现的词是信号，所以提高它的权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档中的词频（Term Frequency），`df` 是文档频率（Document Frequency，包含该词的文档数），`N` 是文档总数。`log` 使权重对无处不在的词保持有界。

关键特性：两者都产生具有可解释轴的稀疏向量。你可以查看训练好的分类器的权重，读出哪些词将文档推向每个类别。你无法用 768 维的 BERT 嵌入做到这一点。

## 构建它

### 步骤 1：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器都可以；本课的 `code/main.py` 使用简化的全小写变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着词索引 0 是第一个文档中出现的第一个词。惯例各不相同；scikit-learn 按字母顺序排序。

### 步骤 2：词袋模型

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档。列是词汇索引。条目 `[i][j]` 是"词 `j` 在文档 `i` 中出现了多少次"。文档 1 有两次 `cat`，因为它确实有。文档 0 有零次 `ran`，因为它没有。

### 步骤 3：词频和文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得指出的平滑技巧。`(n+1)/(d+1)` 避免了 `log(x/0)`。末尾的 `+1` 确保出现在每个文档中的词仍然有 IDF 为 1（而不是 0），与 scikit-learn 的默认值一致。其他实现使用原始的 `log(N/df)`。两者都有效；平滑版本更友好。

### 步骤 4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词汇词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在所有三个文档中，所以它的 IDF 很低。`dog` 出现在一个文档中，所以它的 IDF 很高。向量是稀疏的（大多数条目很小），区分性词突出。

### 步骤 5：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

没有归一化，较长的文档会得到更大的向量并主导相似度分数。L2 归一化将每个文档放在单位超球面上。行之间的余弦相似度现在就是点积。

## 使用它

scikit-learn 提供了生产版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一次调用中完成分词、词汇表构建和 BoW。`TfidfVectorizer` 添加 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。对于 10 万个文档，密集版本放不进内存；在分类器要求密集之前保持稀疏。

改变一切的旋钮：

| 参数 | 效果 |
|------|------|
| `ngram_range=(1, 2)` | 包含二元组（bigram）。通常提升分类效果。 |
| `min_df=2` | 丢弃出现在少于 2 个文档中的词。在噪声数据上修剪词汇表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。近似停用词移除，无需硬编码列表。 |
| `stop_words="english"` | scikit-learn 的内置停用词列表。取决于任务——情感分析*不应*丢弃否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 而不是原始 `tf`。当一个词在一个文档中重复多次时有帮助。 |

### TF-IDF 何时仍然胜出（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。词出现是关键；语义细微差别不重要。
- 低数据场景（数百个标注样本）。TF-IDF 加逻辑回归没有预训练成本。
- 任何延迟重要的地方。TF-IDF 加线性模型在微秒内回答。通过 Transformer 嵌入一个文档需要 10-100 毫秒。
- 必须解释其预测的系统。检查分类器的系数。权重最高的正词就是原因。

### TF-IDF 何时失败

语义盲区失败。考虑这两个文档：

- "The movie was not good at all."
- "The movie was excellent."

一个是否定评论。一个是正面评论。它们的 TF-IDF 重叠恰好是 `{the, movie, was}`。词袋分类器必须记住 `not` 靠近 `good` 会翻转标签。它可以在足够的数据上学会这一点，但永远不会像理解句法的模型那样优雅。

另一个失败：推理时的词汇外（Out-of-Vocabulary, OOV）词。在 IMDb 评论上训练的 BoW 模型不知道如何处理 `Zoomer-approved`，如果该 token 从未在训练中出现过。子词嵌入（第 04 课）处理这个问题。TF-IDF 不能。

### 混合方案：TF-IDF 加权嵌入

2026 年中等数据分类的实用默认方案：使用 TF-IDF 权重作为词嵌入上的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从嵌入中获得语义能力，从 TF-IDF 中获得罕见词强调。分类器在池化后的向量上训练。在低于约 5 万个标注样本的情感、主题和意图分类中，这优于单独使用任何一种。

## 交付它

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: 给定文本分类任务，推荐 BoW、TF-IDF、嵌入或混合方案。
phase: 5
lesson: 02
---

你推荐文本向量化策略。给定任务描述，输出：

1. 表示方式（BoW、TF-IDF、Transformer 嵌入或混合方案）。用一句话解释原因。
2. 具体的向量化器配置。命名库。引用参数（`ngram_range`、`min_df`、`max_df`、`sublinear_tf`、`stop_words`）。
3. 发布前要测试的一个失败模式。

当用户标注样本少于 500 个时，拒绝推荐嵌入，除非他们展示了 TF-IDF 基线的语义失败证据。拒绝为情感分析移除停用词（否定词携带信号）。标记类别不平衡需要的不只是向量化器更改。

示例输入："将 3 万张客户支持工单分类到 12 个类别。大多数工单是 2-3 句话。仅英文。需要可解释性用于审计日志。"

示例输出：

- 表示方式：TF-IDF。3 万个样本不算少；可解释性要求排除了密集嵌入。
- 配置：`TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`。保留停用词，因为类别关键词有时就是停用词（"not working" vs "working"）。
- 要测试的失败：验证 `min_df=3` 不会丢弃罕见的类别关键词。按类别过滤运行 `get_feature_names_out` 并目视检查。
```

## 练习

1. **简单。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，词汇不相交的文档得分为 0.0。
2. **中等。** 为 `bag_of_words` 添加 `n-gram` 支持。参数 `n` 产生 `n`-gram 的计数。测试 `n=2` 在 `["the", "cat", "sat"]` 上产生二元组计数 `["the cat", "cat sat"]`。
3. **困难。** 使用 GloVe 100d 向量（下载一次，缓存）构建上述 TF-IDF 加权嵌入混合方案。在 20 Newsgroups 数据集上比较分类准确率，对比纯 TF-IDF 和纯均值池化嵌入。报告哪种在何处胜出。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| BoW | 词频向量 | 一个文档中词汇词的计数。丢弃顺序。 |
| TF | 词频（Term Frequency） | 一个词在文档中的计数，可选地按文档长度归一化。 |
| DF | 文档频率（Document Frequency） | 至少包含该词一次的文档数。 |
| IDF | 逆文档频率（Inverse Document Frequency） | 平滑后的 `log(N / df)`。降低无处不在的词的权重。 |
| 稀疏向量（Sparse Vector） | 大部分为零 | 词汇表通常为 1 万到 10 万个词；大多数词在任何给定文档中都不存在。 |
| 余弦相似度（Cosine Similarity） | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 进一步阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 权威 API 参考，以及每个旋钮的注释。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 使 TF-IDF 成为十年默认的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年关于旧方法何时胜出以及为什么的观点。
