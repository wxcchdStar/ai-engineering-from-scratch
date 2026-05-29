# 主题建模（Topic Modeling）—— LDA 与 BERTopic

> LDA：文档是主题的混合，主题是词的概率分布。BERTopic：文档在嵌入空间中聚类，簇即为主题。目标相同，底层原语不同。

**类型（Type）：** Learn
**语言（Languages）：** Python
**前置条件（Prerequisites）：** Phase 5 · 02（词袋模型与 TF-IDF）、Phase 5 · 03（Word2Vec）
**时间（Time）：** 约 45 分钟

## 问题描述（The Problem）

你手头有 10,000 条客户支持工单、50,000 篇新闻文章或 200,000 条推文。你需要在不逐篇阅读的情况下了解整个语料库的内容。你没有标注好的类别，甚至不知道有多少个类别存在。

主题建模（Topic Modeling）可以在无监督的情况下回答这个问题。给它一个语料库，它会返回一小组连贯的主题，以及每篇文档在这些主题上的分布。

两大算法家族占据主导地位。LDA（2003 年）将每篇文档视为潜在主题（Latent Topic）的混合，将每个主题视为词的概率分布。其推断是贝叶斯式的。当你需要混合成员（Mixed Membership）主题分配和可解释的词级概率分布时，它仍然在生产环境中服役。

BERTopic（2020 年）使用 BERT 对文档进行编码，用 UMAP 降维，用 HDBSCAN 聚类，并通过基于类别的 TF-IDF（Class-based TF-IDF）提取主题词。它在短文本、社交媒体以及任何语义相似性比词重叠更重要的场景中表现优异。一篇文档只分配一个主题，这对长文本内容来说是一个局限。

本课将建立对这两种方法的直觉，并指出针对给定语料库应选择哪一种。

## 核心概念（The Concept）

![LDA 混合模型 vs BERTopic 聚类](../assets/topic-modeling.svg)

**LDA 的生成故事。** 每个主题是词的一个概率分布。每篇文档是主题的一个混合。要生成文档中的一个词，先从文档的主题混合中采样一个主题，再从该主题的词分布中采样一个词。推断过程则反过来：给定观测到的词，推断每篇文档的主题分布和每个主题的词分布。数学上使用折叠吉布斯采样（Collapsed Gibbs Sampling）或变分贝叶斯（Variational Bayes）来完成。

LDA 的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行之和为 1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行之和为 1（主题的词分布）。

**BERTopic 流水线。**

1. 使用句子变换器（Sentence Transformer，例如 `all-MiniLM-L6-v2`）对每篇文档进行编码。得到 384 维向量。
2. 使用 UMAP 将维度降至约 5 维。BERT 嵌入维度太高，不适合直接聚类。
3. 使用 HDBSCAN 进行聚类。基于密度，产生大小可变的簇和一个"离群点"标签。
4. 对每个簇，计算该簇文档上的基于类别的 TF-IDF，以提取 top 词。

输出是每篇文档一个主题（外加一个 -1 离群点标签）。可选地，可以通过 HDBSCAN 的概率向量获得软成员关系。

## 动手构建（Build It）

### 步骤 1：通过 scikit-learn 使用 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：去除了停用词，`min_df` 和 `max_df` 过滤了稀有词和过于普遍的词，使用 `CountVectorizer`（而非 `TfidfVectorizer`），因为 LDA 期望原始计数。

### 步骤 2：BERTopic（生产环境）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 的过滤条件去掉了 BERTopic 的离群点桶（HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小簇大小；BERTopic 库的默认值是 10。本示例为适应该课程的规模将其显式设置为 15。对于超过 10,000 篇文档的语料库，建议增加到 50 或 100。

### 步骤 3：评估

两种方法都输出主题词。问题在于这些词是否具有连贯性。

- **主题连贯性 c_v（Topic Coherence c_v）。** 结合滑动窗口上下文中 top 词对的 NPMI（归一化点互信息，Normalized Pointwise Mutual Information），将分数聚合为主题向量，并通过余弦相似度比较这些向量。越高越好。使用 `gensim.models.CoherenceModel`，设置 `coherence="c_v"`。
- **主题多样性（Topic Diversity）。** 所有主题 top 词中唯一词的比例。越高越好（主题之间不重叠）。
- **定性检查。** 阅读每个主题的 top 词。它们是否命名了一个真实的事物？人工判断仍然是最后一道防线。

## 何时选择哪种方法（When to pick which）

| 场景 | 选择 |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 具有主题混合的长文档 | LDA |
| 无 GPU / 计算资源有限 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| 结合 LLM 进行主题标注 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 最大化语义连贯性 | BERTopic |

最大的实际考量因素是文档长度。BERT 嵌入会截断；LDA 的计数方法适用于任意长度。对于超过嵌入模型上下文窗口的文档，要么分块后聚合，要么使用 LDA。

## 实际使用（Use It）

2026 年的技术栈：

- **BERTopic。** 短文本和任何语义重要的场景的默认选择。
- **`gensim.models.LdaModel`。** 经典的生产环境 LDA，成熟且久经考验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 用于实验的简易 LDA。
- **NMF。** 非负矩阵分解（Non-negative Matrix Factorization）。LDA 的快速替代方案，在短文本上质量相当。
- **Top2Vec。** 设计思路与 BERTopic 类似。社区较小，但在某些基准测试上表现良好。
- **FASTopic。** 较新的方法，在超大规模语料库上比 BERTopic 更快。
- **基于 LLM 的标注。** 运行任意聚类，然后提示模型为每个簇命名。

## 交付物（Ship It）

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习（Exercises）

1. **简单。** 在 20 Newsgroups 数据集上用 5 个主题拟合 LDA。打印每个主题的 top 10 词。手动为每个主题标注。算法是否找到了真实的类别？
2. **中等。** 在相同的 20 Newsgroups 子集上拟合 BERTopic。比较找到的主题数量、top 词和定性连贯性与 LDA 的差异。哪种方法更清晰地呈现了真实的类别？
3. **困难。** 在你的语料库上计算 LDA 和 BERTopic 的 c_v 连贯性。分别用 5、10、20、50 个主题运行。绘制连贯性 vs 主题数量的图表。报告哪种方法在不同主题数量下更稳定。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 主题（Topic） | 语料库所涉及的事物 | 词的概率分布（LDA）或相似文档的簇（BERTopic）。 |
| 混合成员（Mixed Membership） | 文档属于多个主题 | LDA 为每篇文档分配一个覆盖所有主题的分布。 |
| UMAP | 降维 | 保留局部结构的流形学习（Manifold Learning）；用于 BERTopic。 |
| HDBSCAN | 密度聚类 | 找到大小可变的簇；为离群点生成"噪声"标签（-1）。 |
| c_v 连贯性（c_v Coherence） | 主题质量指标 | 滑动窗口内 top 主题词的平均点互信息。 |

## 延伸阅读（Further Reading）

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 及其同类指标的论文。
- [BERTopic 文档](https://maartengr.github.io/BERTopic/) — 生产环境参考。包含优秀的示例。
