# 情感分析（Sentiment Analysis）

> 经典的 NLP 任务。你需要知道的关于经典文本分类的大部分内容都在这里出现。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 5 · 02（BoW + TF-IDF）、Phase 2 · 14（朴素贝叶斯）
**时间：** 约 75 分钟

## 问题

"The food was not great." 正面还是负面？

情感听起来简单。评论者说他们喜欢或不喜欢某样东西。给句子打标签。它成为经典 NLP 任务的原因是，每个看起来简单的案例都隐藏着一个困难的案例。否定翻转含义。讽刺反转含义。"Not bad at all" 尽管有两个负面编码的词，却是正面的。表情符号比周围的文本携带更多信号。领域词汇很重要（音乐评论中的 `tight` vs. 时尚评论中的 `tight`）。

情感是经典 NLP 的工作实验室。如果你理解为什么每个朴素基线都有特定的失败模式，你就理解了为什么每个更丰富的模型被发明出来。本课从零构建朴素贝叶斯基线，添加逻辑回归，并指出使生产情感成为合规级别问题的陷阱。

## 概念

经典情感分析是一个两步配方。

1. **表示。** 将文本转换为特征向量。BoW、TF-IDF 或 n-gram。
2. **分类。** 在标注样本上拟合线性模型（朴素贝叶斯、逻辑回归、SVM）。

朴素贝叶斯（Naive Bayes）是能用的最笨的模型。假设给定标签后每个特征独立。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。推理时，将概率相乘。"朴素"的独立性假设错得可笑，但结果却惊人地强。原因：对于稀疏文本特征和中等数据量，分类器更关心每个词偏向哪一边，而不是偏向多少。

逻辑回归（Logistic Regression）修复了独立性假设。它为每个特征学习一个权重，包括负权重。`not good` 作为二元组特征获得负权重。朴素贝叶斯无法为它从未标注过的二元组做到这一点。

## 构建它

### 步骤 1：一个真实的迷你数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

故意很小。实际工作使用数万个样本（IMDb、SST-2、Yelp polarity）。数学完全相同。

### 步骤 2：从零实现多项式朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加性平滑（alpha=1.0）是拉普拉斯平滑（Laplace Smoothing）。没有它，在某个类别中未见过的词概率为零，log 会爆炸。实践中 `alpha=0.01` 很常见。`alpha=1.0` 是教学默认值。

### 步骤 3：从零实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 正则化在这里很重要。文本特征是稀疏的；没有 L2，模型会记忆训练样本。从 `0.01` 开始并调参。

### 步骤 4：处理否定（失败模式）

考虑 "not good" 和 "not bad"。BoW 分类器看到 `{not, good}` 和 `{not, bad}`，并从训练中哪个出现更多来学习。二元组分类器看到 `not_good` 和 `not_bad`，并将它们作为不同的特征学习。这通常足够了。

一个在没有二元组时有效的更粗糙的修复：**否定作用域（Negation Scoping）**。将否定词后面的 token 前缀为 `NOT_`，直到下一个标点符号。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同的特征。分类器可以对它们赋予相反的权重。三行预处理，在情感基准上可测量的准确率提升。

### 步骤 5：重要的评估指标

如果类别不平衡，仅准确率会误导。真实情感语料库通常是 70-80% 正面或 70-80% 负面；一个常量多数类分类器获得 80% 准确率，但毫无价值。报告以下每一项：

- **每类精确率（Precision）和召回率（Recall）。** 每个类别一对。宏平均（Macro-average）它们以获得尊重类别平衡的单一数字。
- **宏平均 F1（Macro-F1，不平衡数据的主要指标）。** 每类 F1 分数的均值，等权。当类别不平衡时使用此指标而非准确率。
- **加权 F1（Weighted-F1，替代方案）。** 与宏平均相同但按类别频率加权。当不平衡本身具有业务含义时，与宏平均 F1 一起报告。
- **混淆矩阵（Confusion Matrix）。** 原始计数。在信任任何标量指标之前始终检查；它揭示了模型混淆了哪对类别。
- **每类错误样本。** 每个类别抽取 5 个错误预测。阅读它们。没有什么能替代阅读实际错误。

对于严重不平衡的数据（> 95-5 比例），报告 **AUROC** 和 **AUPRC** 而非准确率。AUPRC 对少数类更敏感，而这通常是你关心的（垃圾邮件、欺诈、罕见情感）。

**要避免的常见 bug。** 在不平衡数据上报告微平均 F1（Micro-F1）而非宏平均 F1，会得到一个看起来很高的数字，因为它被多数类主导。宏平均 F1 迫使你看到少数类的性能。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用它

scikit-learn 用六行代码正确完成。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

三件事要注意。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 添加二元组，使 `not_good` 成为一个特征。`sublinear_tf=True` 抑制重复词。这三个标志是 SST-2 上 75% 准确率基线和 85% 准确率基线之间的区别。

### 何时转向 Transformer

- 讽刺检测。经典模型在这里失败。句号。
- 长评论，情感在文档中间转变。
- 基于方面的情感（Aspect-based Sentiment）。"Camera was great but battery was terrible." 你需要将情感归因到方面。仅 Transformer 或结构化输出模型。
- 非英语、低资源语言。多语言 BERT 免费给你零样本基线。

如果你需要以上任何一项，跳到 Phase 7（Transformer 深入）。否则，TF-IDF 加二元组加否定处理上的朴素贝叶斯或逻辑回归是你 2026 年的生产基线。

### 可复现性陷阱（再次）

重新训练情感模型是常规操作。重新评估它们不是。论文中报告的准确率数字使用特定的划分、特定的预处理、特定的分词器。如果你将新模型与基线比较而不使用相同的流水线，你会得到误导性的增量。始终在你的流水线上重新生成基线，而不是使用论文的数字。

## 交付它

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: 为新数据集设计情感分析基线。
phase: 5
lesson: 05
---

给定数据集描述（领域、语言、大小、标签粒度、延迟预算），你输出：

1. 特征提取配方。指定分词器、n-gram 范围、停用词策略（通常保留）、否定处理（作用域前缀或二元组）。
2. 分类器。朴素贝叶斯用于基线，逻辑回归用于生产，仅当领域需要讽刺 / 方面 / 跨语言时才用 Transformer。
3. 评估计划。报告精确率、召回率、F1、混淆矩阵和每类错误样本（不仅仅是标量）。
4. 部署后要监控的一个失败模式。领域漂移和讽刺是前两名。

拒绝为情感任务推荐丢弃停用词。当类别不平衡时（例如 90% 正面），拒绝仅报告准确率作为唯一指标。标记子词丰富的语言需要 FastText 或 Transformer 嵌入而非词级 TF-IDF。
```

## 练习

1. **简单。** 在 scikit-learn 流水线中添加 `apply_negation` 作为预处理步骤，并在小型情感数据集上测量 F1 增量。
2. **中等。** 实现类别加权逻辑回归（向 scikit-learn 传递 `class_weight="balanced"`，或自己推导梯度）。在合成的 90-10 类别不平衡上测量效果。
3. **困难。** 通过在情感模型的残差上训练第二个分类器来构建讽刺检测器。记录你的实验设置。当你的准确率低于随机水平时警告读者（2 类讽刺的随机水平约为 50%，大多数首次尝试都落在这里）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 极性（Polarity） | 正面或负面 | 二元标签；有时扩展到中性或细粒度（5 星）。 |
| 基于方面的情感（Aspect-based Sentiment） | 每个方面的极性 | 将情感归因到文本中提到的特定实体或属性。 |
| 否定作用域（Negation Scoping） | 反转附近的 token | 将 "not" 后面的 token 前缀为 `NOT_`，直到标点符号。 |
| 拉普拉斯平滑（Laplace Smoothing） | 给计数加 1 | 防止朴素贝叶斯中的零概率特征。 |
| L2 正则化（L2 Regularization） | 收缩权重 | 向损失添加 `lambda * sum(w^2)`。对稀疏文本特征至关重要。 |

## 进一步阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 基础综述。很长，但前四节涵盖了所有经典内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 展示了二元组 + 朴素贝叶斯在短文本上难以击败的论文。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 以及你将调整的每个旋钮的参考。
