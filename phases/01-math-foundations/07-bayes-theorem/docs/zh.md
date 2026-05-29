# 贝叶斯定理 (Bayes Theorem)

> 贝叶斯定理是将你已有信念与新证据相结合的唯一正确方法。

**类型：** 学习 (Learn)
**语言：** Python
**前置要求：** 第一阶段，第06课（概率与分布 (Probability & Distributions)）
**时间：** 约60分钟

## 学习目标

- 从条件概率的定义推导贝叶斯定理并理解先验、似然和后验 (prior, likelihood, posterior) 的含义
- 使用链式法则分解联合概率，并解释其与自回归语言模型 (autoregressive language model) 的关系
- 手动计算一个医疗检测场景，理解为什么即使检测准确率达到99%，罕见病检测的假阳性 (false positive) 率依然很高
- 在垃圾邮件过滤器中实现朴素贝叶斯分类器 (Naive Bayes classifier) 并计算精确率与召回率

## 问题

你醒来感觉不太舒服。做了一个检测。医生说：“这个检测对患者准确率达到99%。”你检测结果呈阳性。你患病的概率是多少？

如果你回答99%，你犯了和大多数医生（以及大多数机器学习工程师）相同的错误。正确答案取决于疾病有多罕见。

如果每1000人中只有1人患病（患病率0.1%），那么给定阳性检测结果后你真正患病的概率约为9%，而不是99%。

贝叶斯定理给出了正确的计算方法。在机器学习中，它无处不在：朴素贝叶斯分类器、贝叶斯神经网络 (Bayesian neural network)、隐马尔可夫模型 (Hidden Markov Model)、卡尔曼滤波器 (Kalman Filter)、变分推断 (Variational Inference) —— 所有这些都依赖于贝叶斯定理。

## 概念

### 条件概率

条件概率 P(A|B) 读作“给定B后A的概率”。它意味着：宇宙已经缩小到只包含B发生的情况。在这个缩小后的宇宙中，A也发生的比例是多少？

```
P(A|B) = P(A ∩ B) / P(B)
```

条件概率不是对称的。P(下雨 | 多云) 很低（大多数阴天不下雨）。P(多云 | 下雨) 很高（下雨时通常多云）。

### 贝叶斯定理

贝叶斯定理翻转条件概率。给定P(A|B)，你可以计算P(B|A)：

```
P(A|B) = P(B|A) * P(A) / P(B)

其中：
  P(A)   = 先验 (prior)：在看到任何证据之前A的概率
  P(A|B) = 后验 (posterior)：在看到证据B之后A的概率
  P(B|A) = 似然 (likelihood)：如果A为真，看到证据B的概率
  P(B)   = 证据 (evidence)：看到B的总体概率（归一化常数）
```

另一种写法：

```
P(A|B) = P(B|A) * P(A) / (P(B|A) * P(A) + P(B|¬A) * P(¬A))
```

这只是一个等式。但它编码了学习的整个过程。你从一个先验信念开始（我在看到数据之前的想法）。你观察到一些证据（数据）。你将这一证据与你如果假设为真或为假时所见数据一致的程度进行比较。你更新你的信念。后验成为你新的先验。重复这个过程。这就是科学方法。这就是机器学习。

### 从条件概率定义推导

你可以用高中数学从条件概率定义推出贝叶斯定理：

```
P(A|B) = P(A ∩ B) / P(B)    →  P(A ∩ B) = P(A|B) * P(B)
P(B|A) = P(A ∩ B) / P(A)    →  P(A ∩ B) = P(B|A) * P(A)

因此：P(A|B) * P(B) = P(B|A) * P(A)

所以：P(A|B) = P(B|A) * P(A) / P(B)
```

P(B) 通常难以计算，但可以通过全概率公式 (law of total probability) 得到：

```
P(B) = P(B|A) * P(A) + P(B|¬A) * P(¬A)
```

### 医学检测示例

最经典的例子：

```
患病率 (prevalence)：          P(D) = 0.001  （每1000人1人）
检测准确率 (sensitivity)：     P(+|D) = 0.99  （患者中99%检测呈阳性）
假阳性率 (false positive)：    P(+|¬D) = 0.01  （健康人中1%检测呈阳性）
```

假设你的检测结果是阳性。关心的问题是：P(D|+) = ?

```
P(+|D) * P(D) = 0.99 * 0.001 = 0.00099
P(+|¬D) * P(¬D) = 0.01 * 0.999 = 0.00999

P(+) = 0.00099 + 0.00999 = 0.01098

P(D|+) = 0.00099 / 0.01098 ≈ 0.090
```

给定阳性检测结果后，你患病概率大约为9%，而不是99%。直观来说：假阳性远远多于真阳性，因为疾病非常罕见。你看到的大部分阳性结果来自健康人。

### 先验压倒似然

贝叶斯定理的一个关键直觉：如果你有强烈的先验信念，即使非常强的证据也改变不了多少。

```
强先验 + 强似然 = 适度后验（先验占主导）
弱先验 + 强似然 = 强后验（证据占主导）
```

当数据稀少时（训练初期，小样本量），先验占主导。当数据丰富时（训练后期，大数据集），似然占主导。先验 vs 数的权衡正是贝叶斯定理的核心。

### 贝叶斯更新

每次你观察到新数据，后验成为新的先验。你不断重复这个过程：

```
初始：P(A)（先验）
看到B1：P(A|B1) = P(B1|A) * P(A) / P(B1)（后1，新的先验）
看到B2：P(A|B2,B1) ∝ P(B2|A) * P(A|B1)
看到B3：P(A|B3,B2,B1) ∝ P(B3|A) * P(A|B2,B1)
...
```

如果所有证据都是独立的（通常在实际中不成立，但它是一个方便的假设），贝叶斯更新等于将似然相乘：

```
P(A|B1,...,Bn) ∝ P(A) * ∏ P(Bi|A)
```

这被称为朴素贝叶斯 (Naive Bayes) 假设。"朴素 (Naive)"是因为它假装所有特征彼此独立——在现实世界中这一假设通常不成立，但算法仍然能惊人地好用。

### 链式法则

联合概率可以通过条件概率拆解：

```
P(A,B,C) = P(A) * P(B|A) * P(C|A,B)
P(A,B,C,D) = P(A) * P(B|A) * P(C|A,B) * P(D|A,B,C)
一般形式：
P(X1,...,Xn) = ∏ P(Xi | X1,...,Xi-1)
```

这看起来像一个理论上的好奇，但它实际上是自回归语言模型的核心。GPT生成文本的方式正是：

```
P(token1, ..., tokenN) = P(token1) * P(token2|token1) * P(token3|token1,token2) * ...
```

每个token都是从给定之前所有token后的条件分布中采样的。这就是链式法则。

### 贝叶斯定理与假设检验

贝叶斯定理为比较假设提供了直接的语言：

假设你有两个假设H1和H2。你可以比较它们的后验概率：

```
P(H1|E) / P(H2|E) = P(E|H1) / P(E|H2) * P(H1) / P(H2)

后验比率 = 贝叶斯因子 (Bayes factor) * 先验比率 (prior ratio)
```

贝叶斯因子P(E|H1)/P(E|H2)量化了证据支持H1相对于H2的程度：
- 贝叶斯因子 = 1：证据不偏袒任何一方
- 贝叶斯因子 > 1：证据支持H1
- 贝叶斯因子 < 1：证据支持H2

在机器学习中，这被用来做模型选择：给定数据，哪个模型更可能正确？

### 朴素贝叶斯分类器

朴素贝叶斯将贝叶斯定理应用于分类问题。给定特征X1,...,Xn，预测类别C：

```
P(C|X1,...,Xn) ∝ P(C) * ∏ P(Xi|C)
```

"朴素"假设意味着我们假装所有特征在给定类别的情况下是条件独立的。这在实际中几乎从不成立，但算法仍然有效，因为尽管概率估计是错的，但不同类别之间的大小比较往往是正确的。

对于连续特征，你假设一个分布（通常是高斯分布 (Gaussian)）。对于离散特征，你使用计数和拉普拉斯平滑 (Laplace smoothing)。

P(feature|class)通过计数来估计：
- 高斯朴素贝叶斯：拟合每个类别的均值和方差，P(x|C)来自高斯概率密度函数 (PDF)
- 多项式朴素贝叶斯 (Multinomial Naive Bayes)：计数出现次数。常见于文本分类（词袋模型）
- 伯努利朴素贝叶斯 (Bernoulli Naive Bayes)：二值特征（单词存在/不存在）

朴素贝叶斯非常快（训练时间O(nd)，预测时间O(d)），内存占用小，训练样本少时表现良好，并且具有高度可解释性——你可以精确查看每个特征对每个类别的贡献。

稀疏高维数据（如文本）的典型pipeline：向量化 (vectorize) → 拟合朴素贝叶斯 → 预测。

### 共轭先验 (Conjugate Prior)

如果你选择一个与似然函数 (likelihood function) 有共轭关系的先验分布，那么后验分布与先验分布属于同一族。这使得计算易于处理。

**Beta-二项分布示例：**

- 似然：二项分布 (Binomial) 似然（抛硬币，正面次数）
- 先验：Beta(alpha, beta) 分布
- 后验：Beta(alpha + #正面, beta + #反面)

在没有共轭先验的情况下，后验不会是一个整洁的闭合形式分布——你需要用数值方法近似它（MCMC、变分推断）。

### MAP 估计

给定后验分布P(θ|D)，有两种总结方式：

1. **完全贝叶斯推断 (Full Bayesian inference)**：保留整个后验分布P(θ|D)。用后验权重对所有可能的θ值进行平均。计算昂贵但原则性强。

2. **最大后验估计 (Maximum A Posteriori, MAP)**：找到使后验最大化的单一θ值。

```
θ_MAP = argmax P(θ|D) = argmax P(D|θ) * P(θ)
```

取对数后：
```
θ_MAP = argmax [log P(D|θ) + log P(θ)]
```

这正是具有正则化的最大似然估计 (MLE)！不同的先验对应不同的正则化项：
- 高斯先验 (Gaussian prior) → L2 正则化 (权重衰减 / weight decay)
- 拉普拉斯先验 (Laplace prior) → L1 正则化 (稀疏)

正则化不是启发式技巧——它是用贝叶斯语言表达的稀有性先验信念。

## 构建它

### 步骤1：贝叶斯更新器

```python
class BayesianUpdater:
    """手动实现贝叶斯更新以建立直觉"""
    def __init__(self, prior):
        self.prior = prior
        self.posterior = prior

    def update(self, likelihood_true, likelihood_false):
        """观察到某事物的证据后更新"""
        # P(E) = P(E|H) * P(H) + P(E|¬H) * P(¬H)
        evidence = (
            likelihood_true * self.posterior +
            likelihood_false * (1 - self.posterior)
        )
        # P(H|E) = P(E|H) * P(H) / P(E)
        self.posterior = (likelihood_true * self.posterior) / evidence
        return self.posterior
```

### 步骤2：高斯朴素贝叶斯

```python
import math

class GaussianNB:
    """高斯朴素贝叶斯分类器"""

    def fit(self, X, y):
        self.classes = sorted(set(y))
        self.n_classes = len(self.classes)
        self.n_features = len(X[0])

        # 计算先验 P(C)
        self.priors = {}
        for c in self.classes:
            self.priors[c] = sum(1 for label in y if label == c) / len(y)

        # 计算每个类别每个特征的均值和标准差
        self.means = {c: [0] * self.n_features for c in self.classes}
        self.stds = {c: [0] * self.n_features for c in self.classes}

        for c in self.classes:
            class_samples = [X[i] for i in range(len(X)) if y[i] == c]
            for j in range(self.n_features):
                values = [sample[j] for sample in class_samples]
                self.means[c][j] = sum(values) / len(values)
                variance = sum((v - self.means[c][j]) ** 2 for v in values) / len(values)
                self.stds[c][j] = math.sqrt(variance + 1e-9)  # 防止除零

    def _gaussian_pdf(self, x, mean, std):
        """P(x|C) = 高斯概率密度函数"""
        if std < 1e-9:
            return 1.0 if abs(x - mean) < 1e-9 else 0.0
        exponent = -0.5 * ((x - mean) / std) ** 2
        return (1 / (std * math.sqrt(2 * math.pi))) * math.exp(exponent)

    def predict_proba(self, x):
        scores = {}
        for c in self.classes:
            # 从先验的对数开始
            log_prob = math.log(self.priors[c])
            # 乘以（求和）似然
            for j in range(self.n_features):
                p = self._gaussian_pdf(x[j], self.means[c][j], self.stds[c][j])
                if p > 0:
                    log_prob += math.log(p)
                else:
                    log_prob = float('-inf')
                    break
            scores[c] = log_prob

        # 归一化：log(P) → P（使用log-sum-exp技巧）
        max_log = max(scores.values())
        exp_scores = {c: math.exp(s - max_log) for c, s in scores.items()}
        total = sum(exp_scores.values())
        return {c: v / total for c, v in exp_scores.items()}

    def predict(self, x):
        probs = self.predict_proba(x)
        return max(probs, key=probs.get)
```

### 步骤3：在文本上测试与评估

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 加载并向量化文本
vectorizer = CountVectorizer(stop_words='english', max_features=5000)

# 在实际pipeline中拆分 → 向量化 → 拟合 → 预测 → 评估。
# 该框同时使用MultinomialNB（处理词袋模型的离散计数）
# 或基于TF-IDF权重翻译为多项式似然。

# sklearn提供：
#   MultinomialNB：词频（整数计数）
#   GaussianNB：连续特征
#   BernoulliNB：二值特征（存在/不存在）
#   ComplementNB：不平衡文本类别的稳健替代方案
```

## 使用它

### Python (scikit-learn)

朴素贝叶斯的一个典型训练循环：

```python
# 文本分类pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

pipeline.fit(docs_train, labels_train)
predictions = pipeline.predict(docs_test)

# 评估：准确率、精确率、召回率以及混淆矩阵
```

在真实场景中完全使用贝叶斯推断（边缘化参数，而非取MAP点估计）通常过于昂贵。Neal's Funnel等结构使得采样变慢，并且大模型的大参数维度使得数值近似难以收敛。这就是为什么深度学习使用SGD并接受MAP / MLE的原因。

## 练习

1. **医疗检测。** 一种疾病每5000人中有1人患病。一项检测有95%的准确率（真阳性 (true positive) 率）和3%的假阳率。给定阳性检测结果，患病的概率是多少？通过贝叶斯更新器运行计算。

2. **垃圾邮件过滤。** 一个词在垃圾邮件中出现40次/1000封，在非垃圾邮件中出现2次/1000封。先验概率P(spam) = 0.3。给定该词出现，是一封垃圾邮件的概率是多少？

3. **链式法则展开。** 将P(A,B,C,D)用条件概率展开。展示完整的4步因式分解。将其与GPT生成4个token的序列进行类比，说明每个条件概率对应模型中的哪一步。

4. **先验敏感性。** 对于医疗检测示例，绘制给定阳性检测结果后的P(D|+)随患病率（先验）从0.0001到0.1变化的曲线。在先验超过哪个阈值时，后验概率会超过95%？

## 现实世界的连接

- 朴素贝叶斯仍然是垃圾邮件过滤、情感分析和文档分类的强有力基线。任何pre-LLM文本分类器基线比较几乎必然包含该算法。它以极快的速度处理高维稀疏特征，并且在训练样本极少的情况下仍然有效。

- 概率性的“先验 → 证据 → 更新 → 新先验”循环对于理解贝叶斯优化 (Bayesian optimization)（超参数调优）、卡尔曼滤波器（机器人技术、跟踪）和强化学习中的信念状态至关重要。

- 链式法则分解是自回归生成的核心：GPT将`P(sequence)`建模为`∏ P(token_i | tokens_{<i})`。每个token是通过模型输出的读出向量上的softmax从该条件分布中采样的。

- 正则化等价于先验：L2权重衰减 = 假设权重服从高斯先验；L1正则化 = 假设权重服从拉普拉斯（稀疏）先验。当你选择正则化方案时，你是在对关于参数合理值的先验信念进行编码。