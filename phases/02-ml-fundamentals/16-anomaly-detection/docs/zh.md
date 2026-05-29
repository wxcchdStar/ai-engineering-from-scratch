# 异常检测（Anomaly Detection）

> 正常很容易定义。异常就是任何不符合正常的东西。

**类型：** 构建（Build）
**语言：** Python
**前置要求：** 第 2 阶段，第 01-09 课
**时间：** 约 75 分钟

## 学习目标（Learning Objectives）

- 从零实现 Z-score、IQR 和孤立森林（Isolation Forest）异常检测方法
- 区分点异常（Point Anomaly）、上下文异常（Contextual Anomaly）和集体异常（Collective Anomaly），并为每种类型选择合适的检测方法
- 解释为什么异常检测被构建为对正常数据建模，而不是对异常进行分类
- 比较无监督异常检测与监督分类，并评估新异常覆盖率与精确率之间的权衡

## 问题（The Problem）

一张信用卡下午 2 点在纽约使用，然后下午 2:05 在东京使用。一个工厂传感器读数为 150 度，而正常范围是 80-120。一台服务器每秒发送 50,000 个请求，而日均值为 200。

这些都是异常。发现它们很重要。欺诈造成数十亿美元的损失。设备故障导致停机。网络入侵导致数据泄露。

挑战：你很少有异常的标注样本。欺诈仅占交易的 0.1%。设备故障每年只发生几次。你无法训练标准分类器，因为"异常"类别中几乎没有任何东西可以学习。即使你有一些标签，你见过的异常也不是你将遇到的唯一类型。明天的欺诈手段与今天的不同。

异常检测翻转了问题。与其学习什么是异常，不如学习什么是正常。任何偏离正常的东西都是可疑的。这在没有标签的情况下有效，能适应新类型的异常，并能扩展到海量数据集。

## 概念（The Concept）

### 异常类型

并非所有异常都相同：

- **点异常（Point Anomalies）。** 单个数据点，无论上下文如何都是不寻常的。温度读数为 500 度。一笔 50,000 美元的交易来自一个通常消费 50 美元的账户。
- **上下文异常（Contextual Anomalies）。** 一个数据点在其上下文中是不寻常的。90 度的温度在夏天是正常的，在冬天是异常的。相同的值，不同的上下文。
- **集体异常（Collective Anomalies）。** 一组数据点作为一个整体是不寻常的，即使每个单独的点可能是正常的。五次登录失败是正常的。连续五十次是暴力攻击。

大多数方法检测点异常。上下文异常需要时间或位置特征。集体异常需要序列感知方法。

```mermaid
flowchart TD
    A[异常类型（Anomaly Types）] --> B[点异常（Point Anomaly）]
    A --> C[上下文异常（Contextual Anomaly）]
    A --> D[集体异常（Collective Anomaly）]

    B --> B1["单个异常值<br/>温度：500F"]
    C --> C1["上下文中异常<br/>1月达到90F"]
    D --> D1["异常序列<br/>50次登录失败"]

    style B fill:#fdd,stroke:#333
    style C fill:#ffd,stroke:#333
    style D fill:#fdf,stroke:#333
```

### 无监督框架

在标准分类中，你有两个类别的标签。在异常检测中，你通常面临三种情况之一：

1. **完全无监督。** 根本没有标签。你在所有数据上拟合检测器，希望异常足够罕见，不会破坏"正常"模型。
2. **半监督。** 你有一个仅包含正常数据的干净数据集。你在这个干净集上拟合，并对其他所有数据进行评分。这是可能情况下的最强设置。
3. **弱监督。** 你有少量标注的异常。将其用于评估，而非训练。进行无监督训练，然后在标注子集上测量精确率（Precision）/召回率（Recall）。

关键洞察：异常检测与分类有根本性的不同。你在建模正常数据的分布，而不是两个类别之间的决策边界。

### 监督 vs 无监督：权衡

如果你确实有标注的异常，你是应该将其用于训练（监督分类）还是仅用于评估（无监督检测）？

**监督（作为分类处理）：**
- 捕获你以前见过的确切异常类型
- 对已知异常类型的精确率更高
- 完全遗漏新类型的异常
- 当新异常类型出现时需要重新训练
- 需要足够的异常样本（通常太少）

**无监督（建模正常，标记偏离）：**
- 捕获任何偏离正常的情况，包括新类型
- 不需要标注的异常
- 误报率较高（并非所有不寻常的都是坏的）
- 对分布偏移更鲁棒

在实践中，最好的系统结合两者：无监督检测用于广泛覆盖，监督模型用于已知的高优先级异常类型，人工审查用于模糊情况。

### Z-Score 方法

最简单的方法。计算每个特征的均值和标准差。标记任何距离均值超过 k 个标准差的点。

```text
z_score = (x - mean) / std
anomaly if |z_score| > threshold
```

默认阈值是 3.0（对于高斯分布，99.7% 的正常数据落在 3 个标准差之内）。

**优点：** 简单。快速。可解释（"这个值距离正常值 4.5 个标准差"）。

**缺点：** 假设数据服从正态分布。对训练数据中的异常值敏感（异常值会偏移均值并膨胀标准差，使它们更难被检测到）。在多峰分布上失效。

**何时效果良好：** 数据大致呈钟形的单特征监控。服务器响应时间、制造公差、具有稳定基线的传感器读数。

**何时失效：** 多簇数据（两个办公室地点有不同的基线温度）、偏斜数据（交易金额中 1000 美元罕见但不是异常）、训练集中存在异常值的数据。

### IQR 方法

比 Z-score 更鲁棒。使用四分位距（Interquartile Range）而不是均值和标准差。

```
Q1 = 第25百分位数
Q3 = 第75百分位数
IQR = Q3 - Q1
lower_bound = Q1 - factor * IQR
upper_bound = Q3 + factor * IQR
anomaly if x < lower_bound or x > upper_bound
```

默认因子是 1.5。

**优点：** 对异常值鲁棒（百分位数不受极端值影响）。适用于偏斜分布。不需要正态性假设。

**缺点：** 仅限单变量（对每个特征独立应用）。无法检测仅在综合考虑特征时才不寻常的异常（一个点可能在每个特征上单独看是正常的，但在联合空间中却是异常的）。

**实用说明：** IQR 中的 1.5 因子对应箱线图（Box Plot）中的须线。须线外的点是潜在的异常值。使用 3.0 而不是 1.5 使检测器更保守（更少的标记，更少的误报）。正确的因子取决于你对误报的容忍度。

### 孤立森林（Isolation Forest）

关键洞察：异常是少且不同的。在数据的随机划分中，异常更容易被孤立——它们需要更少的随机分割就能与其他数据分开。

```mermaid
flowchart TD
    A[所有数据点] --> B{随机特征 + 随机分割}
    B --> C[左分区]
    B --> D[右分区]
    C --> E{随机特征 + 随机分割}
    E --> F[正常点 - 在树中很深]
    E --> G[需要更多分割......]
    D --> H["异常点 - 快速被孤立（短路径）"]

    style H fill:#fdd,stroke:#333
    style F fill:#dfd,stroke:#333
```

**工作原理：**
1. 构建许多随机树（一个孤立森林）
2. 在每个节点，选择一个随机特征和该特征最小值和最大值之间的随机分割值
3. 继续分割直到每个点被孤立（在自己的叶子中）
4. 异常在所有树中具有较短的平均路径长度

**为什么有效：** 正常点位于密集区域。需要许多随机分割才能将一个正常点与其邻居分开。异常点位于稀疏区域。只需一两次随机分割就足以孤立它们。

异常分数基于所有树中的平均路径长度，按随机二叉搜索树的期望路径长度进行归一化：

```
score(x) = 2^(-average_path_length(x) / c(n))
```

其中 `c(n)` 是 n 个样本的期望路径长度。分数接近 1 表示异常。分数接近 0.5 表示正常。分数接近 0 表示非常正常（深在密集簇中）。

**优点：** 无分布假设。适用于高维数据。扩展性好（样本量亚线性，因为每棵树使用子样本）。处理混合特征类型。

**缺点：** 在密集区域中的异常检测困难（掩蔽效应，Masking Effect）。当许多特征无关时，随机分割效率降低。

**关键超参数：**
- `n_estimators`：树的数量。100 通常足够。更多树给出更稳定的分数但计算更慢。
- `max_samples`：每棵树的样本数。原论文默认为 256。较小的值使单棵树不太准确但增加了多样性。子采样是孤立森林快速的原因——每棵树只看到数据的一小部分。
- `contamination`：预期的异常比例。仅用于设置阈值。不影响分数本身。

### 局部异常因子（Local Outlier Factor，LOF）

LOF 将一个点周围的局部密度与其邻居周围的密度进行比较。位于稀疏区域但被密集区域包围的点是异常的。

**工作原理：**
1. 对于每个点，找到其 k 个最近邻
2. 计算局部可达密度（Local Reachability Density，邻域有多密集）
3. 将每个点的密度与其邻居的密度进行比较
4. 如果一个点的密度远低于其邻居，则为离群点

**LOF 分数：**
- LOF 接近 1.0 表示密度与邻居相似（正常）
- LOF 大于 1.0 表示密度低于邻居（可能异常）
- LOF 远大于 1.0（例如，2.0+）表示密度显著较低（很可能是异常）

"局部"部分至关重要。考虑一个具有两个簇的数据集：一个包含 1000 个点的密集簇和一个包含 50 个点的稀疏簇。稀疏簇边缘的一个点不是全局不寻常的——它有 50 个邻居。但如果它的直接邻居比它更密集，则它在局部是不寻常的。LOF 捕获了全局方法所遗漏的这种细微差别。

**优点：** 检测局部异常（在其邻域中不寻常的点，即使它们在全局上不是不寻常的）。适用于不同密度的簇。

**缺点：** 在大型数据集上慢（朴素实现为 O(n^2)）。对 k 的选择敏感。在非常高维中效果不好（维数灾难影响距离计算）。

### 比较

| 方法 | 假设 | 速度 | 处理高维 | 检测局部异常 |
|------|------|------|---------|------------|
| Z-score | 正态分布 | 非常快 | 是（逐特征） | 否 |
| IQR | 无（逐特征） | 非常快 | 是（逐特征） | 否 |
| 孤立森林（Isolation Forest） | 无 | 快 | 是 | 部分 |
| LOF | 距离有意义 | 慢 | 差 | 是 |

### 评估挑战

评估异常检测器比评估分类器更难：

- **极端类别不平衡。** 0.1% 的异常，预测所有内容为"正常"可获得 99.9% 的准确率。准确率无用。
- **AUROC 有误导性。** 在严重不平衡的情况下，即使模型在实用阈值处遗漏了大多数异常，AUROC 也可能看起来不错。
- **更好的指标：** Precision@k（标记的前 k 个项目中有多少是真正的异常）、AUPRC（精确率-召回率曲线下面积）以及在固定误报率下的召回率。

```mermaid
flowchart LR
    A[原始数据（Raw Data）] --> B[仅在正常数据上训练]
    B --> C[对所有测试数据评分]
    C --> D[按异常分数排序]
    D --> E[评估 Top-K 标记项目]
    E --> F[Precision at K / AUPRC]

    style A fill:#f9f,stroke:#333
    style F fill:#9f9,stroke:#333
```

### 异常检测流水线

在实践中，异常检测遵循以下工作流：

1. **收集基线数据。** 理想情况下，是一个你知道没有（或很少有）异常的时期。
2. **特征工程（Feature Engineering）。** 原始特征加上派生特征（滚动统计量、时间特征、比率）。
3. **训练检测器。** 在基线数据上拟合。模型学习"正常"的样子。
4. **对新数据评分。** 每个新观测值获得一个异常分数。
5. **阈值选择。** 选择分数截止值。这是一个业务决策：更高的阈值意味着更少的误报但更多的遗漏异常。
6. **告警和调查。** 标记的点进入人工审查或自动响应。
7. **反馈收集。** 记录标记的项目是真正的异常还是误报。使用这些数据评估检测器并随时间调整阈值。

流水线永远不会"完成"。数据分布会变化，新的异常类型会出现，阈值需要调整。将异常检测视为一个活系统，而不是一次性模型。

## 构建它（Build It）

`code/anomaly_detection.py` 中的代码从零实现了 Z-score、IQR 和孤立森林。

### Z-Score 检测器

```python
def zscore_detect(X, threshold=3.0):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    z = np.abs((X - mean) / std)
    return z.max(axis=1) > threshold
```

简单且向量化。如果任何特征超过阈值，则标记该点。

### IQR 检测器

```python
def iqr_detect(X, factor=1.5):
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    outside = (X < lower) | (X > upper)
    return outside.any(axis=1)
```

### 孤立森林从零实现

从零实现构建了随机划分特征空间的孤立树：

```python
class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, depth=0):
        n, p = X.shape
        if depth >= self.max_depth or n <= 1:
            self.is_leaf = True
            self.size = n
            return self
        self.is_leaf = False
        self.feature = np.random.randint(p)
        x_min = X[:, self.feature].min()
        x_max = X[:, self.feature].max()
        if x_min == x_max:
            self.is_leaf = True
            self.size = n
            return self
        self.threshold = np.random.uniform(x_min, x_max)
        left_mask = X[:, self.feature] < self.threshold
        self.left = IsolationTree(self.max_depth).fit(X[left_mask], depth + 1)
        self.right = IsolationTree(self.max_depth).fit(X[~left_mask], depth + 1)
        return self
```

孤立一个点的路径长度决定了其异常分数。更短的路径意味着更异常。

`IsolationForest` 类包装了多棵树：

```python
class IsolationForest:
    def __init__(self, n_estimators=100, max_samples=256, seed=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples

    def fit(self, X):
        sample_size = min(self.max_samples, X.shape[0])
        max_depth = int(np.ceil(np.log2(sample_size)))
        for _ in range(self.n_estimators):
            idx = rng.choice(X.shape[0], size=sample_size, replace=False)
            tree = IsolationTree(max_depth=max_depth)
            tree.fit(X[idx])
            self.trees.append(tree)

    def anomaly_score(self, X):
        avg_path = 所有树的平均路径长度
        scores = 2.0 ** (-avg_path / c(max_samples))
        return scores
```

归一化因子 `c(n)` 是在具有 n 个元素的二叉搜索树中不成功搜索的期望路径长度。它等于 `2 * H(n-1) - 2*(n-1)/n`，其中 `H` 是调和数。这种归一化确保分数在不同大小的数据集之间可比。

### 演示场景

代码生成多个测试场景：

1. **具有离群点的单个簇。** 一个二维高斯簇，在远离中心的地方注入了异常。所有方法应该在这里都有效。
2. **多峰数据。** 三个不同大小和密度的簇。簇之间的点是异常的。Z-score 表现困难，因为逐特征范围很宽。
3. **高维数据。** 50 个特征，但异常仅在其中 5 个中有所不同。测试方法是否能在特征子集中找到异常。

每个演示使用精确率、召回率、F1 和 Precision@k 比较所有方法。

## 使用它（Use It）

使用 sklearn（使用库实现，而非从零实现）：

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(X_train)
predictions = iso.predict(X_test)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof.fit(X_train)
predictions = lof.predict(X_test)
```

注意 `contamination` 设置预期的异常比例。正确设置很重要——太低会遗漏异常，太高会产生误报。

`anomaly_detection.py` 中的代码在相同数据上比较了从零实现与 sklearn。

### sklearn Contamination 参数

sklearn 中的 `contamination` 参数决定了将连续异常分数转换为二元预测的阈值。它不会改变底层分数。

```python
iso_5 = IsolationForest(contamination=0.05)
iso_10 = IsolationForest(contamination=0.10)
```

两者产生相同的异常分数。但 `iso_5` 标记前 5%，而 `iso_10` 标记前 10%。如果你不知道真实的异常率（通常你不知道），将 contamination 设置为 "auto" 并直接使用原始分数。根据误报和漏报之间的成本权衡，设定你自己的阈值。

### 单类 SVM（One-Class SVM）

另一个值得了解的无监督异常检测器。单类 SVM 在高维特征空间中（使用核技巧）围绕正常数据拟合一个边界。

```python
from sklearn.svm import OneClassSVM

oc_svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
oc_svm.fit(X_train)
predictions = oc_svm.predict(X_test)
```

`nu` 参数近似于异常的比例。单类 SVM 在小型到中型数据集上效果很好，但不能扩展到非常大的数据（核矩阵呈二次增长）。

### 自编码器方法（Autoencoder Approach）预览

自编码器（Autoencoder）是学习压缩和重建数据的神经网络。在正常数据上训练。在测试时，异常具有高重建误差，因为网络只学会了重建正常模式。

这在第 3 阶段（深度学习）中涵盖，但原理是相同的：建模什么是正常的，标记偏离的。

### 集成异常检测

正如集成方法改善分类（第 11 课），组合多个异常检测器可以改善检测。最简单的方法：

1. 运行多个检测器（Z-score、IQR、孤立森林、LOF）
2. 将每个检测器的分数归一化到 [0, 1]
3. 对归一化分数取平均
4. 标记平均分数超过阈值的点

这减少了误报，因为不同的方法有不同的失效模式。被所有四种方法标记的点几乎可以肯定是异常的。只被一种方法标记的点可能只是该方法的特例。

更复杂的集成会根据每个检测器的估计可靠性（如果有的话，在具有已知异常的验证集上测量）进行加权。

### 生产环境考量

1. **阈值漂移。** 随着数据分布变化，固定阈值会过时。监控异常分数的分布并定期调整。
2. **告警疲劳。** 太多误报，操作员就不再关注。从高阈值开始（更少但更可靠的告警），随着信任的建立再降低。
3. **集成方法。** 在生产中，组合多个检测器。仅当多个方法都同意某个点是异常时才标记它。这显著减少了误报。
4. **特征工程。** 原始特征很少足够。添加滚动统计量、比率、距上次事件的时间以及领域特定特征。好的特征集比检测器的选择更重要。
5. **反馈循环。** 当操作员调查标记的项目并确认或驳回时，将其反馈到系统中。随时间积累标注数据以评估和改进检测器。

## 交付物（Ship It）

本课产生：
- `outputs/skill-anomaly-detector.md`——一个选择正确检测器的决策技能
- `code/anomaly_detection.py`——从零实现 Z-score、IQR 和孤立森林，并与 sklearn 比较

### 选择阈值

异常分数是连续的。你需要一个阈值来进行二元决策。这是一个业务决策，而不是技术决策。

考虑两种场景：
- **欺诈检测。** 遗漏欺诈代价高昂（拒付、客户信任）。误报成本是人工分析师 5 分钟的调查时间。设置较低阈值以捕获更多欺诈，接受更多误报。
- **设备维护。** 误报意味着不必要的停机，成本 50,000 美元。遗漏故障意味着 500,000 美元的维修费用。设置阈值以平衡这些成本。

在这两种情况下，最优阈值取决于误报和漏报之间的成本比率。在不同阈值处绘制精确率和召回率，叠加成本函数，选择最小成本点。

### 扩展到生产环境

对于生产中的实时异常检测：

1. **批量训练，在线评分。** 定期（每天、每周）在最近的正常数据上训练模型。在新观测到达时对其进行评分。
2. **特征计算必须匹配。** 如果你使用 30 天的滚动统计量进行训练，你需要 30 天的历史数据来为新观测计算特征。缓冲所需的历史数据。
3. **分数分布监控。** 随时间跟踪异常分数的分布。如果中位数分数向上漂移，要么数据在变化，要么模型过时了。
4. **可解释性。** 当标记异常时，说明原因。Z-score："特征 X 高于正常值 4.2 个标准差。"孤立森林："该点平均在 3.1 次分割中被孤立（正常点需要 8.5 次）。"

## 练习（Exercises）

1. **阈值调优。** 以 0.5 为步长，从 1.0 到 5.0 运行 Z-score 检测器。在每个阈值处绘制精确率和召回率。你的数据的最佳点在何处？

2. **多变量异常。** 创建二维数据，其中每个特征单独看起来正常，但组合是异常的（例如，远离主簇对角线的点）。展示逐特征 Z-score 会遗漏这些异常，但孤立森林能捕获它们。

3. **从零实现 LOF。** 使用 k-最近邻实现局部异常因子。在相同数据上比较 sklearn 的 LocalOutlierFactor。使用 k=10 和 k=50——k 的选择如何影响结果？

4. **流式异常检测。** 修改 Z-score 检测器以在流式设置中工作：在新点到达时使用 Welford 在线算法更新运行均值和方差。在相同数据上比较批量 Z-score。

5. **真实世界评估。** 取一个具有已知异常的数据集（例如，Kaggle 上的信用卡欺诈）。使用 precision@100、precision@500 和 AUPRC 评估所有四种方法。哪种方法效果最好？为什么？

## 关键术语（Key Terms）

| 术语 | 人们的说法 | 实际含义 |
|------|----------|---------|
| 异常（Anomaly） | "离群点，异常值" | 显著偏离正常数据预期模式的数据点 |
| 点异常（Point Anomaly） | "单个异常值" | 一个无论上下文如何都不寻常的个体观测 |
| 上下文异常（Contextual Anomaly） | "正常的值，错误的上下文" | 一个在其上下文（时间、位置等）中不寻常但在另一上下文中可能正常的观测 |
| 孤立森林（Isolation Forest） | "随机分割找离群点" | 一种随机树的集成，用比正常点更少的分割孤立异常点 |
| 局部异常因子（Local Outlier Factor） | "将密度与邻居比较" | 一种方法，标记局部密度远低于邻居密度的点 |
| Z-score | "与均值的标准差距离" | (x - mean) / std，以标准差为单位衡量一个点离中心多远 |
| IQR | "四分位距（Interquartile Range）" | Q3 - Q1，衡量中间 50% 数据分布的宽度，用于鲁棒的异常检测 |
| 污染率（Contamination） | "预期的异常比例" | 一个超参数，告诉检测器应将多少比例的数据标记为异常 |
| Precision@k | "前 k 个标记中有多少是真的" | 仅对 k 个最可疑的点计算的精确率，适用于不平衡的异常检测 |
| AUPRC | "精确率-召回率曲线下面积" | 一个总结所有阈值下精确率-召回率性能的指标，对于不平衡数据比 AUROC 更好 |

## 进一步阅读（Further Reading）

- [Liu et al., Isolation Forest (2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)——原始孤立森林论文
- [Breunig et al., LOF: Identifying Density-Based Local Outliers (2000)](https://dl.acm.org/doi/10.1145/342009.335388)——原始 LOF 论文
- [scikit-learn Outlier Detection docs](https://scikit-learn.org/stable/modules/outlier_detection.html)——所有 sklearn 异常检测器概览
- [Chandola et al., Anomaly Detection: A Survey (2009)](https://dl.acm.org/doi/10.1145/1541880.1541882)——异常检测方法综合综述
- [Goldstein and Uchida, A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms (2016)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152173)——在真实数据集上对 10 种方法的实证比较