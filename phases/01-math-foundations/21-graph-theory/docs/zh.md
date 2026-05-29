# 图论与机器学习 (Graph Theory for Machine Learning)

> 图是关系的数据结构。如果你的数据有连接，你就需要图论。

**类型：** 构建 (Build)
**语言：** Python
**前置要求：** 第一阶段，第 01-03 课（线性代数、矩阵）
**时间：** 约 90 分钟

## 学习目标 (Learning Objectives)

- 使用邻接矩阵/邻接表表示构建图类，并实现 BFS 和 DFS 遍历
- 计算图拉普拉斯矩阵 (Graph Laplacian)，利用其特征值检测连通分量并对节点进行聚类
- 实现一轮 GNN 风格的消息传递 (Message Passing)，即归一化邻接矩阵乘法
- 使用 Fiedler 向量 (Fiedler Vector) 对图进行谱聚类 (Spectral Clustering)

## 问题 (The Problem)

社交网络、分子、知识图谱、引文网络、道路地图——这些都是图。传统机器学习将数据视为扁平表格。每一行是独立的。每个特征是一列。但当连接结构很重要时，表格就失效了。

考虑一个社交网络。你想预测用户会购买什么产品。他们的购买历史很重要。但他们朋友的购买历史更重要。连接中携带着信号。

或者考虑一个分子。你想预测它是否会与蛋白质结合。原子很重要，但真正重要的是原子之间如何键合。结构本身就是数据。

图神经网络 (Graph Neural Networks, GNNs) 是深度学习领域增长最快的方向。它们驱动着药物发现、社交推荐、欺诈检测和知识图谱推理。每个 GNN 都建立在同一个基础之上：基本的图论。

你需要四样东西：
1. 一种将图表示为矩阵的方法（这样你就可以对它们做乘法）
2. 探索图结构的遍历算法
3. 拉普拉斯矩阵——谱图理论中最重要的矩阵
4. 消息传递——使 GNN 工作的核心操作

## 概念 (The Concept)

### 图：节点与边 (Graphs: Nodes and Edges)

图 G = (V, E) 由顶点（节点）V 和边 E 组成。每条边连接两个节点。

**有向 vs 无向。** 在无向图中，边 (u, v) 意味着 u 连接到 v 且 v 连接到 u。在有向图中，边 (u, v) 意味着 u 指向 v，但不一定反过来。

**加权 vs 无权。** 在无权图中，边要么存在要么不存在。在加权图中，每条边有一个数值权重——距离、成本或强度。

| 图类型 | 示例 |
|-----------|---------|
| 无向、无权 | Facebook 好友网络 |
| 有向、无权 | Twitter 关注网络 |
| 无向、加权 | 道路地图（距离） |
| 有向、加权 | 网页链接（PageRank 分数） |

### 邻接矩阵 (The Adjacency Matrix)

邻接矩阵 A 是核心表示。对于有 n 个节点的图：

```
A[i][j] = 1    如果存在从节点 i 到节点 j 的边
A[i][j] = 0    否则
```

对于无向图，A 是对称的：A[i][j] = A[j][i]。对于加权图，A[i][j] = 边 (i, j) 的权重。

**示例——一个三角形：**

```
节点: 0, 1, 2
边: (0,1), (1,2), (0,2)

A = [[0, 1, 1],
     [1, 0, 1],
     [1, 1, 0]]
```

邻接矩阵是每个 GNN 的输入。对 A 的矩阵操作对应于对图的操作。

### 度 (Degree)

节点的度是连接到它的边的数量。对于有向图，有入度（进入的边）和出度（出去的边）。

度矩阵 D 是对角的：

```
D[i][i] = 节点 i 的度
D[i][j] = 0    当 i != j
```

对于三角形示例：D = diag(2, 2, 2)，因为每个节点都连接到另外两个节点。

度告诉你节点的重要性。高度数 = 枢纽节点。网络的度分布揭示了其结构。社交网络遵循幂律分布（少数枢纽，大量叶子节点）。随机图的度服从泊松分布。

### BFS 与 DFS

两种基本的图遍历算法。你两者都需要。

**广度优先搜索 (Breadth-First Search, BFS)：** 先探索所有邻居，再探索邻居的邻居。使用队列（FIFO）。

```
从节点 0 开始 BFS：
  访问 0
  队列: [1, 2]        （0 的邻居）
  访问 1
  队列: [2, 3]        （加入 1 的邻居）
  访问 2
  队列: [3]           （2 的邻居已访问）
  访问 3
  队列: []            （完成）
```

BFS 在无权图中找到最短路径。从起点到任意节点的距离等于该节点首次被发现的 BFS 层级。这就是为什么 BFS 用于社交网络中的跳数距离。

**深度优先搜索 (Depth-First Search, DFS)：** 尽可能深入，然后回溯。使用栈（LIFO）或递归。

```
从节点 0 开始 DFS：
  访问 0
  栈: [1, 2]        （0 的邻居）
  访问 2               （从栈中弹出）
  栈: [1, 3]         （加入 2 的邻居）
  访问 3               （从栈中弹出）
  栈: [1]
  访问 1               （从栈中弹出）
  栈: []             （完成）
```

DFS 适用于：
- 查找连通分量（从未访问节点运行 DFS）
- 环检测（DFS 树中的回边）
- 拓扑排序（反向 DFS 完成顺序）

| 算法 | 数据结构 | 找到 | 用例 |
|-----------|---------------|-------|----------|
| BFS | 队列 | 最短路径 | 社交网络距离、知识图谱遍历 |
| DFS | 栈 | 连通分量、环 | 连通性、拓扑排序 |

### 图拉普拉斯矩阵 (The Graph Laplacian)

L = D - A。谱图理论中最重要的矩阵。

对于三角形：

```
D = [[2, 0, 0],    A = [[0, 1, 1],    L = [[2, -1, -1],
     [0, 2, 0],         [1, 0, 1],         [-1, 2, -1],
     [0, 0, 2]]         [1, 1, 0]]         [-1, -1,  2]]
```

拉普拉斯矩阵具有显著的性质：

1. **L 是半正定的。** 所有特征值 >= 0。

2. **零特征值的数量等于连通分量的数量。** 一个连通图恰好有一个零特征值。一个有 3 个不连通分量的图有三个零特征值。

3. **最小的非零特征值（Fiedler 值）衡量连通性。** 大的 Fiedler 值意味着图是良好连通的。小的 Fiedler 值意味着图有一个薄弱点——一个瓶颈。

4. **Fiedler 值的特征向量（Fiedler 向量）揭示了最佳分割。** 正值的节点归入一组，负值的节点归入另一组。这就是谱聚类。

```mermaid
graph TD
    subgraph "图到矩阵"
        G["图 G"] --> A["邻接矩阵 A"]
        G --> D["度矩阵 D"]
        A --> L["拉普拉斯矩阵 L = D - A"]
        D --> L
    end
    subgraph "谱分析"
        L --> E["L 的特征值"]
        L --> V["L 的特征向量"]
        E --> C["连通分量（零特征值）"]
        E --> F["连通性（Fiedler 值）"]
        V --> S["谱聚类"]
    end
```

### 谱性质 (Spectral Properties)

邻接矩阵和拉普拉斯矩阵的特征值无需任何遍历即可揭示结构性质。

**谱聚类** 的工作原理如下：
1. 计算拉普拉斯矩阵 L
2. 找到 L 的 k 个最小特征向量（跳过第一个，对于连通图它是全 1 向量）
3. 将这些特征向量用作每个节点的新坐标
4. 在这些坐标上运行 k-means

为什么这有效？L 的特征向量编码了图上"最平滑"的函数。良好连通的节点获得相似的特征向量值。被瓶颈分隔的节点获得不同的值。特征向量自然地分离聚类。

**随机游走联系。** 归一化拉普拉斯矩阵与图上的随机游走相关。随机游走的平稳分布与节点度成正比。混合时间（游走收敛的速度）取决于谱间隙。

### 消息传递 (Message Passing)

图神经网络的核心操作。每个节点从其邻居收集消息，聚合它们，并更新自己的状态。

```
h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u in neighbors(v)}))
```

在最简单的形式中，AGGREGATE = 均值，UPDATE = 线性变换 + 激活：

```
h_v^(k+1) = sigma(W * mean({h_u^(k) : u in neighbors(v)}))
```

这实际上是矩阵乘法。如果 H 是所有节点特征的矩阵，A 是邻接矩阵：

```
H^(k+1) = sigma(A_norm * H^(k) * W)
```

其中 A_norm 是归一化邻接矩阵（每行和为 1）。

一轮消息传递让每个节点"看到"其直接邻居。两轮让它看到邻居的邻居。K 轮让每个节点获得其 K 跳邻域的信息。

```mermaid
graph LR
    subgraph "第 0 轮"
        A0["节点 A: [1,0]"]
        B0["节点 B: [0,1]"]
        C0["节点 C: [1,1]"]
    end
    subgraph "第 1 轮（聚合邻居）"
        A1["节点 A: avg(B,C) = [0.5, 1.0]"]
        B1["节点 B: avg(A,C) = [1.0, 0.5]"]
        C1["节点 C: avg(A,B) = [0.5, 0.5]"]
    end
    A0 --> A1
    B0 --> A1
    C0 --> A1
    A0 --> B1
    C0 --> B1
    A0 --> C1
    B0 --> C1
```

### 概念与机器学习应用

| 概念 | 机器学习应用 |
|---------|---------------|
| 邻接矩阵 | GNN 输入表示 |
| 图拉普拉斯矩阵 | 谱聚类、社区检测 |
| BFS/DFS | 知识图谱遍历、路径查找 |
| 度分布 | 节点重要性、特征工程 |
| 消息传递 | GNN 层（GCN、GAT、GraphSAGE） |
| L 的特征值 | 社区检测、图划分 |
| 谱聚类 | 无监督节点分组 |
| PageRank | 节点重要性、网页搜索 |

## 构建 (Build It)

### 步骤 1：从零实现图类

```python
class Graph:
    def __init__(self, n_nodes, directed=False):
        self.n = n_nodes
        self.directed = directed
        self.adj = {i: {} for i in range(n_nodes)}

    def add_edge(self, u, v, weight=1.0):
        self.adj[u][v] = weight
        if not self.directed:
            self.adj[v][u] = weight

    def neighbors(self, node):
        return list(self.adj[node].keys())

    def degree(self, node):
        return len(self.adj[node])

    def adjacency_matrix(self):
        import numpy as np
        A = np.zeros((self.n, self.n))
        for u in range(self.n):
            for v, w in self.adj[u].items():
                A[u][v] = w
        return A

    def degree_matrix(self):
        import numpy as np
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            D[i][i] = self.degree(i)
        return D

    def laplacian(self):
        return self.degree_matrix() - self.adjacency_matrix()
```

邻接表 (`self.adj`) 高效地存储邻居。邻接矩阵转换使用 numpy，因为所有谱操作都需要它。

### 步骤 2：BFS 和 DFS

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    order = []
    distances = {}
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        node, dist = queue.popleft()
        order.append(node)
        distances[node] = dist
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return order, distances


def dfs(graph, start):
    visited = set()
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in reversed(graph.neighbors(node)):
            if neighbor not in visited:
                stack.append(neighbor)
    return order
```

BFS 使用 deque（双端队列）实现 O(1) 的 popleft。DFS 使用列表作为栈。两者都恰好访问每个节点一次——O(V + E) 时间。

### 步骤 3：连通分量和拉普拉斯特征值

```python
def connected_components(graph):
    visited = set()
    components = []
    for node in range(graph.n):
        if node not in visited:
            order, _ = bfs(graph, node)
            visited.update(order)
            components.append(order)
    return components


def laplacian_eigenvalues(graph):
    import numpy as np
    L = graph.laplacian()
    eigenvalues = np.linalg.eigvalsh(L)
    return eigenvalues
```

`eigvalsh` 用于对称矩阵——对于无向图，拉普拉斯矩阵总是对称的。它按升序返回特征值。数零的个数即可找到连通分量的数量。

### 步骤 4：谱聚类

```python
def spectral_clustering(graph, k=2):
    import numpy as np
    L = graph.laplacian()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    features = eigenvectors[:, 1:k+1]

    labels = np.zeros(graph.n, dtype=int)
    for i in range(graph.n):
        if features[i, 0] >= 0:
            labels[i] = 0
        else:
            labels[i] = 1
    return labels
```

对于 k=2，Fiedler 向量的符号将图分成两个聚类。对于 k>2，你需要在前 k 个特征向量上运行 k-means（排除平凡的全 1 特征向量）。

### 步骤 5：消息传递

```python
def message_passing(graph, features, weight_matrix):
    import numpy as np
    A = graph.adjacency_matrix()
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_norm = A / row_sums
    aggregated = A_norm @ features
    output = aggregated @ weight_matrix
    return output
```

这是一轮 GNN 消息传递。每个节点的新特征是其邻居特征的加权平均，经过权重矩阵变换。堆叠多轮以将信息传播得更远。

## 使用 (Use It)

使用 networkx 和 numpy，同样的操作只需一行代码：

```python
import networkx as nx
import numpy as np

G = nx.karate_club_graph()

A = nx.adjacency_matrix(G).toarray()
L = nx.laplacian_matrix(G).toarray()

eigenvalues = np.linalg.eigvalsh(L.astype(float))
print(f"Smallest eigenvalues: {eigenvalues[:5]}")
print(f"Connected components: {nx.number_connected_components(G)}")

communities = nx.community.greedy_modularity_communities(G)
print(f"Communities found: {len(communities)}")

pr = nx.pagerank(G)
top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"Top 5 PageRank nodes: {top_nodes}")
```

networkx 使用优化的 C 后端处理任意大小的图。在生产环境中使用它。使用你从零实现的版本来理解它的工作原理。

### numpy 谱分析

```python
import numpy as np

A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
])

D = np.diag(A.sum(axis=1))
L = D - A

eigenvalues, eigenvectors = np.linalg.eigh(L)
print(f"Eigenvalues: {np.round(eigenvalues, 4)}")
print(f"Fiedler value: {eigenvalues[1]:.4f}")
print(f"Fiedler vector: {np.round(eigenvectors[:, 1], 4)}")

fiedler = eigenvectors[:, 1]
group_a = np.where(fiedler >= 0)[0]
group_b = np.where(fiedler < 0)[0]
print(f"Cluster A: {group_a}")
print(f"Cluster B: {group_b}")
```

Fiedler 向量承担了繁重的工作。正值条目在一个聚类中，负值条目在另一个聚类中。不需要迭代优化——只需一次特征分解。

## 产出 (Ship It)

本课产出：
- `outputs/skill-graph-analysis.md`——分析图结构数据的技能参考

## 关联 (Connections)

| 概念 | 出现场景 |
|---------|------------------|
| 邻接矩阵 | GCN、GAT、GraphSAGE 输入 |
| 拉普拉斯矩阵 | 谱聚类、ChebNet 滤波器 |
| BFS | 知识图谱遍历、最短路径查询 |
| 消息传递 | 每个 GNN 层、神经消息传递 |
| 谱间隙 | 图连通性、随机游走的混合时间 |
| 度分布 | 幂律网络、节点特征工程 |
| 连通分量 | 预处理、处理不连通图 |
| PageRank | 节点重要性排序、注意力初始化 |

GNN 值得特别提及。GCN（Kipf & Welling, 2017）中的图卷积操作使用添加了自环的邻接矩阵 A_hat = A + I：

```text
H^(l+1) = sigma(D_hat^(-1/2) * A_hat * D_hat^(-1/2) * H^(l) * W^(l))
```

其中 A_hat = A + I（邻接矩阵加自环），D_hat 是 A_hat 的度矩阵。自环确保每个节点在聚合时包含自己的特征。这正是带有对称归一化的消息传递。D_hat^(-1/2) * A_hat * D_hat^(-1/2) 是归一化邻接矩阵。拉普拉斯矩阵出现在这里，因为这种归一化与 L_sym = I - D^(-1/2) * A * D^(-1/2) 相关。理解拉普拉斯矩阵意味着理解 GCN 为什么有效。

## 练习 (Exercises)

1. **从零实现 PageRank。** 从均匀分数开始。每一步：score(v) = (1-d)/n + d * sum(score(u)/out_degree(u))，对所有指向 v 的 u 求和。使用 d=0.85。运行直到收敛（变化 < 1e-6）。在一个小型网页图上测试。

2. **使用谱聚类查找社区。** 创建一个有两个明显分离聚类的图（例如，由一条边连接的两个团）。运行谱聚类并验证它找到了正确的分割。当你添加更多跨聚类边时会发生什么？

3. **实现 Dijkstra 算法** 用于加权图中的最短路径。将结果与在相同图上使用均匀权重的 BFS 进行比较。

4. **构建一个 2 层消息传递网络。** 使用不同的权重矩阵应用两次消息传递。展示经过 2 轮后，每个节点拥有其 2 跳邻域的信息。

5. **分析一个真实世界的图。** 使用空手道俱乐部图（Karate Club graph，34 个节点，78 条边）。计算度分布、拉普拉斯特征值和谱聚类。将谱聚类结果与已知的真实分割进行比较。

## 关键术语 (Key Terms)

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 图 (Graph) | "节点和边" | 编码成对关系的数学结构 G=(V,E) |
| 邻接矩阵 (Adjacency matrix) | "连接表" | 一个 n x n 矩阵，其中 A[i][j] = 1 表示节点 i 和 j 相连 |
| 度 (Degree) | "一个节点的连接程度" | 接触一个节点的边的数量 |
| 拉普拉斯矩阵 (Laplacian) | "D 减 A" | L = D - A，其特征值揭示图结构的矩阵 |
| Fiedler 值 (Fiedler value) | "代数连通性" | L 的最小非零特征值，衡量图的连通程度 |
| BFS | "逐层搜索" | 在深入之前访问所有邻居的遍历，找到最短路径 |
| DFS | "先深入" | 沿着一条路径走到尽头再回溯的遍历 |
| 消息传递 (Message passing) | "节点与邻居通信" | 每个节点聚合其邻居的信息，GNN 的核心 |
| 谱聚类 (Spectral clustering) | "通过特征向量聚类" | 使用拉普拉斯矩阵的特征向量对图进行划分 |
| 连通分量 (Connected component) | "一个独立的部分" | 每个节点都能到达其他每个节点的最大子图 |

## 进一步阅读 (Further Reading)

- **Kipf & Welling (2017)**——"Semi-Supervised Classification with Graph Convolutional Networks." 开创现代 GNN 的论文。展示了谱图卷积如何简化为消息传递。
- **Spielman (2012)**——"Spectral Graph Theory" 讲义。关于拉普拉斯矩阵、谱间隙和图划分的权威入门。
- **Hamilton (2020)**——"Graph Representation Learning." 涵盖从基础到应用的 GNN 书籍。
- **Bronstein et al. (2021)**——"Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." 统一框架论文。
- **Veličković et al. (2018)**——"Graph Attention Networks." 用注意力机制扩展消息传递。
