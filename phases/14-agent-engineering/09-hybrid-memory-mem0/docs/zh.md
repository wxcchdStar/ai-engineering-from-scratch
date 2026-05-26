# 混合记忆：向量 + 图 + KV (Hybrid Memory: Mem0)

> Mem0 (Chhikara et al., 2025) 将记忆视为三个并行的存储——向量用于语义相似度，KV 用于快速事实查找，图用于实体关系推理。检索时通过评分层融合三者。这是 2026 年外部记忆的生产标准。

**类型：** 构建 (Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta 块)
**时间：** 约 75 分钟

## 学习目标

- 解释为什么单一存储（仅向量、仅图、仅 KV）对 Agent 记忆是不够的。
- 列举 Mem0 的三个并行存储及其各自优化的目标。
- 描述 Mem0 的融合评分——相关性、重要性、新鲜度——以及为什么它是加权和而非层级结构。
- 用标准库实现一个玩具三存储记忆，包含写入所有三个存储的 `add()` 和融合结果的 `search()`。

## 问题

单一存储对三类查询中的一类是错误的：

- **语义相似度** — "我们上周关于 Agent 漂移讨论了什么？" 向量胜出；KV 和图会遗漏。
- **事实查找** — "用户的电话号码是多少？" KV 胜出；向量是浪费，图是过度设计。
- **关系推理** — "哪些客户共享同一个账单实体？" 图胜出；向量和 KV 无法回答。

生产 Agent 在一个会话中发出所有三类查询。单一存储记忆对其中的两类总是错误的。Mem0 的贡献是将三者连接在单一的 `add`/`search` 接口后面，并用评分函数融合它们。

## 概念

### 三个并行存储

Mem0 (arXiv:2504.19413, 2025 年 4 月) 在 `add(text, user_id, metadata)` 时：

1. 从文本中提取候选事实（LLM 驱动的步骤）。
2. 将每个事实写入向量存储（嵌入）用于语义搜索。
3. 将每个事实写入 KV 存储，键为 (user_id, fact_type, entity) 用于 O(1) 查找。
4. 将每个事实写入图存储 (Mem0g) 作为类型化边用于关系查询。

在 `search(query, user_id)` 时：

1. 向量存储按嵌入余弦返回 top-k。
2. KV 存储返回基于查询派生的 (user_id, type, entity) 的直接命中。
3. 图存储返回从查询实体可达的子图。
4. 评分层融合三者。

### 融合评分

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **相关性 (Relevance)** — 向量余弦、KV 精确匹配、图路径权重。
- **重要性 (Importance)** — 写入时标记或学习得到（某些事实更重要：姓名、ID、策略）。
- **新鲜度 (Recency)** — 自上次写入或读取以来的指数衰减。

权重按产品调整。聊天 Agent 提高 `w_recency`；合规 Agent 提高 `w_importance`；检索 Agent 提高 `w_relevance`。

### Mem0g 与时间推理

Mem0g 添加了冲突检测器。当新事实与现有边矛盾时，现有边被标记为无效但不删除。时间查询（"用户三月的城市是哪里？"）遍历在指定时间有效的子图。

这是 Letta 的失效模式所泛化的合规级行为。

### 基准数据

Mem0 论文报告 (2025)：

- **LoCoMo** (长对话记忆): 91.6
- **LongMemEval** (长程情景记忆): 93.4
- **BEAM 1M** (1M token 记忆基准): 64.1

对比基线（全上下文 128k LLM、扁平向量存储、扁平 KV）全部落后 10+ 分。基准本身不能证明选择——运维形态才能——但这些数字表明融合设计不是舍入误差。

### 范围分类

Mem0 按范围拆分记忆：

- **用户记忆 (User memory)** — 跨会话持久化，键为 `user_id`。
- **会话记忆 (Session memory)** — 在一个线程内持久化。
- **Agent 记忆 (Agent memory)** — 每个 Agent 实例的状态。

每次写入选择一个范围。检索可以跨范围查询，每个范围有各自的权重。不加思考地混合范围会导致"助手把 Alice 的项目告诉了 Bob"这类事故。

### 这个模式哪里会出错

- **嵌入漂移。** 在前一百次查询中看起来正确的向量结果，随着语料增长而退化。定期对前 N 条最常用记录重新嵌入。
- **KV schema 蔓延。** `(user_id, type, entity)` 看起来简单，直到每个团队都添加自己的 `type`。每季度审计 type 集合。
- **图爆炸。** 一个嘈杂的提取器每次消息添加 50 条边。限制每次 `add` 调用的图写入；丢弃低置信度边。

## 构建

`code/main.py` 用标准库实现三存储模式：

- `VectorStore` — 朴素的 token 重叠相似度作为嵌入的替代。
- `KVStore` — 键为 `(user_id, fact_type, entity)` 的字典。
- `GraphStore` — 类型化边 (subject, relation, object, valid)。
- `Mem0` — 顶层门面，包含 `add()`、`search()`、融合评分和范围感知检索。
- 一个多用户、多会话对话的工作追踪。

运行：

```
python3 code/main.py
```

输出显示三条独立的召回路径以及融合后的 top-k。在 `main()` 顶部翻转评分权重，观察排名变化。

## 使用

- **Mem0 (Apache 2.0)** — 生产就绪。用 Postgres + Qdrant + Neo4j 自托管，或使用托管云。
- **Letta** — 三层 core/recall/archival；自带向量和图后端。
- **Zep** — 商业替代方案，带时间知识图谱和事实提取。
- **自定义构建** — 当你需要对提取器（合规）或融合权重（语音 Agent 中新鲜度占主导）的精确控制时。

## 交付

`outputs/skill-hybrid-memory.md` 生成一个三存储记忆脚手架，包含融合评分器、范围分类和时间失效。

## 练习

1. 将玩具向量相似度替换为真实的嵌入模型（sentence-transformers、Ollama、OpenAI embeddings）。在合成长对话上测量 recall@10。1000 次写入后排名会漂移吗？
2. 添加时间查询：`search(query, as_of=timestamp)`。仅返回在该时间或之前有效的记录。哪个存储需要最多的工作？
3. 实现冲突检测器：如果传入事实与图边矛盾，使旧边失效并记录两者。测试"用户住在柏林" → "用户住在里斯本"。
4. 将融合评分器移植为包含 `user_feedback` 维度（对检索到的记录点赞）。如何防止博弈（Agent 只返回它已经点赞的记录）？
5. 阅读 Mem0 文档 (`docs.mem0.ai`)。将玩具实现移植到 `mem0` 客户端调用。在相同的 20 个测试查询上比较检索质量。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 混合记忆 (Hybrid memory) | "向量加图加 KV" | 三个存储并行写入，检索时融合 |
| 事实提取 (Fact extraction) | "记忆摄入" | LLM 步骤，将文本分解为 (实体, 关系, 事实) 元组 |
| 融合评分 (Fusion scoring) | "相关性排名" | 相关性、重要性、新鲜度的加权和 |
| 范围 (Scope) | "记忆命名空间" | user / session / agent——决定谁能看到什么 |
| Mem0g | "记忆图" | 带时间有效性的类型化边，用于关系查询 |
| 时间失效 (Temporal invalidation) | "软删除" | 将矛盾的边标记为无效；永不删除 |
| 嵌入漂移 (Embedding drift) | "检索腐烂" | 向量质量随语料增长而退化；定期重新嵌入 |

## 扩展阅读

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — 原始论文
- [Mem0 docs](https://docs.mem0.ai/platform/overview) — 生产 API、SDK、托管云
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — 虚拟上下文的前身
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — 三层兄弟设计
