# RAG 的分块策略（Chunking Strategies）

> 分块配置对检索质量的影响与嵌入模型（Embedding Model）的选择同等重要（Vectara NAACL 2025）。分块做错了，再多的重排序（Reranking）也救不了你。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 14（信息检索 Information Retrieval），阶段 5 · 22（嵌入模型 Embedding Models）
**时间：** 约 60 分钟

## 问题

你把一份 50 页的合同放入 RAG 系统。用户问："终止条款是什么？"检索器返回了封面页。为什么？因为模型是在 512 个 token 的分块上训练的，而终止条款位于 20 页之后，被分页符分割，且没有局部关键词将其与查询关联起来。

解决方案不是"买一个更好的嵌入模型"。解决方案是分块（Chunking）。多大？重叠多少？在哪里分割？是否附带周围上下文？

2026 年 2 月的基准测试显示了令人惊讶的结果：

- Vectara 2026 年研究：递归 512 token 分块以 69% 对 54% 的准确率击败了语义分块（Semantic Chunking）。
- SPLADE + Mistral-8B 在 Natural Questions 上的表现：重叠（Overlap）没有提供任何可测量的收益。
- 上下文悬崖（Context Cliff）：响应质量在大约 2,500 个 token 的上下文处急剧下降。

那个"显而易见"的答案（语义分块、20% 重叠、1000 个 token）往往是错的。本课将建立对六种策略的直觉，并告诉你何时该用哪一种。

## 概念

![六种分块策略在同一段文本上的可视化](../assets/chunking.svg)

**固定分块（Fixed Chunking）。** 每 N 个字符或 token 分割一次。最简单的基线。会在句子中间断开。压缩效果好，但连贯性差。

**递归分块（Recursive）。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试按 `\n\n` 分割，然后按 `\n`，再按 `.`，最后按空格。逐级优雅回退。2026 年的默认选择。

**语义分块（Semantic）。** 对每个句子进行嵌入。计算相邻句子之间的余弦相似度（Cosine Similarity）。在相似度低于阈值的地方分割。保持主题连贯性。速度较慢；有时会产生 40 个 token 的微小片段，反而损害检索效果。

**句子分块（Sentence）。** 按句子边界分割。每个分块一个句子，或 N 个句子的滑动窗口。在约 5k token 以内，效果与语义分块相当，但成本仅为后者的一小部分。

**父文档（Parent-Document）。** 存储较小的子分块（Child Chunk）用于检索，同时存储较大的父分块（Parent Chunk）用于提供上下文。按子分块检索；返回父分块。优雅降级：即使子分块匹配不佳，仍能返回合理的父分块。

**延迟分块（Late Chunking，2024）。** 先在 token 级别对整个文档进行嵌入，然后将 token 嵌入池化为分块嵌入。保留跨分块的上下文信息。适用于长上下文嵌入器（BGE-M3、Jina v3）。计算成本更高。

**上下文检索（Contextual Retrieval，Anthropic，2024）。** 在每个分块前添加一段由 LLM 生成的摘要，描述该分块在文档中的位置（"本分块是终止条款的第 3.2 节……"）。在 Anthropic 自己的基准测试中，检索效果提升了 35-50%。索引成本高昂。

### 胜过所有默认设置的规则

将分块大小与查询类型匹配：

| 查询类型 | 分块大小 |
|------------|-----------|
| 事实型（"CEO 叫什么名字？"） | 256-512 token |
| 分析型 / 多跳推理（Multi-Hop） | 512-1024 token |
| 整节理解 | 1024-2048 token |

NVIDIA 2026 年基准测试。分块应足够大以包含答案及其局部上下文，同时足够小以使检索器的 Top-K 结果聚焦于答案而非上下文噪声。

## 动手构建

### 步骤 1：固定分块与递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步骤 2：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域数据上调优 `threshold`。太高 → 碎片化。太低 → 一个巨大的分块。

### 步骤 3：父文档

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞察：对父文档去重。多个子分块可能映射到同一个父文档；全部返回会浪费上下文。

### 步骤 4：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

对添加上下文后的分块进行索引。在查询时，检索会受益于额外的周围信号。

### 步骤 5：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终进行基准测试。对你的语料库而言，"最佳"策略可能与任何博客文章都不一致。

## 常见陷阱

- **仅在事实型查询上评估分块。** 多跳推理查询会揭示截然不同的优胜者。使用按查询类型分层（Query-Type-Stratified）的评估集。
- **语义分块没有设置最小大小。** 会产生 40 个 token 的碎片，损害检索效果。始终强制设置 `min_tokens`。
- **把重叠当作盲目跟风的迷信（Cargo Cult）。** 2026 年的研究发现，重叠通常没有任何收益，却使索引成本翻倍。要实测，不要假设。
- **没有最小/最大限制。** 5 个 token 或 5000 个 token 的分块都会破坏检索。必须钳制（Clamp）。
- **跨文档分块。** 绝不要让一个分块跨越两个文档。始终按文档分块，然后再合并。

## 使用指南

2026 年技术栈：

| 场景 | 策略 |
|-----------|----------|
| 首次构建，语料库未知 | 递归分块，512 token，无重叠 |
| 事实型问答（Factoid QA） | 递归分块，256-512 token |
| 分析型 / 多跳推理 | 递归分块，512-1024 token + 父文档 |
| 大量交叉引用（合同、论文） | 延迟分块或上下文检索 |
| 对话 / 会话语料库 | 轮次级分块 + 说话人元数据 |
| 短文本（推文、评论） | 一个文档 = 一个分块 |

从递归 512 开始。在 50 条查询的评估集上测量 Recall@5。然后在此基础上调优。

## 交付

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **简单。** 用固定分块（512, 0）、递归分块（512, 0）和递归分块（512, 100）对一份 20 页的文档进行分块。比较分块数量和边界质量。
2. **中等。** 在 5 份文档上构建一个 30 条查询的评估集。测量递归分块、语义分块和父文档的 Recall@5。哪种策略胜出？结果与博客文章一致吗？
3. **困难。** 实现上下文检索。测量相对于基线递归分块的 MRR 提升。报告索引成本（LLM 调用次数）与准确率收益的对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 分块（Chunk） | 文档的一个片段 | 被嵌入、索引和检索的子文档单元。 |
| 重叠（Overlap） | 安全余量 | 相邻分块之间共享的 N 个 token；在 2026 年的基准测试中通常毫无用处。 |
| 语义分块（Semantic Chunking） | 智能分块 | 在相邻句子嵌入相似度下降的地方进行分割。 |
| 父文档（Parent-Document） | 两级检索 | 检索小的子分块，返回较大的父分块。 |
| 延迟分块（Late Chunking） | 嵌入后再分块 | 在 token 级别对整个文档进行嵌入，然后池化为分块向量。 |
| 上下文检索（Contextual Retrieval） | Anthropic 的技巧 | 在索引前，将 LLM 生成的摘要添加到每个分块前面。 |
| 上下文悬崖（Context Cliff） | 2500 token 之墙 | 在 RAG 中，上下文 token 约 2.5k 时观察到的质量下降（2026 年 1 月）。 |

## 延伸阅读

- [Yepes 等人 / LangChain — 递归字符分割文档](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境中的默认方案。
- [Vectara（2024，NAACL 2025）。分块配置分析](https://arxiv.org/abs/2410.13070) — 分块的重要性与嵌入模型的选择相当。
- [Jina AI — 长上下文嵌入模型中的延迟分块（2024）](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 延迟分块论文。
- [Anthropic — 上下文检索](https://www.anthropic.com/news/contextual-retrieval) — 通过 LLM 生成的上下文前缀实现 35-50% 的检索提升。
- [NVIDIA 2026 分块大小基准测试 — Premai 摘要](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型选择分块大小。
