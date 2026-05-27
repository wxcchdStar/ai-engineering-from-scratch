# 实战项目 02 — 代码库 RAG（检索增强生成）

> 2026 年，Cursor 3 的代码库 RAG、Sourcegraph Cody、Google 的 Codey 和 Continue.dev 都遵循同一种形态：tree-sitter 分块、混合检索（密集 + BM25）、重排序、合成。代码库 RAG 是编程智能体的检索骨干——实战项目 01 和 10 都依赖它。本实战项目是构建一个独立的代码库 RAG 管道，在 100 个仓库上建立索引，并发布一份与 Cursor 3 和 Cody 的并排对比报告。

**类型：** 实战项目（Capstone）
**语言：** Python（管道）、TypeScript（VS Code 扩展）
**前置课程：** 第 5 阶段（NLP）、第 7 阶段（Transformers）、第 11 阶段（LLM 工程）、第 17 阶段（基础设施）
**涉及阶段：** P5 · P7 · P11 · P17
**时间：** 25 小时

## 问题

代码库 RAG 是 2026 年编程智能体的检索骨干。Cursor 3 的代码库索引为 Tab 补全和智能体模式提供支持。Sourcegraph Cody 为整个组织仓库建立索引。Google 的 Codey 将 Google 搜索质量的检索技术引入代码。Continue.dev 将可插拔的代码库 RAG 引入 VS Code。它们都遵循同一种形态：使用 tree-sitter 或 AST 解析器对代码进行分块，使用代码专用嵌入模型进行嵌入，混合检索（密集 + BM25），使用代码专用模型进行重排序，以及使用 LLM 进行合成。

困难之处不在于模型，而在于分块策略（按函数、按类、按文件？）、索引新鲜度（代码在索引更新之前就已更改）以及延迟预算（检索 + 重排序 + 合成必须在 500 毫秒内完成，才能用于自动补全）。本实战项目要求你构建整个管道，在 100 个仓库上建立索引，并发布一份与 Cursor 3 和 Cody 的并排对比报告。

## 概念

该管道有两个阶段。**索引阶段**：遍历仓库，使用 tree-sitter 将每个文件解析为 AST，按语义边界（函数、类、方法）分块，使用 Voyage-code-3 或 Qwen3-Coder-Embed 为每个块生成嵌入，并构建一个包含密集向量和 BM25 稀疏索引的混合索引。**查询阶段**：接收自然语言查询或代码片段，运行混合检索，使用 codebert-ranker 或 Qwen3-Coder-Reranker 进行重排序，并将排名前 k 的块输入合成 LLM。

评估在 CodeRAG-Bench（2026 年代码检索标准基准）和 SWE-bench Pro 检索任务上进行。指标包括：nDCG@10、MRR 以及检索到的块是否包含解决 issue 所需的确切代码行。

## 架构

```
仓库
      |
      v
tree-sitter AST 解析 + 语义分块（函数、类、方法）
      |
      v
Voyage-code-3 / Qwen3-Coder-Embed 嵌入
      |
      +------> 密集索引（Qdrant 或 pgvector）
      +------> 稀疏索引（Tantivy BM25）
      |
查询 ----+----> 混合检索（密集 + BM25，RRF 融合）
      |
      v
codebert-ranker / Qwen3-Coder-Reranker 重排序
      |
      v
合成（Claude Sonnet 4.7 或 GPT-5.4-Codex）
      |
      v
带引用的答案 + 源文件路径 + 行号
```

## 技术栈

- 分块：tree-sitter 语言包（Python、TypeScript、Go、Rust、Java）
- 嵌入：Voyage-code-3（托管）或 Qwen3-Coder-Embed（自托管）
- 密集索引：Qdrant 或 pgvector + pgvectorscale
- 稀疏索引：Tantivy BM25，按字段加权（函数名 > 文档字符串 > 主体）
- 重排序：codebert-ranker 或 Qwen3-Coder-Reranker
- 合成：Claude Sonnet 4.7 配合提示缓存
- 评估：CodeRAG-Bench、SWE-bench Pro 检索任务
- VS Code 扩展：用于查询和差异预览的侧边栏面板

## 构建步骤

1. **分块器。** 使用 tree-sitter 解析每种语言。按语义边界分块：函数、方法、类、接口。保留文档字符串和类型注解。每个块获得一个 `{repo, file_path, start_line, end_line, chunk_type}` 标识符。

2. **嵌入。** 使用 Voyage-code-3（每块 1024 维）或 Qwen3-Coder-Embed 对每个块进行嵌入。对于超过 8192 个 token 的块，按函数边界拆分。

3. **混合索引。** 密集向量存入 Qdrant。BM25 索引存入 Tantivy，按字段加权（函数名权重 3 倍，文档字符串权重 2 倍，主体权重 1 倍）。

4. **查询管道。** 接收自然语言或代码片段。并行运行密集检索和 BM25 检索。使用倒数排名融合（RRF）合并结果。将前 20 个结果发送给重排序器。将前 5 个结果发送给合成器。

5. **重排序。** codebert-ranker 或 Qwen3-Coder-Reranker 对前 20 个块进行评分。返回前 5 个。

6. **合成。** Claude Sonnet 4.7 配合提示缓存。系统提示 + 重排序后的上下文作为缓存前缀。用户问题作为未缓存后缀。

7. **新鲜度。** 文件保存时触发重新索引。使用文件哈希检测变更。仅重新嵌入已更改的块。

8. **VS Code 扩展。** 侧边栏面板：输入查询，查看带文件路径和行号的结果，点击跳转到文件。

9. **评估。** 在 CodeRAG-Bench 和 SWE-bench Pro 检索任务上运行。与 Cursor 3 和 Cody 在 20 个共享查询上进行并排对比。

## 使用方式

```
$ code-rag query "认证中间件在哪里验证 JWT token？"
[检索]   混合检索返回 20 个块，耗时 80 毫秒
[重排序] 前 5 个块，codebert-ranker，耗时 45 毫秒
[合成]   claude-sonnet-4.7，缓存命中率 82%，耗时 0.6 秒
答案：
  JWT 验证发生在 src/auth/middleware.py:34-67 的 `verify_token()` 中。
  该函数从 Authorization header 中提取 token，使用 HS256 解码，
  并根据数据库检查过期时间。
  引用：[src/auth/middleware.py:34-67, src/auth/jwt_utils.py:12-28]
```

## 交付标准

`outputs/skill-code-rag.md` 描述交付物。一个独立的代码库 RAG 管道，在 100 个仓库上建立索引，附有与 Cursor 3 和 Cody 的并排对比报告。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | CodeRAG-Bench nDCG@10 | 在标准基准上的检索质量 |
| 20 | 延迟 | 检索 + 重排序 + 合成的 p95 延迟 |
| 20 | 分块质量 | 语义边界召回率（块是否在函数/类边界处分割？） |
| 20 | 索引新鲜度 | 从文件保存到索引更新的时间 |
| 15 | VS Code 扩展 UX | 延迟、结果质量、跳转到文件功能 |
| **100** | | |

## 练习

1. 将嵌入模型从 Voyage-code-3 替换为 Qwen3-Coder-Embed。测量 nDCG@10 的差异和每次查询的成本。
2. 添加"按示例搜索"功能：粘贴一个代码片段，检索语义上相似的块。测量这与自然语言查询的差异。
3. 实现增量索引：仅重新索引已更改的文件。测量包含 10,000 个文件的仓库的索引时间。
4. 添加跨仓库搜索：在 10 个仓库上建立索引，运行一次查询，返回跨仓库的结果。按仓库对结果进行分组。
5. 测量分块策略的影响：按函数分块 vs 按文件分块 vs 固定 512 token 分块。报告 nDCG@10 的差异。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| 语义分块 | "AST 感知分割" | 按函数、类和方法边界分割代码，而非固定长度 |
| 混合检索 | "密集 + 稀疏" | 向量相似度搜索与 BM25 关键词搜索并行运行 |
| RRF | "倒数排名融合" | 一种合并多个排名列表的经典技巧 |
| 重排序 | "第二阶段评分" | 一个更精确（但更慢）的模型对检索到的候选进行重新评分 |
| CodeRAG-Bench | "代码检索基准" | 2026 年用于评估代码库 RAG 系统的标准基准 |
| 索引新鲜度 | "过时检测" | 从代码变更到索引反映该变更的时间 |
| nDCG@10 | "检索指标" | 归一化折损累计增益，衡量前 10 个结果的排名质量 |

## 扩展阅读

- [Cursor 3 代码库索引](https://docs.cursor.com) — 参考生产形态
- [Sourcegraph Cody](https://sourcegraph.com/cody) — 备选参考
- [Continue.dev](https://continue.dev) — 开源备选方案
- [Voyage-code-3](https://www.voyageai.com) — 代码专用嵌入模型
- [Qwen3-Coder-Embed](https://huggingface.co/Qwen) — 备选嵌入模型
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) — AST 解析框架
- [Tantivy](https://github.com/quickwit-oss/tantivy) — BM25 搜索引擎
- [CodeRAG-Bench](https://arxiv.org/abs/2406.14497) — 评估基准
