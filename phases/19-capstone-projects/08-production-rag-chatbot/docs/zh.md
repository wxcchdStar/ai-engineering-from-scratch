# 实战项目 08 — 面向受监管垂直领域的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都运行着相同的生产形态。使用 docling 或 Unstructured 以及 ColPali 处理视觉内容进行数据摄入。混合搜索。使用 bge-reranker-v2-gemma 进行重排序。使用 Claude Sonnet 4.7 配合提示缓存（60-80% 命中率）进行合成。使用 Llama Guard 4 和 NeMo Guardrails 进行防护。使用 Langfuse 和 Phoenix 进行监控。使用 RAGAS 在 200 个问题的黄金集上进行评分。在受监管领域（法律、临床、保险）构建一个，通过黄金集、红队测试和漂移仪表盘即为完成本实战项目。

**类型：** 实战项目（Capstone）
**语言：** Python（管道 + API）、TypeScript（聊天 UI）
**前置课程：** 第 5 阶段（NLP）、第 7 阶段（Transformers）、第 11 阶段（LLM 工程）、第 12 阶段（多模态）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及阶段：** P5 · P7 · P11 · P12 · P17 · P18
**时间：** 30 小时

## 问题

受监管领域的 RAG（法律合同、临床试验方案、保险政策）是 2026 年最常交付的生产形态，因为 ROI 显而易见且风险具体。Harvey（Allen & Overy）为法律领域构建了它。Mendable 提供开发者文档版本。Glean 覆盖企业搜索。其模式是：高保真摄入、混合检索加重排序、带引用强制和提示缓存的合成、多层安全防护，以及持续漂移监控。

困难之处不在于模型。而在于辖区感知合规（HIPAA、GDPR、SOC2）、引用级可审计性、成本控制（当命中率高时，提示缓存可节省 60-90% 的成本）、通过 RAGAS 忠实度进行幻觉检测，以及当源文档更新而索引未及时跟进时的漂移检测。本实战项目要求你在 200 个问题的黄金集上交付所有内容，并同时配备红队测试套件。

## 概念

该管道有两个方面。**摄入**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；块获得摘要、标签和基于角色的访问标签。向量存入 pgvector + pgvectorscale（5000 万向量以下）或 Qdrant Cloud；稀疏 BM25 并行运行。**对话**：LangGraph 处理记忆和多轮对话；每个查询运行混合检索，使用 bge-reranker-v2-gemma-2b 重排序，使用 Claude Sonnet 4.7（提示缓存）合成，通过 Llama Guard 4 和 NeMo Guardrails 传递输出，并发出引用锚定的响应。

评估栈有四个层次。**黄金集**（200 个带引用的标注 Q/A）用于正确性。**红队**（越狱、PII 提取尝试、领域外问题）用于安全。**RAGAS** 用于每轮自动评估忠实度/答案相关性/上下文精确度。**漂移仪表盘**（Arize Phoenix）每周监控检索质量和幻觉评分。

提示缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示 + 检索到的上下文。在 60-80% 的命中率下，每次查询成本下降 3-5 倍。管道必须设计为稳定的前缀（系统提示 + 重排序后的上下文优先）以实现高缓存命中率。

## 架构

```
文档（合同、方案、政策）
      |
      v
docling / Unstructured 解析 + ColPali 处理视觉内容
      |
      v
块 + 摘要 + 角色标签 + 辖区标签
      |
      v
pgvector + pgvectorscale  +  BM25（Tantivy）
      |
查询 + 角色 + 辖区
      |
      v
LangGraph 对话智能体
   +--- 检索（混合）
   +--- 按角色 + 辖区过滤
   +--- 重排序（bge-reranker-v2-gemma-2b 或 Voyage rerank-2）
   +--- 合成（Claude Sonnet 4.7，提示缓存）
   +--- 防护（Llama Guard 4 + NeMo Guardrails + Presidio 输出 PII 脱敏）
   +--- 引用 + 返回
      |
      v
评估：
  RAGAS 忠实度 / 答案相关性 / 上下文精确度（在线）
  Langfuse 标注队列（采样）
  Arize Phoenix 漂移（每周）
  红队套件（发布前）
```

## 技术栈

- 摄入：Unstructured.io 或 docling 用于结构化文档；ColPali 用于视觉丰富的 PDF
- 向量数据库：pgvector + pgvectorscale（5000 万向量以下）；否则使用 Qdrant Cloud
- 稀疏检索：Tantivy BM25，按字段加权
- 编排：LlamaIndex Workflows（摄入）+ LangGraph（对话）
- 重排序器：bge-reranker-v2-gemma-2b 自托管或 Voyage rerank-2 托管
- LLM：Claude Sonnet 4.7 配合提示缓存；备选 Llama 3.3 70B 自托管
- 评估：RAGAS 0.2 在线，DeepEval 用于幻觉和越狱套件
- 可观测性：Langfuse 自托管配合标注队列；Arize Phoenix 用于漂移
- 防护栏：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 脱敏
- 合规：块上的基于角色的访问标签；GDPR/HIPAA 的辖区标签

## 构建步骤

1. **摄入。** 使用 Unstructured 或 docling 解析你的语料库（严肃构建需要 1000-10000 份文档）。对于扫描/视觉密集的页面，路由到 ColPali。生成带有摘要、角色标签、辖区标签的块。

2. **索引。** 密集嵌入（Voyage-3 或 Nomic-embed-v2）存入 pgvector + pgvectorscale。通过 Tantivy 建立 BM25 辅助索引。角色和辖区过滤器作为负载。

3. **混合检索。** 首先按角色+辖区过滤；然后并行密集 + BM25；使用倒数排名融合合并；前 20 个送入重排序器；前 5 个送入合成器。

4. **使用提示缓存合成。** 系统提示 + 静态策略在缓存头中；重排序后的上下文作为缓存扩展；用户问题作为未缓存后缀。稳态下目标 60-80% 缓存命中率。

5. **防护栏。** Llama Guard 4 用于输入；NeMo Guardrails 规则阻止领域外问题或策略禁止的主题；Presidio 脱敏输出中的意外 PII；引用强制后过滤。

6. **黄金集。** 由领域专家标注的 200 个 Q/A 对，包含（答案、引用）。对智能体进行精确引用匹配、答案正确性、忠实度（RAGAS）评分。

7. **红队。** 50 个对抗性提示：越狱（PAIR、TAP）、PII 外泄尝试、领域外、跨辖区泄露。使用通过/失败和严重性进行评分。

8. **漂移仪表盘。** Arize Phoenix 每周跟踪检索质量（nDCG、引用忠实度）。在下降 5% 时告警。

9. **成本报告。** Langfuse：提示缓存命中率、每次查询的 token 数、按阶段分解的 $/查询。

## 使用方式

```
$ chat --role=analyst --jurisdiction=GDPR
> 根据我们的合同，欧盟用户档案的数据保留义务是什么？
[检索]   混合检索前 20 个，过滤到 GDPR + 分析师角色
[重排序] 保留前 5 个
[合成]   claude-sonnet-4.7，缓存命中率 74%，0.8 秒
答案：
  合同（第 12.4 条，2024 年 3 月 11 日主服务协议）规定，
  根据 GDPR 第 17 条，欧盟用户档案必须在终止后 30 天内删除。
  DPA 修正案（DPA-v2.1，第 5 条）将"受限"类别数据的期限延长至 14 天。
  引用：[MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付标准

`outputs/skill-production-rag.md` 描述交付物。一个部署了合规标签、通过了评分标准、并通过实时漂移监控进行观测的受监管领域聊天机器人。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS 忠实度 + 答案相关性 | 在黄金集（200 个 Q/A）上的在线评分 |
| 20 | 引用正确性 | 具有可验证源锚点的答案比例 |
| 20 | 防护栏覆盖率 | Llama Guard 4 通过率 + 越狱套件结果 |
| 20 | 成本/延迟工程 | 提示缓存命中率、p95 延迟、$/查询 |
| 15 | 漂移监控仪表盘 | Phoenix 实时仪表盘，包含每周检索质量趋势 |
| **100** | | |

## 练习

1. 在不同辖区（例如，HIPAA 与 GDPR 并列）下构建第二个语料库切片。在 20 个问题的跨辖区探测上展示角色+辖区过滤防止跨泄露。
2. 在一周的生产流量中测量提示缓存命中率。识别哪些查询破坏了缓存前缀。重构。
3. 使用 10k token 摘要缓冲区添加多轮记忆。测量随着对话增长，忠实度是否下降。
4. 将 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 $/查询和忠实度差异。
5. 添加"不确定"模式：如果重排序后的最高分低于阈值，智能体说"我没有可信的引用"而不是回答。测量误信心的减少。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| 提示缓存 | "缓存的系统 + 上下文" | Claude/OpenAI 功能：缓存的前缀 token 在命中时折扣 60-90% |
| RAGAS | "RAG 评估器" | 自动评分忠实度、答案相关性、上下文精确度 |
| 黄金集 | "标注评估集" | 200+ 个专家标注的 Q/A，包含引用；即真实标准 |
| 辖区标签 | "合规标签" | 附加到块的 GDPR/HIPAA/SOC2 范围；由检索过滤器强制执行 |
| 引用忠实度 | "有依据的答案率" | 由可检索源片段支持的声明比例 |
| 漂移 | "检索质量衰减" | nDCG 或引用分数的每周变化；告警阈值 5% |
| 红队 | "对抗性评估" | 发布前的越狱、PII 提取、领域外探测 |

## 扩展阅读

- [Harvey AI](https://www.harvey.ai) — 参考法律生产栈
- [Glean 企业搜索](https://www.glean.com) — 参考企业级 RAG
- [Mendable 文档](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄入
- [Anthropic 提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 文档](https://docs.ragas.io/) — 规范 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 参考漂移可观测性
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略规则框架
