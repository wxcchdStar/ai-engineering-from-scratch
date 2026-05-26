# Agent 可观测性：Langfuse、Phoenix、Opik (Agent Observability: Langfuse, Phoenix, Opik)

> 三个开源 Agent 可观测性平台主导 2026 年。Langfuse (MIT) — 6M+ 月安装量，追踪 + prompt 管理 + 评估 + 会话重放。Arize Phoenix (Elastic 2.0) — 深度 Agent 特定评估、RAG 相关性、OpenInference 自动 instrumentation。Comet Opik (Apache 2.0) — 自动 prompt 优化、护栏、LLM 评判幻觉检测。

**类型：** 学习 (Learn)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 23 (OTel GenAI)
**时间：** 约 45 分钟

## 学习目标

- 列举三个顶级开源 Agent 可观测性平台及其许可证。
- 区分每个平台的最强项：Langfuse（prompt 管理 + 会话）、Phoenix（RAG + 自动 instrumentation）、Opik（优化 + 护栏）。
- 解释为什么到 2026 年 89% 的组织报告已部署 Agent 可观测性。
- 用标准库实现一个追踪到仪表板的流水线，带 LLM 评判评估。

## 问题

OTel GenAI (第 23 课) 给你 schema。你仍然需要摄入 span、运行评估、存储 prompt 版本并发现回归的平台。三个竞争者各自强调生命周期的不同部分。

## 概念

### Langfuse (MIT)

- 6M+ SDK 月安装量，19k+ GitHub stars。
- 功能：追踪、带版本控制 + 游乐场的 prompt 管理、评估（LLM 作为评判者、用户反馈、自定义）、会话重放。
- 2025 年 6 月：以前商业化的模块（LLM 作为评判者、标注队列、prompt 实验、Playground）在 MIT 下开源。
- 最强项：端到端可观测性，带紧密的 prompt 管理循环。

### Arize Phoenix (Elastic License 2.0)

- 更深的 Agent 特定评估：追踪聚类、异常检测、RAG 检索相关性。
- 原生 OpenInference 自动 instrumentation。
- 与托管的 Arize AX 配对用于生产。
- 无 prompt 版本控制——定位为漂移/行为回归工具，与更广泛的平台配合使用。
- 最强项：RAG 相关性、行为漂移、异常检测。

### Comet Opik (Apache 2.0)

- 通过 A/B 实验的自动 prompt 优化。
- 护栏（PII 脱敏、主题约束）。
- LLM 评判幻觉检测。
- Comet 自身测量的基准：Opik 日志 + 评估 23.44s vs Langfuse 327.15s（约 14 倍差距）——将供应商基准视为方向性的。
- 最强项：优化循环、自动实验、护栏执行。

### 行业数据

据 Maxim（2026 年现场分析）：89% 的组织已部署 Agent 可观测性；质量问题是首要生产障碍（32% 的受访者提到）。

### 选择

| 需求 | 选择 |
|------|------|
| 带 prompt 管理的一体化 | Langfuse |
| 深度 RAG 评估 + 漂移 | Phoenix |
| 自动优化 + 护栏 | Opik |
| 开放许可，非 ELv2 | Langfuse (MIT) 或 Opik (Apache 2.0) |
| Datadog / New Relic 集成 | 任意——它们都导出 OTel |

### 这个模式哪里会出错

- **没有评估策略。** 没有评估的追踪只是昂贵的日志。
- **自建 LLM 评判者没有锚定。** CRITIC 模式 (第 05 课) 适用——评判者需要外部工具进行事实验证。
- **Prompt 版本未与追踪关联。** 当生产回归时，你无法二分查找导致回归的 prompt。

## 构建

`code/main.py` 实现一个标准库追踪收集器 + LLM 评判评估器：

- 摄入 GenAI 形态的 span。
- 按会话分组，标记失败的运行（护栏触发、低置信度评估）。
- 一个脚本化 LLM 评判者，按评分标准对 Agent 响应打分。
- 一个类似仪表板的摘要：失败率、首要失败原因、评估分数分布。

运行：

```
python3 code/main.py
```

输出：每个会话的评估分数和失败分类，匹配 Langfuse/Phoenix/Opik 会显示的内容。

## 使用

- **Langfuse** 自托管或云；通过 OTel 或其 SDK 接入。
- **Arize Phoenix** 自托管；自动 instrumentation OpenInference。
- **Comet Opik** 自托管或云；自动优化循环。
- **Datadog LLM Observability** 用于已经运行 Datadog 的混合运维+ML 团队。

## 交付

`outputs/skill-obs-platform-wiring.md` 选择一个平台并将追踪 + 评估 + prompt 版本接入现有 Agent。

## 练习

1. 将一周的 OTel 追踪导出到 Langfuse 云（免费层）。哪些会话失败了？为什么？
2. 为你的领域编写 LLM 评判评分标准（事实正确性、语气、范围遵守）。在 50 条追踪上测试。
3. 比较 Langfuse prompt 版本控制与 Phoenix 的追踪聚类。哪个更快告诉你什么坏了？
4. 阅读 Opik 的护栏文档。将 PII 脱敏护栏接入你的一个 Agent 运行。
5. 在你的语料库上对三者进行基准测试。忽略供应商发布的数字；测量你自己的。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 追踪 (Tracing) | "Span 收集器" | 摄入 OTel / SDK span；按会话索引 |
| Prompt 管理 (Prompt management) | "Prompt CMS" | 与追踪关联的版本化 prompt |
| LLM 作为评判者 (LLM-as-judge) | "自动评估" | 单独的 LLM 按评分标准对 Agent 输出打分 |
| 会话重放 (Session replay) | "追踪回放" | 逐步回放过去的运行以进行调试 |
| RAG 相关性 (RAG relevancy) | "检索质量" | 检索到的上下文是否匹配查询 |
| 追踪聚类 (Trace clustering) | "行为分组" | 将相似运行聚类以进行漂移检测 |
| 护栏执行 (Guardrail enforcement) | "日志时策略" | 对记录内容的 PII/毒性/范围检查 |

## 扩展阅读

- [Langfuse docs](https://langfuse.com/) — 追踪、评估、prompt 管理
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — 自动 instrumentation、漂移
- [Comet Opik](https://www.comet.com/site/products/opik/) — 优化 + 护栏
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三者都消费的 schema
