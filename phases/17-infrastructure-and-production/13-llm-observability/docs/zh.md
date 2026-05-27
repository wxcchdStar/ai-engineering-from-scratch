# LLM 可观测性 — OpenTelemetry、LangSmith 与分布式追踪

> LLM 可观测性（Observability）在 2026 年建立在三个支柱之上：**追踪（Tracing）**、**日志（Logging）** 和 **指标（Metrics）**。**分布式追踪（Distributed tracing）** 通过 LLM 调用链传播 trace ID — 从用户请求 → 网关 → 模型 → 向量数据库 → 响应。**OpenTelemetry（OTel）** 是标准 — 它定义了 GenAI 语义约定（semantic conventions）：`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`、`gen_ai.response.id` 等。**LangSmith**（LangChain 旗下）是领先的商业 LLM 可观测性平台 — 它捕获完整的提示词/响应对、token 使用量、延迟和用户反馈。**提示词/响应日志记录（Prompt/response logging）** 必须平衡调试需求与隐私 — 记录提示词哈希值用于审计，仅在明确选择加入的情况下记录原始提示词。**Token 使用量追踪（Token usage tracking）** 按用户、租户和功能进行归因（第 17 阶段 · 27 的 FinOps）。2026 年的标准堆栈：OTel SDK → OTel Collector → 可观测性后端（Datadog、Grafana、LangSmith）。关键指标：TTFT（首 token 时间，Time To First Token）P50/P95/P99、TPOT（每输出 token 时间，Time Per Output Token）、有效吞吐量（goodput）、请求成功率、token 使用量/成本。

**类型：** 学习
**语言：** Python（标准库，OTel 追踪模拟器）
**前置要求：** 第 17 阶段 · 08（推理指标）
**时间：** 约 60 分钟

## 学习目标

- 列举可观测性的三个支柱（追踪、日志、指标）并解释每个支柱在 LLM 上下文中的作用。
- 说出 OpenTelemetry GenAI 语义约定中的至少五个属性。
- 解释为什么提示词/响应日志记录需要隐私控制（哈希 vs 原始文本）。
- 绘制 2026 年标准可观测性堆栈：OTel SDK → Collector → 后端。

## 问题

用户报告"聊天机器人给出了奇怪的回答"。你检查了服务器日志 — 200 OK，延迟正常。你不知道：
- 使用了哪个模型版本。
- 提示词是什么（它经过了检索增强和模板化）。
- 响应是否被 guardrail 修改过。
- 这个请求是否触发了对向量数据库的调用。

没有分布式追踪，调试 LLM 应用就像在没有堆栈追踪的情况下调试代码。你看到输出，但看不到它是如何产生的。

## 概念

### 三个支柱

**追踪（Tracing）**：跨服务边界跟踪请求。一个 trace 由多个 span 组成 — 每个 span 代表一个工作单元（LLM 调用、向量搜索、guardrail 检查）。Span 携带属性（模型名称、token 数量、延迟）和事件（提示词发送、第一个 token、响应完成）。

**日志（Logging）**：离散事件记录。"调用 OpenAI 失败，429，重试 2/3。" 日志是时间点记录；追踪是端到端的旅程。

**指标（Metrics）**：聚合的时间序列数据。TTFT P99 在 5 分钟窗口内。Token 使用量/小时。错误率。指标驱动仪表盘和告警；追踪和日志驱动调试。

### OpenTelemetry GenAI 语义约定

OTel 1.28+ 定义了 LLM 调用的标准属性：

| 属性 | 描述 | 示例 |
|------|------|------|
| `gen_ai.request.model` | 模型名称 | `gpt-5`, `claude-sonnet-4-20250514` |
| `gen_ai.usage.input_tokens` | 输入 token 数 | `1800` |
| `gen_ai.usage.output_tokens` | 输出 token 数 | `150` |
| `gen_ai.response.id` | 响应 ID | `chatcmpl-abc123` |
| `gen_ai.system` | 供应商 | `openai`, `anthropic` |
| `gen_ai.request.temperature` | 采样温度 | `0.7` |
| `gen_ai.request.max_tokens` | 最大输出 token | `4096` |

这些属性确保跨供应商和框架的可观测性一致。

### 提示词/响应日志记录

**问题**：提示词可能包含 PII（个人身份信息，Personally Identifiable Information）、PHI（受保护的健康信息，Protected Health Information）、商业秘密。将原始提示词记录到可观测性平台会带来隐私风险。

**解决方案**：默认记录提示词哈希值（SHA-256）。仅在明确选择加入的情况下记录原始提示词（用于调试）。在记录之前使用 PII 脱敏（第 17 阶段 · 25）。将原始提示词存储在经过访问控制的单独日志存储中。

### Token 使用量追踪

每次 LLM 调用发出一个包含以下内容的指标事件：
- `user_id`：哪个用户。
- `tenant_id`：哪个客户。
- `task_id`：哪个功能。
- `input_tokens`、`output_tokens`：token 数量。
- `model`：哪个模型。
- `cost_usd`：成本。

聚合这些数据可以驱动 FinOps（第 17 阶段 · 27）— 按租户计费、按功能进行成本归因、预算告警。

### 2026 年标准堆栈

```
应用代码（OTel SDK）
    │
    ▼
OTel Collector（接收、处理、导出）
    │
    ├──► Datadog / Grafana（指标 + 仪表盘）
    ├──► LangSmith（LLM 追踪 + 提示词日志）
    └──► S3 / Snowflake（长期日志存储）
```

**OTel SDK**：在应用代码中进行仪器化。Python：`opentelemetry-instrumentation-openai`、`opentelemetry-instrumentation-langchain`。

**OTel Collector**：接收、处理（批处理、过滤、脱敏）并导出到后端。

**后端**：Datadog（指标 + APM）、Grafana（指标 + 仪表盘）、LangSmith（LLM 特定追踪）、S3（日志归档）。

### 你应该记住的数字

- 三个支柱：追踪、日志、指标。
- OTel GenAI 属性：`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens` 等。
- 提示词日志记录：默认哈希值，选择加入原始文本。
- Token 归因维度：user_id、tenant_id、task_id。
- 标准堆栈：OTel SDK → Collector → Datadog/Grafana/LangSmith。

## 动手实践

`code/main.py` 模拟一个包含多个 span 的分布式追踪（网关 → LLM → 向量搜索 → guardrail）。输出 OTel 格式的 trace。

## 交付物

本课产出 `outputs/skill-observability-stack.md`。给定基础设施和合规要求，设计可观测性堆栈。

## 练习

1. 运行 `code/main.py`。追踪 LLM 调用中的 span 数量。每个 span 代表什么？
2. 为 LLM 调用设计 OTel 属性：包括模型、token、延迟和用户上下文。
3. 提示词日志记录：为医疗应用设计隐私保护策略。
4. 将 token 使用量归因到租户：设计发出正确指标事件的代码。
5. 比较 LangSmith 与自建 OTel + Grafana 堆栈。在什么场景下各自胜出？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 可观测性（Observability） | "了解系统内部" | 追踪 + 日志 + 指标 |
| 分布式追踪（Distributed tracing） | "请求旅程" | 跨服务的端到端请求追踪 |
| OpenTelemetry（OTel） | "可观测性标准" | CNCF 的可观测性框架 |
| 语义约定（Semantic conventions） | "标准属性" | 跨供应商的一致属性名称 |
| Span | "工作单元" | Trace 中的单个操作 |
| LangSmith | "LLM 可观测性" | LangChain 的商业 LLM 追踪平台 |
| 提示词哈希（Prompt hash） | "隐私保护日志" | SHA-256 哈希值，而非原始提示词 |
| Token 归因（Token attribution） | "谁用了什么" | 按用户/租户/功能追踪 token 使用量 |
