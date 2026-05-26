# OpenTelemetry GenAI 语义约定 (OpenTelemetry GenAI Semantic Conventions)

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了 Agent 遥测的标准 schema。Span 名称、属性和内容捕获规则在各供应商之间收敛，使 Agent 追踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中含义一致。

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 13 (LangGraph), Phase 14 · 24 (可观测性平台)
**时间：** 约 60 分钟

## 学习目标

- 列举 GenAI span 类别：model/client、agent、tool。
- 区分 `invoke_agent` 的 CLIENT 与 INTERNAL span 及其各自的适用场景。
- 列举顶层 GenAI 属性：provider name、request model、data-source ID。
- 解释内容捕获契约：opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用建议。

## 问题

每个供应商发明自己的 span 名称。运维团队最终构建每个框架的仪表板。OpenTelemetry 的 GenAI SIG 通过定义整个生态系统都瞄准的一个标准来解决这个问题。

## 概念

### Span 类别

1. **Model / client span。** 覆盖原始 LLM 调用。由提供商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **Agent span。** `create_agent`（Agent 构造时）和 `invoke_agent`（运行时）。
3. **Tool span。** 每个工具调用一个；通过父子关系连接到 Agent span。

### Agent span 命名

- Span 名称：`invoke_agent {gen_ai.agent.name}`（如果命名）；回退到 `invoke_agent`。
- Span 类型：
  - **CLIENT** — 用于远程 Agent 服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内 Agent 框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 解析后的模型（可能因路由而与请求不同）。
- `gen_ai.agent.name` — Agent 标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于 RAG：咨询了哪个语料库或存储。

存在针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 的技术特定约定。

### 内容捕获

默认规则：instrumentation 默认不应捕获输入/输出。捕获通过以下方式 opt-in：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、你的日志存储），在 span 上记录引用（指针 ID，而非文本）。这是第 27 课内容投毒防御接入可观测性。

### 稳定性

截至 2026 年 3 月，大多数约定是实验性的。通过以下方式选择加入稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 将 GenAI 属性原生映射到其 LLM Observability schema。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这个模式哪里会出错

- **在 span 中捕获完整 prompt。** PII、密钥、客户数据在运维可读的追踪中。存储在外部。
- **没有 `gen_ai.provider.name`。** 多提供商仪表板在缺少归属时崩溃。
- **没有父链接的 span。** 孤立的工具 span。始终传播上下文。
- **不设置稳定性 opt-in。** 你的属性可能在后端升级时被重命名。

## 构建

`code/main.py` 实现一个匹配 GenAI 约定的标准库 span 发出器：

- `Span` 带 GenAI 属性 schema。
- `Tracer` 带 `start_span`、嵌套上下文。
- 一个脚本化 Agent 运行，发出：`create_agent`、`invoke_agent` (INTERNAL)、每个工具的 span、LLM 调用的 `chat` span。
- 一个内容捕获模式，将 prompt 存储在外部并在 span 上记录 ID。

运行：

```
python3 code/main.py
```

输出：一个包含所有必需 GenAI 属性的 span 树，以及一个显示 opt-in 内容引用的"外部存储"。

## 使用

- **Datadog LLM Observability** (v1.37+) 原生映射属性。
- **Langfuse / Phoenix / Opik** (第 24 课) — 自动 instrumentation 生态系统。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel 追踪；从 GenAI 属性构建仪表板。
- **自托管** — 运行带 GenAI 处理器的 OTel Collector。

## 交付

`outputs/skill-otel-genai.md` 将 OTel GenAI span 接入现有 Agent，包含内容捕获默认值和外部引用存储。

## 练习

1. 用 `invoke_agent` (INTERNAL) + 每个工具的 span 对你的第 01 课 ReAct 循环进行 instrumentation。发送到 Jaeger 实例。
2. 添加"仅引用"模式的内容捕获：prompt 到 SQLite，span 属性仅携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其接入你的第 09 课 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证你的属性不会被 collector 重命名。
5. 构建一个仪表板：仅从 GenAI 属性中找出"哪些工具错误与哪些模型相关"。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| GenAI SIG | "OpenTelemetry GenAI 组" | 定义 schema 的 OTel 工作组 |
| invoke_agent | "Agent span" | 表示 Agent 运行的 span 名称 |
| CLIENT span | "远程调用" | 对远程 Agent 服务调用的 span |
| INTERNAL span | "进程内" | 进程内 Agent 运行的 span |
| gen_ai.provider.name | "提供商" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 来源" | 检索命中了哪个语料库/存储 |
| 内容捕获 (Content capture) | "Prompt 日志" | Opt-in 的消息捕获；生产环境中存储在外部 |
| 稳定性 opt-in (Stability opt-in) | "预览模式" | 固定实验性约定的环境变量 |

## 扩展阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认 GenAI span
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 追踪上下文传播
