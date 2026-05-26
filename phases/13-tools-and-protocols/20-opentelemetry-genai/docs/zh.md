# OpenTelemetry GenAI ——端到端追踪工具调用

> 一个 agent 调用了五个工具、三个 MCP 服务器和两个子 agent。你需要一个跨所有环节的 trace。OpenTelemetry GenAI 语义约定（v1.37 及以上版本的稳定属性）是 2026 年的标准，由 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课列出必需属性，走读 span 层级 (agent → LLM → tool)，并提供一个可接入任何 OTel exporter 的标准库 span emitter。

**类型：** 构建
**语言：** Python（标准库，OTel span emitter）
**前置课程：** Phase 13 · 07（MCP 服务器），Phase 13 · 08（MCP 客户端）
**时长：** 约 75 分钟

## 学习目标

- 列出 LLM span 和工具执行 span 所需的 OTel GenAI 属性。
- 构建涵盖 agent 循环、LLM 调用、工具调用和 MCP 客户端分发的 trace 层级。
- 决定哪些内容需要捕获（opt-in）与默认脱敏。
- 将 span 发送到本地收集器（Jaeger、Langfuse）而不重写工具代码。

## 问题背景

2026 年 2 月的一次调试：用户报告"我的 agent 有时 30 秒才响应，有时 3 秒"。没有 trace。日志显示 LLM 调用，但不显示工具分发，不显示 MCP 服务器往返，不显示子 agent。你只能猜。最终你发现：一个 MCP 服务器偶尔在冷启动时挂起。

没有端到端追踪，你无法定位。OTel GenAI 解决了这个问题。

约定在 2025-2026 年在 OpenTelemetry semantic-conventions 组中沉淀成型。它们定义了稳定的属性名称，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的 span。一次埋点；发送到任何后端。

## 核心概念

### Span 层级

```
agent.invoke_agent  (顶层，INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整个结构嵌套在同一个 trace id 下。Span id 链接父子关系。

### 必需属性

根据 2025-2026 semconv：

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际提供服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 提供商的响应 ID，用于关联。

对于工具 span：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 具体的调用 ID。
- `gen_ai.tool.description` — 工具描述（可选）。

对于 agent span：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span 类型

- `SpanKind.CLIENT` 用于跨进程边界的调用（LLM 提供商、MCP 服务器）。
- `SpanKind.INTERNAL` 用于 agent 自身循环步骤和工具执行。

### 按需内容捕获

默认情况下，span 携带指标和计时——不包含提示词或补全结果。大负载和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定的内容捕获环境变量以包含内容。在生产环境启用前请仔细审查。

### Span 上的事件

令牌级事件可作为 span event 添加：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件在 span 内按时间排序，用于详细回放。

### Exporter

OTel span 可导出到：

- **Jaeger / Tempo。** 开源，本地部署。
- **Langfuse。** 专门面向 LLM 可观测性；可视化令牌使用情况。
- **Arize Phoenix。** 评估 + 追踪一体化。
- **Datadog。** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列式存储；查询友好。

全部使用 OTLP 有线格式。你的代码无需关心。

### 跨 MCP 传播

当 MCP 客户端调用服务器时，将 W3C traceparent 头注入请求。Streamable HTTP 支持标准头。Stdio 本身不携带 HTTP 头；规范的 2026 路线图讨论了在 JSON-RPC 调用上添加 `_meta.traceparent` 字段。

在那之前：手动在每个请求的 `_meta` 中包含 traceparent。服务器记录 trace id。

### 指标

与 span 平行，GenAI semconv 定义了指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

用于不需要逐调用细节的仪表盘。

### AgentOps 层

AgentOps（2024 年创立）专注于 GenAI 可观测性。它包装了流行框架（LangGraph、Pydantic AI、CrewAI）以自动发出 OTel span。如果你的技术栈使用受支持的框架，这将很有用；否则使用手动埋点。

## 动手实践

`code/main.py` 向 stdout 发出 OTel 形态的 span（类似 OTLP-JSON 格式），模拟一个调用 LLM、分发两个工具并进行一次 MCP 往返的 agent。无真实 exporter——课程聚焦于 span 形态和属性集。将输出粘贴到与 OTLP 兼容的查看器中，或直接阅读。

关注点：

- Trace id 在所有 span 中共享。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；一个场景通过环境变量开启。

## 交付产物

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定 agent 代码库，该技能产出埋点计划：在何处添加 span、填充哪些属性、以及目标使用哪些 exporter。

## 练习

1. 运行 `code/main.py`。统计 span 数量并识别哪个是 CLIENT 与 INTERNAL。

2. 开启内容捕获（环境变量），确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意其对 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration` 并作为每次调用的直方图样本发出。

4. 从父 agent span 将 traceparent 传播到 MCP 请求的 `_meta.traceparent` 字段中。验证 MCP 服务器是否会看到相同的 trace id。

5. 阅读 OTel GenAI semconv 规范。找出 semconv 中列出但本课代码未发出的一个属性。添加它。

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| OTel | "OpenTelemetry" | trace、metrics、logs 的开放标准 |
| GenAI semconv | "GenAI 语义约定" | LLM / tool / agent span 的稳定属性名称 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀 |
| Span | "计时操作" | 具有开始、结束和属性的工作单元 |
| Trace | "跨 span 的祖先树" | 共享一个 trace id 的 span 树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 关于 span 方向的提示 |
| OTLP | "OpenTelemetry Line Protocol" | exporter 的有线格式 |
| Opt-in content | "提示词/补全捕获" | 默认关闭；由环境变量启用 |
| traceparent | "W3C header" | 跨服务传播 trace 上下文 |
| Exporter | "后端特定的发送组件" | 将 span 发送到 Jaeger / Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI span、metric 和 event 的标准约定
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行 span 属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — agent 级别 `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 托管的权威来源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产环境集成演练