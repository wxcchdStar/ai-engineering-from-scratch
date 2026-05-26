# OpenAI Agents SDK：交接、护栏、追踪 (OpenAI Agents SDK: Handoffs, Guardrails, Tracing)

> OpenAI Agents SDK 是构建在 Responses API 之上的轻量级多 Agent 框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。交接是名为 `transfer_to_<agent>` 的工具。护栏在输入或输出时触发。追踪默认开启。

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 01 (Agent 循环), Phase 14 · 06 (工具使用)
**时间：** 约 75 分钟

## 学习目标

- 列举 OpenAI Agents SDK 的五个原语。
- 解释交接：为什么它们被建模为工具，模型看到什么名称形状，以及上下文如何传递。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` vs 阻塞模式。
- 用标准库实现一个带交接 + 护栏 + span 风格追踪的运行时。

## 问题

不能干净委派的 Agent 最终会把所有东西塞进一个 prompt。没有护栏的 Agent 会发布 PII、违反策略的输出或无限循环。OpenAI 的 SDK 将三个使多 Agent 工作可驾驭的原语固化为代码。

## 概念

### 五个原语

1. **Agent。** LLM + instructions + tools + handoffs。
2. **Handoff。** 委派给另一个 Agent。对模型表示为名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail。** 对输入（仅第一个 Agent）、输出（仅最后一个 Agent）或工具调用（每个函数工具）的验证。
4. **Session。** 跨轮次的自动对话历史。
5. **Tracing。** 内置 span，覆盖 LLM 生成、工具调用、交接、护栏。

### 交接即工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它通知运行时：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 折叠它）。
2. 用其指令初始化目标 Agent。
3. 用目标 Agent 继续运行。

这是监督者模式 (第 13 课 / 第 28 课) 的产品化。

### 护栏

三种风格：

- **输入护栏。** 在第一个 Agent 的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **输出护栏。** 在最后一个 Agent 的输出上运行。捕获 PII 泄露、策略违规、格式错误的响应。
- **工具护栏。** 在每个函数工具上运行。验证参数、检查权限、审计执行。

模式：

- **并行** (默认)。护栏 LLM 与主 LLM 并行运行。尾部延迟更低。如果触发，主 LLM 的工作被丢弃（token 浪费）。
- **阻塞** (`run_in_parallel=False`)。护栏 LLM 先运行。如果触发，主调用不浪费 token。

触发引发 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每次 LLM 生成、工具调用、交接和护栏发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可退出。`add_trace_processor(processor)` 将 span 扇出到你自己的后端以及 OpenAI 的。

### 会话

`Session` 在后端（SQLite、Redis、自定义）存储对话历史。`Runner.run(agent, input, session=session)` 自动加载和追加。

### 这个模式哪里会出错

- **交接漂移。** Agent A 交接给 Agent B，B 又交接回 A。添加跳数计数器。
- **护栏绕过。** 工具护栏仅在函数工具上触发；内置工具（文件读取器、网络抓取）需要单独的策略。
- **过度追踪。** span 中的敏感内容。与 OTel GenAI 内容捕获规则 (第 23 课) 配对——外部存储，按 ID 引用。

## 构建

`code/main.py` 用标准库实现 SDK 形态：

- `Agent`、`FunctionTool`、`Handoff`（作为带传递语义的函数工具）。
- `Runner` 带输入/输出/工具护栏、交接分发和跳数计数器。
- 一个简单的 span 发出器展示追踪形态。
- 一个 triage Agent，根据用户查询交接给 billing 或 support；护栏在一个输入上触发。

运行：

```
python3 code/main.py
```

追踪显示两次成功的交接、一次输入护栏触发，以及镜像真实 SDK 发出内容的 span 树。

## 使用

- **OpenAI Agents SDK** 用于 OpenAI 优先的产品。
- **Claude Agent SDK** (第 17 课) 用于 Claude 优先的产品。
- **LangGraph** (第 13 课) 当你需要显式状态和持久化恢复时。
- **自定义** 当你需要精确控制时（语音、多提供商、联邦部署）。

## 交付

`outputs/skill-agents-sdk-scaffold.md` 搭建一个 Agents SDK 应用，包含 triage Agent、交接、输入/输出/工具护栏、会话存储和追踪处理器。

## 练习

1. 添加交接跳数计数器：在 N 次传递后拒绝。追踪行为。
2. 将 `nest_handoff_history` 实现为一个选项——在传递前将先前消息折叠为一个摘要。
3. 编写一个阻塞输出护栏。比较在会触发它的 prompt 和会通过的 prompt 上的延迟。
4. 将 `add_trace_processor` 接入 JSON 日志记录器。每个 span 发出什么形状？
5. 阅读 SDK 文档。将你的标准库玩具移植到 `openai-agents-python`。你建模错了什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Agent | "LLM + 指令" | SDK 中的 Agent 类型；拥有工具和交接 |
| Handoff | "传递" | 模型调用以委派给另一个 Agent 的工具 |
| Guardrail | "策略检查" | 对输入 / 输出 / 工具调用的验证 |
| Tripwire | "护栏触发" | 护栏拒绝时引发的异常 |
| Session | "历史存储" | 在运行之间持久化的对话记忆 |
| Tracing | "Span" | 内置可观测性，覆盖 LLM + 工具 + 交接 + 护栏 |
| 阻塞护栏 (Blocking guardrail) | "顺序检查" | 护栏先运行；触发时不浪费 token |
| 并行护栏 (Parallel guardrail) | "并发检查" | 护栏并行运行；延迟更低，触发时浪费 token |

## 扩展阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 原语、交接、护栏、追踪
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应物
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 何时根本不需要交接
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK span 映射到的标准
