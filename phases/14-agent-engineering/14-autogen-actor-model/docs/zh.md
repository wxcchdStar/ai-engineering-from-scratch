# AutoGen v0.4：Actor 模型与 Agent 框架 (AutoGen v0.4: Actor Model and Agent Framework)

> AutoGen v0.4 (Microsoft Research, 2025 年 1 月) 围绕 actor 模型重新设计了 Agent 编排。异步消息交换、事件驱动 Agent、故障隔离、原生并发。该框架目前处于维护模式，Microsoft Agent Framework (2025 年 10 月公开预览) 成为其继任者。

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 01 (Agent 循环), Phase 14 · 12 (工作流模式)
**时间：** 约 75 分钟

## 学习目标

- 描述 actor 模型：Agent 即 actor，消息是唯一的 IPC，每个 actor 故障隔离。
- 列举 AutoGen v0.4 的三个 API 层——Core、AgentChat、Extensions——及其各自的用途。
- 解释为什么将消息传递与处理解耦能带来故障隔离和原生并发。
- 用标准库在 Python 中实现一个 actor 运行时，并将一个双 Agent 代码审查流程移植到其上。

## 问题

大多数 Agent 框架是同步的：一个 Agent 生产，一个 Agent 消费，在调用栈中。故障会崩溃整个栈。并发是后来附加的。分布式需要重写。

AutoGen v0.4 的答案：actor 模型。每个 Agent 是一个带有私有收件箱的 actor。消息是唯一的交互方式。运行时将传递与处理解耦。故障隔离到单个 actor。并发是原生的。分布式只是不同的传输。

## 概念

### Actor

一个 actor 有：

- 私有状态（永远不能从外部直接触及）。
- 一个收件箱（消息队列）。
- 一个处理器：`receive(message) -> effects`，其中 effects 可以是"回复"、"发送给其他 actor"、"生成新 actor"、"更新状态"、"停止自身"。

两个 actor 不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三个 API 层

1. **Core。** 底层 actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 任务驱动的高级 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a 直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱并返回。运行时稍后传递。三个后果：

- **故障隔离。** Agent B 崩溃不会崩溃 Agent A——运行时在 B 的处理器中捕获故障并决定做什么（记录、重试、死信）。
- **原生并发。** 许多消息同时飞行；actor 并发处理其收件箱。
- **分布式就绪。** 收件箱 + 传输是相同的抽象，无论 actor 在进程内还是在另一台主机上。

### 拓扑

- **RoundRobinGroupChat。** Agent 按固定轮换轮流发言。
- **SelectorGroupChat。** 一个选择器 Agent 基于对话上下文选择谁下一个发言。
- **Magentic-One。** 参考多 Agent 团队，用于网页浏览、代码执行、文件处理。基于 AgentChat 构建。

### 可观测性

内置 OpenTelemetry 支持。每条消息发出一个 span；工具调用携带 `gen_ai.*` 属性，遵循 2026 年 OTel GenAI 语义约定 (第 23 课)。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 对研究和原型开发是稳定的。Microsoft 已将活跃开发转移到 Microsoft Agent Framework (2025 年 10 月 1 日公开预览；1.0 GA 目标为 2026 年 Q1 末)。AutoGen 模式可以干净地向前移植——actor 模型是持久的思想。

## 构建

`code/main.py` 用标准库实现一个 actor 运行时：

- `Message` — 类型化负载，包含 `sender`、`recipient`、`topic`、`body`。
- `Actor` — 抽象类，包含 `receive(message, runtime)`。
- `Runtime` — 事件循环，包含共享队列、传递、故障隔离。
- 一个双 actor 演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 运行检查清单；它们交换消息直到达成共识。

运行：

```
python3 code/main.py
```

追踪显示消息传递、一个 actor 中的模拟故障不会崩溃另一个 actor，以及收敛到共享裁决。

## 使用

- **AutoGen v0.4/v0.7** (维护模式) — 对研究、原型开发、多 Agent 模式稳定。
- **Microsoft Agent Framework** (公开预览) — 前进路径；相同的 actor 模型思想，API 焕新。
- **LangGraph swarm 拓扑** (第 13 课) — 通过共享工具交接的类似模式。
- **自定义 actor 运行时** — 当你需要特定的传输（NATS、RabbitMQ、gRPC）时。

## 交付

`outputs/skill-actor-runtime.md` 为给定的多 Agent 任务生成一个最小 actor 运行时加一个团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列：当处理器抛出异常时，将失败的消息停放以供人工检查。在你的玩具中 DLQ 被命中的频率如何？
2. 实现 `SelectorGroupChat`：一个选择器 actor 基于对话状态选择谁处理下一条消息。
3. 添加分布式传输：将进程内队列替换为 JSON-over-HTTP 服务器，使 actor 可以在不同进程中运行。
4. 为每条消息接入一个 OTel span（或 no-op 替代）。按第 23 课发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具移植到真实的 `autogen_core` API。你跳过了哪些在生产中重要的东西？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Actor | "Agent" | 私有状态 + 收件箱 + 处理器；无共享内存 |
| 消息 (Message) | "事件" | 类型化负载；actor 之间唯一的交互方式 |
| 收件箱 (Inbox) | "邮箱" | 每个 actor 的待处理消息队列 |
| 运行时 (Runtime) | "Agent 主机" | 路由消息并隔离故障的事件循环 |
| 主题 (Topic) | "通道" | actor 之间的命名发布-订阅路由 |
| 故障隔离 (Fault isolation) | "让它崩溃" | 一个 actor 失败不会崩溃其他 actor |
| RoundRobinGroupChat | "固定轮换团队" | Agent 按顺序轮流发言 |
| SelectorGroupChat | "上下文路由团队" | 选择器选择谁下一个发言 |
| Magentic-One | "参考团队" | 用于网页 + 代码 + 文件的多 Agent 小队 |

## 扩展阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重新设计文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 图形状替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的 span
