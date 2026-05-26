# 生产运行时：队列、事件、定时 (Production Runtimes: Queue, Event, Cron)

> 生产 Agent 运行在六种运行时形态上：请求-响应、流式、持久化执行、基于队列的后台、事件驱动和定时。在选择框架之前选择形态。可观测性在每种形态中都是关键。

**类型：** 学习 (Learn)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 13 (LangGraph), Phase 14 · 22 (语音)
**时间：** 约 60 分钟

## 学习目标

- 列举六种生产运行时形态并将每种匹配到框架/产品模式。
- 解释为什么持久化执行 (LangGraph) 对长程任务很重要。
- 描述事件驱动运行时以及 Claude Managed Agents 何时适用。
- 解释多步 Agent 的可观测性即关键的主张。

## 问题

生产 Agent 以 Jupyter notebook 不会暴露的方式失败：第 37 步的网络超时、用户在语音通话中途挂断、定时任务在机器重启时死亡、后台工作器内存耗尽。运行时形态决定了哪些失败是可存活的。

## 概念

### 请求-响应

- 同步 HTTP。用户等待完成。
- 仅对短任务可行（<30s）。
- 技术栈：Agno (Python + FastAPI)、Mastra (TypeScript + Express/Hono/Fastify/Koa)。
- 可观测性：标准 HTTP 访问日志 + OTel span。

### 流式

- SSE 或 WebSocket 用于渐进式输出。
- LiveKit 将其扩展到 WebRTC 用于语音/视频 (第 22 课)。
- 技术栈：任何支持流式的框架 + 处理 SSE/WS 的前端。
- 可观测性：每块计时、首个 token 延迟、尾部延迟。

### 持久化执行

- 每步后状态做检查点；失败时自动恢复。
- AutoGen v0.4 actor 模型将故障隔离到单个 Agent (第 14 课)。
- LangGraph 的核心差异化 (第 13 课)。
- 当步骤数未知且恢复成本高时至关重要。

### 基于队列 / 后台

- 作业进入队列，工作器拾取，结果通过 webhook 或 pub/sub 流回。
- 对长程 Agent 至关重要（每个任务数十到数百步，据 Anthropic 的 computer use 公告）。
- 技术栈：Celery (Python)、BullMQ (Node)、SQS + Lambda (AWS)、自定义。
- 可观测性：队列深度、每个作业的延迟分布、DLQ 大小。

### 事件驱动

- Agent 订阅触发器：新邮件、PR 打开、定时触发。
- Claude Managed Agents 开箱即用覆盖此场景 (第 17 课)。
- CrewAI Flows (第 15 课) 结构化事件驱动确定性工作流。
- 可观测性：触发源、事件到启动延迟、Agent 延迟。

### 定时

- 定期运行的 Cron 形态 Agent。
- 与持久化执行结合，使失败的夜间运行在下一个周期恢复。
- 技术栈：Kubernetes CronJob + 持久化框架；托管（Render cron、Vercel cron）。

### 2026 年部署模式

- **CrewAI Flows** 用于事件驱动生产。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音 (第 22 课)。
- **Claude Managed Agents** 用于托管长时间运行异步。

### 可观测性是关键

没有 OpenTelemetry GenAI span (第 23 课) 加上 Langfuse/Phoenix/Opik 后端 (第 24 课)，你无法调试在第 40 步失败的多步 Agent。这对生产不是可选的。这是"我们快速调试"和"我们用更多日志从头重放"之间的区别。

### 生产运行时在哪里失败

- **错误的形态选择。** 为 5 分钟任务选择请求-响应。用户挂断；工作器堆积；重试叠加。
- **没有 DLQ。** 没有死信的队列工作器。失败的作业消失。
- **不透明的后台工作。** 后台 Agent 运行没有追踪导出。失败在用户报告之前不可见。
- **跳过持久化状态。** 任何超过 30 秒且你无法承受重启的运行都需要持久化执行。

## 构建

`code/main.py` 是一个标准库多形态演示：

- 请求-响应端点（纯函数）。
- 流式处理器（生成器）。
- 带 DLQ 的基于队列的工作器。
- 事件触发器注册表。
- Cron 形态调度器。

运行：

```bash
python3 code/main.py
```

输出：五条追踪显示每种形态在相同任务上的行为。相同的 Agent 逻辑，不同的外壳。持久化执行（第六种形态）有意在第 13 课中用 LangGraph 检查点覆盖。

## 使用

- **请求-响应** 用于聊天风格 UX。
- **流式** 用于渐进式响应。
- **持久化** 用于长程任务。
- **队列** 用于批处理 / 异步 / 长时间运行。
- **事件** 用于 Agent 反应性。
- **Cron** 用于内务（记忆整合、评估、成本报告）。

## 交付

`outputs/skill-runtime-shape.md` 为任务选择运行时形态并接入可观测性需求。

## 练习

1. 将你的第 01 课 ReAct 循环移植到你的技术栈中的全部六种形态。哪种形态适合哪个产品表面？
2. 为基于队列的演示添加 DLQ。模拟 10% 作业失败；显示 DLQ 大小。
3. 编写一个 cron 触发的评估 Agent，每晚针对当天前 20 条追踪运行。
4. 实现带背压的流式：如果客户端慢，暂停 Agent。这如何与轮次预算交互？
5. 阅读 Claude Managed Agents 文档。你何时会将自托管长程 Agent 迁移到托管？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 请求-响应 (Request-response) | "同步" | 用户等待；仅短任务 |
| 流式 (Streaming) | "SSE / WS" | 渐进式输出；更好的 UX；每块延迟可观测 |
| 持久化执行 (Durable execution) | "从失败中恢复" | 检查点状态；在最后一步重新开始 |
| 基于队列 (Queue-based) | "后台作业" | 生产者 / 工作器池 / DLQ |
| 事件驱动 (Event-driven) | "基于触发器" | Agent 对外部事件做出反应 |
| DLQ | "死信队列" | 失败作业的停放场 |
| Claude Managed Agents | "托管 harness" | Anthropic 托管的长时间运行异步，带缓存 + 压缩 |

## 扩展阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 持久化执行细节
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管长时间运行异步
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "每个任务数十到数百步"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor 模型故障隔离
