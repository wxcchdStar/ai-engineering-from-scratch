# LangGraph：有状态图与持久化执行 (LangGraph: Stateful Graphs and Durable Execution)

> LangGraph 是 2026 年底层有状态编排的参考标准。Agent 是一个状态机；节点是函数；边是转换；状态是不可变的，每一步后都做检查点。从任何失败处精确恢复。

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 01 (Agent 循环), Phase 14 · 12 (工作流模式)
**时间：** 约 75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：带不可变状态的状态机、函数节点、条件边和步骤后检查点。
- 列举文档强调的四种能力：持久化执行、流式传输、人机协同、全面记忆。
- 解释 LangGraph 支持的三种编排拓扑：监督者、点对点 (swarm)、层级式 (嵌套子图)。
- 用标准库实现一个带不可变状态、条件边和检查点/恢复循环的状态图。

## 问题

Agent 和工作流共享一个问题：当一个 40 步的运行在第 38 步失败时，你希望从第 38 步恢复，而不是从头开始。二等的状态模型让运维人员在假设每次都是全新运行的库上拼凑重试。

LangGraph 的设计答案：状态是一等类型对象，变更是显式的，检查点在每个节点后持久化。恢复是一个 `load_state(session_id)` 调用。

## 概念

### 图

一个图由以下定义：

- **状态类型。** 一个类型化字典（或 Pydantic 模型），每个节点读取和变更它。
- **节点。** 纯函数 `(state) -> state_update`。更新在返回后合并到状态中。
- **边。** 节点之间的条件或直接转换。
- **入口和出口。** `START` 和 `END` 哨兵节点标记边界。

示例：一个带有 `classify`、`refund`、`bug`、`sales`、`done` 节点的 Agent——一个作为图的路由工作流。

### 持久化执行

每个节点返回后，运行时序列化状态并将其写入检查点器（SQLite、Postgres、Redis、自定义）。在第 N 步失败时，运行时可以 `resume(session_id)` 并从第 N+1 步以精确状态继续。

LangGraph 文档明确强调了这很重要的生产用户：Klarna、Uber、J.P. Morgan。其主张不是图的形状，而是图的形状加上检查点使恢复变得廉价。

### 流式传输

每个节点可以产出部分输出。图将每个节点的增量事件流式传输给调用者，使 UI 在图运行时更新。

### 人机协同

在节点之间检查和修改状态。实现：在关键节点前暂停，将状态呈现给人类，接受修改，恢复。检查点器使这变得容易，因为状态已经被序列化。

### 记忆

短期（单次运行内——状态中的对话历史）和长期（跨运行——通过检查点器加上单独的长期存储持久化）。LangGraph 通过工具与外部记忆系统（Mem0、自定义）集成。

### 三种拓扑

1. **监督者。** 中心路由 LLM 分发给专家子 Agent。`langgraph-supervisor` 中的 `create_supervisor()`（尽管 LangChain 团队在 2026 年建议通过工具调用直接实现以获得更多上下文控制）。
2. **Swarm / 点对点。** Agent 通过共享工具接口直接交接。没有中心路由器。
3. **层级式。** 监督者管理子监督者，实现为嵌套子图。

### 这个模式哪里会出错

- **检查点太小。** 仅对对话轮次做检查点会使工具状态和记忆写入无法恢复。完整状态必须序列化。
- **非确定性节点。** 恢复假设节点输入产生相同的状态更新。随机种子、墙上时钟、外部 API 必须被捕获。
- **过度使用条件边。** 每条边都是条件的图是一个无法推理的状态机。优先使用线性链加偶尔的分支。

## 构建

`code/main.py` 用标准库实现一个有状态图：

- `State` — 一个类型化字典，包含 `messages`、`step`、`route`、`output`、`human_approval`。
- `Node` — 接受状态并返回更新字典的可调用对象。
- `StateGraph` — 节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存中的假实现）— 每个节点后序列化状态；`load(session_id)` 恢复。
- 一个演示图：classify -> branch(refund / bug / sales) -> human gate -> send。

运行：

```
python3 code/main.py
```

追踪显示第一次运行在 human gate 处失败，持久化，然后恢复产生最终输出。

## 使用

- **LangGraph** — 参考标准，生产就绪。使用 `create_react_agent`、`create_supervisor`，或构建自己的图。
- **AutoGen v0.4** (第 14 课) — 高并发场景的 actor 模型替代方案。
- **Claude Agent SDK** (第 17 课) — 带内置会话存储的托管 harness。
- **自定义** — 当你需要对状态形状或检查点器后端的精确控制时。

## 交付

`outputs/skill-state-graph.md` 在任何目标运行时中生成 LangGraph 形态的状态图，包含检查点和恢复。

## 练习

1. 添加一条从 `classify` 到 `end` 的条件边，当分类置信度低于阈值时触发。在人类手动设置 `route` 后恢复运行。
2. 将类 SQLite 的假实现替换为真实的 SQLite 检查点器。测量每步序列化开销。
3. 实现并行边：两个节点并发运行，通过自定义 reducer 合并。不可变状态在这里带来了什么好处？
4. 阅读 `langgraph-supervisor` 参考。将玩具移植到 `create_supervisor`。比较追踪形态。
5. 添加流式传输：每个节点在运行时产出部分状态。在增量到达时打印它们。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 状态图 (State graph) | "Agent 即状态机" | 类型化状态 + 节点 + 边 + reducer |
| 检查点器 (Checkpointer) | "持久化后端" | 每个节点后序列化状态；使恢复成为可能 |
| Reducer | "状态合并器" | 将当前状态与节点更新组合的函数 |
| 条件边 (Conditional edge) | "分支" | 由状态函数选择的边 |
| 子图 (Subgraph) | "嵌套图" | 在另一个图中作为节点使用的图 |
| 持久化执行 (Durable execution) | "从失败中恢复" | 在最后一个成功节点处以精确状态重新开始 |
| 监督者 (Supervisor) | "路由 LLM" | 专家子 Agent 的中心分发器 |
| Swarm | "P2P Agent" | Agent 通过共享工具交接；无中心路由器 |

## 扩展阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — 监督者模式 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor 模型替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — 会话存储和子 Agent
