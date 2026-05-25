# LangGraph — 面向 Agent 的状态机

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 编写的 ReAct 循环是一张你可以进行检查点、中断、分支和时间旅行的图。Agent 没有变，变的是围绕它的框架。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 11 · 09（函数调用）、Phase 11 · 14（模型上下文协议）
**时间：** ~75 分钟

## 问题所在

你发布了一个函数调用 Agent。它在前三轮运行正常，然后出了问题：模型尝试调用的工具返回了 500 错误，用户在任务中途改变了主意，或者 Agent 决定在没有人工签署的情况下退款。那个 `while True:` 循环没有任何钩子。你无法暂停它，无法回退它，也无法分支探索"如果模型选择了另一个工具会怎样"。一旦你把它从演示阶段推到生产环境，这个 Agent 就变成了一个黑箱——要么成功，要么失败。

一旦你理解了这一点，下一步就显而易见。Agent 本身已经是一个状态机——系统提示词 + 消息历史 + 待处理的工具调用 + 下一步动作。把状态机显式化：用节点表示"模型思考"、"工具运行"、"人工审批"，用边表示它们之间的条件转换。一旦图被显式化，框架就自动获得了四项能力：检查点（在步骤之间保存状态）、中断（暂停等待人工）、流式传输（流式输出 token 和中间事件）以及时间旅行（回退到先前状态并尝试不同分支）。

LangGraph 就是提供这种抽象的库。它不是 LangChain 意义上的 Agent 框架（"这里是一个 AgentExecutor，祝你好运"）。它是一个具有一等状态、一等持久化和一等中断的图运行时。Agent 循环是画出来的，而不是手写出来的。

## 核心概念

![LangGraph StateGraph：节点、边和检查点](../assets/langgraph-stategraph.svg)

一个 `StateGraph`（状态图）包含三样东西。

1. **状态（State）。** 一个在图中流动的类型化字典（TypedDict 或 Pydantic 模型）。每个节点接收完整的状态并返回部分更新，LangGraph 使用每个字段的*归约器*来合并——列表累加用 `operator.add`，默认覆盖。
2. **节点（Nodes）。** Python 函数 `state -> partial_state`。每个函数是一个离散步骤："调用模型"、"运行工具"、"摘要"。
3. **边（Edges）。** 节点之间的转换。静态边只去一个地方。条件边接受一个路由函数 `state -> next_node_name`，这样图就可以根据模型输出来分支。

你编译这个图。编译会绑定拓扑结构，附加一个检查点（可选但对生产环境至关重要），并返回一个可运行对象。你用初始状态和一个 `thread_id`（线程 ID）来调用它。执行的每一步都会将检查点持久化，以 `(thread_id, checkpoint_id)` 作为键。

### 四大超能力

**检查点。** 每次节点转换都将新状态写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。通过用相同的 `thread_id` 再次调用图来恢复执行。图会从它暂停的地方继续。

**中断。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行会在该节点运行之前停止。状态被持久化。你的 API 向用户返回"等待审批"。之后对同一个 `thread_id` 发起带有 `Command(resume=...)` 的请求即可恢复执行。

**流式传输。** `graph.stream(state, mode="updates")` 会在事件发生时产出状态增量。`mode="messages"` 会在模型节点内部流式输出 LLM token。`mode="values"` 产出完整快照。你可以选择在 UI 中展示什么。

**时间旅行。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。将任意先前的 `checkpoint_id` 传给 `graph.invoke`，你就可以从那个点分叉。非常适合调试（"如果模型选了工具 B 会怎样？"）以及用于回放生产轨迹的回归测试。

### 归约器是关键

每个状态字段都有一个归约器。大多数默认值没问题——新值覆盖旧值。但消息列表需要 `operator.add`，这样新消息会追加而不是替换。并行边通过归约器合并它们的更新。如果两个节点都更新了 `messages` 而你忘记了 `Annotated[list, add_messages]`，第二个会静默胜出，你会丢失一半的轮次。归约器是这个库中唯一微妙的地方；只要搞对了，其余部分自然组合。

### 四个节点组成的 ReAct 图

一个生产级 ReAct Agent 由四个节点和两条边组成：

1. `agent` — 用当前消息历史调用 LLM。返回助手消息（可能包含 tool_calls）。
2. `tools` — 执行最后一条助手消息中的任何 tool_calls，将工具结果作为工具消息追加。
3. 从 `agent` 出发的条件边：如果最后一条消息有 tool_calls，则路由到 `tools`，否则路由到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

仅此而已。你在大约 40 行代码中就获得了完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → …），并自带检查点、中断和流式传输。

### StateGraph（状态图） vs Send（扇出）

`Send(node_name, state)` 允许一个节点派发并行子图。例如：Agent 决定同时查询三个检索器。每个 `Send` 会启动目标节点的并行执行；它们的输出通过状态归约器合并。这就是 LangGraph 表达编排器-工作者模式的方式，无需线程原语。

### 子图

一个已编译的图可以作为另一个图中的节点。外层图看到的是一个单一节点；内层图有自己的状态和自己的检查点。这就是团队构建监督者-工作者 Agent 的方式：监督者图将用户意图路由到每个领域的工作者子图。

## 动手构建

### 步骤 1：状态与节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是让消息列表累加而不是覆盖的归约器。忘记它是 LangGraph 中最常见的 bug。

### 步骤 2：使用线程运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个字典 `{node_name: state_delta}`。你的前端可以将其流式传输到 UI，让用户看到"Agent 正在思考… 正在调用 search_web… 已获取结果… 正在回答。"

### 步骤 3：添加人机协作中断

标记一个节点，使执行在运行之前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 在每次工具调用前暂停
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
```

# state["__interrupt__"] 已被设置。检查建议的工具调用。
# 如果批准：
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 如果拒绝：写入拒绝消息并恢复
app.update_state(config, {"messages": [AIMessage("被人工审核者阻止。")]})
```

状态、检查点和线程在中断期间全部持久化保留。执行期间之外，没有任何数据驻留在内存中。

### 第 4 步：时间旅行调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从之前的检查点分叉
target = history[3].config  # 回退三步
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从该点向前重放
```

传入 `None` 作为输入会从给定检查点重放；传入一个值则会在恢复前将该值作为对检查点状态的更新追加。这就是你无需重新运行整个对话即可复现一个出错的 agent 运行的方法。

### 第 5 步：为生产环境切换检查点存储

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都已内置提供。`MemorySaver` 仅供测试使用。任何需要跨重启持久化的场景都需要真正的存储后端。

## 核心技能

> 你把 agent 构建成图，而不是 `while True` 循环。

在使用 LangGraph 之前，先花 60 秒做设计：

1. **命名节点。** 每个离散的决策或产生副作用的动作都是一个节点。"agent 思考"、"工具运行"、"审核者批准"、"响应流式输出"。如果你列不出这些节点，说明这个任务还不是 agent 形态。
2. **声明状态。** 最小化的 TypedDict，为每个列表字段配备 reducer。不要把一切都塞进 `messages`；将任务特定字段（工作中的 `plan`、`budget` 计数器、`retrieved_docs` 列表）提升到顶层。
3. **画出边。** 除非下一步依赖模型输出，否则使用静态边。每条条件边都需要一个带命名分支的路由函数。
4. **从一开始就选定检查点存储。** 测试用 `MemorySaver`，其他场景用 Postgres/Redis/SQLite。没有检查点存储就不要发布——没有它意味着无法恢复、无法中断、无法时间旅行。
5. **在工具运行之前而非之后决定中断。** 批准机制放在进入产生副作用的节点的边上，这样可以在造成损害之前取消；验证机制放在模型输出的边上，这样可以低成本地拒绝错误的调用。
6. **默认使用流式输出。** UI 用 `mode="updates"`，模型节点内的 token 级流式用 `mode="messages"`，评估期间的全量快照用 `mode="values"`。

拒绝发布没有检查点存储的 LangGraph agent。拒绝发布在副作用**之后**才中断的 agent。拒绝发布 `messages` 字段不使用 `add_messages` 作为 reducer 的 agent。

## 练习

1. **简单。** 用计算器工具和网页搜索工具实现上述四节点 ReAct 图。验证 `list(app.get_state_history(config))` 对一轮两回合对话至少返回四个检查点。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，将结构化的 `plan: list[str]` 写入状态。让 `agent` 将计划步骤标记为已完成。如果 `plan` 在检查点恢复后丢失，则测试失败（reducer 配置错误）。
3. **困难。** 构建一个 supervisor 图，使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图都有自己的状态和检查点存储。在外层图上添加 `interrupt_before=["writer"]`，以便人工审核研究简报。确认从之前的检查点进行时间旅行只会重新运行分叉的分支。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraph 的图" | 在编译之前添加节点和边的构建器对象。 |
| Reducer | "字段如何合并" | 一个 `(old, new) -> merged` 函数，当节点返回该字段的更新时被调用；默认为覆盖，`add_messages` 则为追加。 |
| Thread | "一个对话 ID" | 一个 `thread_id` 字符串，用于界定一个会话的所有检查点的范围。 |
| Checkpoint | "一个暂停的状态" | 在节点转换之后持久化的完整图状态快照，以 `(thread_id, checkpoint_id)` 作为键。 |
| Interrupt | "暂停等待人工介入" | `interrupt_before` / `interrupt_after` 在节点边界处停止执行；通过 `Command(resume=...)` 恢复。 |
| Time-travel | "从之前的步骤分叉" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放。 |
| Send | "并行子图分发" | 节点可以返回的一个构造器，用于生成目标节点的 N 个并行执行实例。 |
| Subgraph | "作为节点的已编译图" | 一个已编译的 StateGraph，作为节点用于另一个图中；保留其自身的状态作用域。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、reducer、检查点存储和中断的权威参考。
- [LangGraph 概念：状态、reducer、检查点存储](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课使用的思维模型，直接来自官方来源。
- [LangGraph 持久化与检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的详细信息。
- [LangGraph 人机协作](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)` 以及编辑状态的模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个 LangGraph agent 实现的模式；阅读它以理解推理追踪的理论依据。
- [Anthropic — Building effective agents (2024 年 12 月)](https://www.anthropic.com/research/building-effective-agents) — 哪些图形状（链式、路由器、编排器-工作者、评估器-优化器）应优先选择以及何时使用。
- Phase 11 · 09（函数调用）— 每个 LangGraph agent 节点复用的工具调用原语。
- Phase 11 · 14（模型上下文协议）— 通过 MCP 适配器插入 LangGraph `ToolNode` 的外部工具发现机制。
- Phase 11 · 17（Agent 框架权衡）— 何时选择 LangGraph 而非 CrewAI、AutoGen 或 Agno。
