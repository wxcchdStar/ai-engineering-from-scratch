# Agent 框架权衡 — LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都卖着同一个演示（研究 Agent 生成一份报告），也藏着同一个 bug（状态 schema 与编排层打架）。选择那个其抽象与你的问题形状相匹配的框架；其余的都是你要写两遍的胶水代码。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 11 · 09（函数调用），Phase 11 · 16（LangGraph）
**时间：** 约 45 分钟

## 问题

你有一个需要多次 LLM 调用的任务。它可能是一个研究工作流（规划、搜索、总结、引用）。也可能是一个代码审查流水线（解析 diff、评审、打补丁、验证）。还可能是一个能订机票、写邮件、报差旅费的多轮助手。你选了一个框架。

三天后，你发现框架的抽象开始泄漏。CrewAI 给了你角色，但当"研究员"需要把结构化计划交给"写手"时，它就跟你对着干。AutoGen 给了你 Agent 之间的对话，但没有一等公民的状态，所以你的检查点只是一个对话日志的 pickle。LangGraph 给了你状态图，但强迫你在知道 Agent 会做什么之前就命名每一个转移。Agno 给了你单 Agent 原语，但当你试图扇出到三个并发 worker 时它就尖叫。

解决办法不是"选最好的框架"，而是将框架的核心抽象与你的问题形状匹配起来。本课将绘制这张地图。

## 概念

![Agent 框架矩阵：核心抽象 vs 问题形状](../assets/framework-matrix.svg)

2026 年的格局由四个框架主导。它们的核心抽象各不相同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph`（状态图）—— 类型化状态、节点、条件边、检查点器。 | 具有显式状态和人在回路中断的工作流；需要时间旅行调试的生产级 Agent。 | 拓扑结构未知的松散、角色驱动的头脑风暴。 |
| **CrewAI** | `Crew`（团队）—— 角色（目标、背景故事）、任务、流程（顺序或层级）。 | 具有简短线性/层级计划的角色扮演或人设驱动工作流。 | 超出 Crew 轮次历史的任何有状态内容；复杂的分支。 |
| **AutoGen** | `ConversableAgent` 对 —— 两个或更多 Agent 轮流对话，直到满足退出条件。 | 多 Agent *对话*（师生、提议者-批评者、演员-评审者），思维从对话中涌现。 | 具有已知 DAG 的确定性工作流；任何需要在重启后持久化状态的内容。 |
| **Agno** | `Agent` —— 单个 LLM + 工具 + 记忆，可组合成团队。 | 快速构建的单 Agent 和轻量级团队；强大的多模态能力和内置存储驱动。 | 具有自定义 reducer 的深层、显式分支图。 |

### "抽象"到底意味着什么

框架的核心抽象就是你在白板上推销架构时画的那个东西。

- **LangGraph** → 你画一张图。节点是步骤，边是转移，每个时间点的状态对象都有类型。心智模型是一台状态机。
- **CrewAI** → 你画一张组织架构图。每个角色有一份职位描述，一个管理者路由任务。心智模型是一个小型专家团队。
- **AutoGen** → 你画一条 Slack 私信。两个 Agent 互相发消息；如果你需要主持人，第三个加入。心智模型是聊天。
- **Agno** → 你画一个带工具挂件的方框。把方框并排放置就是一个团队。心智模型是"开箱即用的 Agent"。

### 状态问题

状态是大多数框架选择在生产环境中崩溃的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型），逐字段 reducer，一等公民的检查点器（SQLite/Postgres/Redis）。恢复、中断和时间旅行都是免费的。*（参见 Phase 11 · 16。）*
- **CrewAI。** 状态通过 `context` 字段以字符串形式在任务之间流动，或通过 `output_pydantic` 以结构化方式流动。没有开箱即用的持久化 Crew 存储；如果 Crew 必须在重启后存活，你得自己外挂一个。
- **AutoGen。** 状态就是聊天历史和你定义的任何 `context`。对话记录可以持久化；任意工作流状态不能，除非你编写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB），通过 `storage=` 附加到 `Agent` 上 —— 对话会话和用户记忆自动持久化。不是完整的图检查点器；而是一个会话存储。

### 分支问题

每个非平凡的 Agent 都会分支。谁来决定分支很关键。

- **LangGraph** —— 你通过条件边来决定。路由是一个带有命名分支的 Python 函数。分支在编译后的图中是一等公民；检查点器记录走了哪个分支。
- **CrewAI** —— 在层级模式下由管理者决定；在顺序模式下由你在构建时决定。路由隐含在任务列表中；在管理者的 prompt 之外没有一等公民的"if"。
- **AutoGen** —— Agent 通过对话来决定。分支从谁下一个发言中涌现。`GroupChat`（群聊）选择下一个发言者；你可以手写 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** —— Agent 通过调用哪个工具来决定。团队有协调者/路由/协作者模式；超出此范围的分支由开发者负责。

### 可观测性问题

- **LangGraph** —— 通过 LangSmith 或任何 OTel 导出器实现 OpenTelemetry。每个节点转移都是一个 trace span；检查点兼作可回放的 trace。LangSmith 是一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** —— 自 2025 年底以来有一等公民的 OpenTelemetry；集成 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen** —— 通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 有连接器。追踪粒度是每个 Agent 消息级别，而非每个节点级别。
- **Agno** —— 内置 `monitoring=True` 标志加上 OpenTelemetry 导出器；与 Langfuse 紧密集成以获取会话追踪。

### 成本与延迟

这四个框架都会增加每次调用的开销（框架逻辑、验证、序列化）。开销从低到高大致为：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要由框架做了多少额外的 LLM 路由决定。CrewAI 的层级管理者消耗 token 来决定谁下一个上场；AutoGen 的 `GroupChatManager` 同理。LangGraph 只在你写 `llm.invoke` 的地方消耗 token。Agno 的单 Agent 路径很薄。

当每次运行的成本很重要时，优先选择显式路由（LangGraph 边、AutoGen 的 `speaker_selection_method`），而非 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。一等公民的 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具均可适配接入。通过 `allow_delegation=True` 实现 Crew 到 Crew 的委托。
- **AutoGen** → `FunctionTool` 包装任意 Python 可调用对象；MCP 适配器可用。Agent 到 Agent 模式与 AG2 生态系统紧密耦合。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可在 Agent 和团队之间共享。

## 技能

> 你能用一句话解释为什么某个框架适合某个 Agent 问题。

构建前检查清单：

1. **画出形状。** 这是一个图（类型化状态、命名转移）？一个角色扮演（专家交接工作）？一次聊天（Agent 对话直到完成）？一个带工具的单一 Agent？
2. **决定谁分支。** 开发者决定分支 → LangGraph。管理者 Agent 决定 → CrewAI 层级模式。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 你需要从检查点恢复吗？时间旅行？运行中途人工中断？如果是，LangGraph 是默认选择；Agno 会话覆盖对话范围的状态。
4. **检查成本预算。** LLM 选择的路由每次轮次消耗额外 token。如果 Agent 每天运行数千次，优先选择显式路由。
5. **预算框架开销。** 每个框架都是额外的依赖。如果任务是两次 LLM 调用加一个工具，写 30 行纯 Python；没有框架比有框架更便宜。

在你画不出那个图、组织架构图、聊天或 Agent 方框之前，不要伸手去拿框架。不要选一个迫使你为真正需要的东西与其状态模型打架的框架。

## 决策矩阵

| 问题形态 | 推荐框架 | 理由 |
|----------|---------|------|
| 带类型化状态、人工审批、长时间运行的工作流 DAG | LangGraph | 一流的状态管理、检查点、中断、时间旅行。 |
| 带不同角色的研究/写作流水线 | CrewAI（顺序模式）或 LangGraph 子图 | CrewAI 中按角色分配任务表达成本很低；当分支变复杂时用 LangGraph 扩展。 |
| 提议者-评判者或教师-学生对谈 | AutoGen | 双智能体对话是其原生形态。 |
| 带工具、会话、记忆的单个智能体 | Agno | 最简配置，内置存储和记忆。 |
| 带 reducer 的数千并行扇出 | LangGraph + `Send` | 唯一提供一流并行分发原语的框架。 |
| 快速原型，不绑定框架 | 纯 Python + 提供商 SDK | 无框架即最快框架。 |

## 练习

1. **简单。** 对同一任务——"调研 Anthropic 的总部地址，撰写 200 字简报，注明来源"——分别用 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个角色：researcher、writer、editor）实现。报告每次运行的 token 消耗和代码行数。
2. **中等。** 用 AutoGen（researcher ↔ writer 对谈，editor 通过 `GroupChat` 加入）和 Agno（一个带 `search_tools` 和 `write_tools` 的智能体，外加会话存储）实现同一任务。从以下维度对四种实现进行排名：(a) 每次运行成本，(b) 崩溃后恢复能力，(c) 在写入步骤前注入人工审批的能力。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，输入简短的问题描述（JSON 格式：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），返回推荐方案及一句话理由。用自己设计的六个案例验证。

## 关键术语

| 术语 | 别人怎么说 | 实际含义 |
|------|-----------|---------|
| 编排 (Orchestration) | "智能体之间如何协调" | 决定下一个节点/角色/智能体运行的调度层。 |
| 持久化状态 (Durable state) | "重启后恢复" | 能经受进程消亡而存活的状态，挂载在检查点或会话存储上。 |
| LLM 选定路由 (LLM-selected routing) | "让模型决定" | 每轮由规划 LLM 选择下一步；灵活但每次决策都要消耗 token。 |
| 显式路由 (Explicit routing) | "开发者决定" | 由 Python 函数或静态边决定下一步；成本低且可审计。 |
| Crew | "一个 CrewAI 团队" | 角色 + 任务 + 流程（顺序或层级式）绑定为一次可运行单元。 |
| GroupChat | "AutoGen 的多智能体聊天" | N 个智能体之间由发言选择器管理的对话。 |
| Team (Agno) | "Agno 多智能体" | 在一组智能体上运行的路由/协调/协作模式。 |
| StateGraph | "LangGraph 的图" | 类型化状态、节点、条件边、检查点原语。 |

## 扩展阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph、checkpointer、interrupt、time-travel。
- [CrewAI documentation](https://docs.crewai.com/) — Crew、Flow、Agent、Task、Process。
- [AutoGen documentation](https://microsoft.github.io/autogen/) — ConversableAgent、GroupChat、team、tool。
- [Agno documentation](https://docs.agno.com/) — Agent、Team、Workflow、storage、memory。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 模式库（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer），与框架无关。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 所有框架包装的底层原语。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — AutoGen 的设计论文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — CrewAI 风格角色栈所基于的角色扮演基础。
- Phase 11 · 16 (LangGraph) — 本课进行基准对比的框架。
- Phase 11 · 19 (Reflexion) — 可清晰映射到 LangGraph 但难以映射到 CrewAI 的模式。
- Phase 11 · 22 (Production observability) — 如何为你选定的框架接入可观测性。
