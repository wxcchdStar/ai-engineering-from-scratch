# 多智能体原语模型（Primitive Model）

> 2026 年发布的每个多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。只有四个原语（primitives）：智能体（agent）、交接（handoff）、共享状态（shared state）、编排器（orchestrator）。本课程从零开始构建它们，在所有四个原语上运行一个玩具系统，然后将每个主要框架映射到相同的坐标轴上，这样你就可以在一个段落内阅读任何新版本。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 第 14 阶段（智能体工程，Agent Engineering）、第 16 阶段 · 01（为什么是多智能体）
**时间：** ~60 分钟

## 问题

每六个月就会发布一个新的多智能体框架。2023 年的 AutoGen。2024 年的 CrewAI。2024 年的 LangGraph 和 OpenAI Swarm。2025 年 4 月的 Google ADK。2026 年 2 月的 Microsoft Agent Framework RC。每个新闻稿都声称自己是"正确的抽象"。

如果你试图一个一个地学习它们，你会筋疲力尽。API 看起来不同。文档对"智能体"的定义不一致。一个框架称其共享内存为"黑板（blackboard）"，另一个称其为"消息池（message pool）"，第三个称其为"StateGraph"。你开始怀疑这个领域只是在不断折腾。

其实不是。在营销之下，四个原语是稳定的。学习它们一次，就能在一个段落内阅读每个新框架。

## 概念

### 四个原语

1.  **智能体（Agent）**——一个系统提示词（system prompt）加上一个工具列表（tool list）。无状态；每次运行从其系统提示词和当前消息历史开始。
2.  **交接（Handoff）**——从智能体 A 到智能体 B 的结构化控制转移。在机制上，是一个返回新智能体的工具调用，或是一个遵循条件的图边（graph edge）。
3.  **共享状态（Shared state）**——多个智能体可以读取（有时写入）的任何数据结构。消息池、黑板、键值存储、向量记忆（vector memory）。
4.  **编排器（Orchestrator）**——决定谁下一个发言的角色。选项：显式图（确定性）、LLM 发言者选择器（软性）、最后一个发言者的交接调用（OpenAI Swarm），或队列上的调度器（群体架构，swarm architecture）。

这就是整个设计空间。每个框架为每个轴选择默认值；其余的都是表面语法。

### 每个 2026 年框架如何映射到它

| 框架 | 智能体 | 交接 | 共享状态 | 编排器 |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用者的问题 | LLM 的下一个交接调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的发言者选择器 | 消息池 | 选择器函数（LLM 或轮转） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 任务输出链式传递 | 管理者 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` 归约器（reducer） | 图，确定性的 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 模式特定 | 线程 / 上下文 | 模式特定 |
| Google ADK | 智能体 + A2A 卡片 | A2A 任务 | A2A 工件 | 宿主决定 |

表面差异看起来很大。底层：相同的四个旋钮。

### 为什么这很重要

一旦你看到了原语，框架比较就变成了一个简短的清单：

-   编排器是信任 LLM 来路由（Swarm），还是将路由固定在代码中（LangGraph）？
-   共享状态是全量历史（GroupChat）还是投影（projected，StateGraph 归约器）？
-   智能体可以修改彼此的提示词（CrewAI 管理者）还是只能交接（Swarm）？

这三个问题回答了 80% 的哪个框架适合给定问题。你不再寻找"最好的多智能体框架"，而是开始为你真正关心的轴进行设计。

### 无状态洞察

除共享状态外，每个原语都是无状态的。智能体是 `(prompt, tools)` 的函数。交接是一个函数调用。编排器是一个调度器。**系统中唯一有状态的东西是共享状态。** 所有有趣的 bug 都在那里：记忆污染（memory poisoning，第 15 课）、消息排序、版本控制、写入争用。

隐藏共享状态的框架（Swarm）将问题推给调用者。集中管理共享状态的框架（LangGraph 检查点、AutoGen 池）使其可检查，但将协调成本转移到共享状态实现上。

### 单个原语的剖析

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

没有记忆。没有状态。具有相同系统提示词和工具的两个智能体是可互换的。所有看起来像每个智能体状态的东西实际上都在共享状态或交接协议中。

#### 交接

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导地位：

-   **函数返回**——工具返回下一个智能体。这是 OpenAI Swarm 模式。智能体在其工具模式中携带路由。
-   **图边**——LangGraph。边是声明式的。LLM 产生一个值；条件选择下一个节点。
-   **发言者选择**——AutoGen GroupChat。一个选择器函数（有时本身就是一个 LLM 调用）读取池并选择下一个发言者。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

至少是一个消息列表。通常更多：结构化工件（CrewAI 任务输出）、类型化上下文（LangGraph 归约器）、外部记忆（MCP、向量数据库）。

两种拓扑：**全量池**（每个智能体看到每条消息）和**投影**（智能体看到角色范围视图）。全量池简单但扩展性差。投影池可扩展但需要前期模式设计。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种风格：

-   **静态**——图在构建时固定（LangGraph 确定性、CrewAI Sequential）。
-   **LLM 选择**——LLM 读取池并选择下一个发言者（AutoGen、CrewAI Hierarchical）。
-   **交接驱动**——当前智能体通过调用交接工具来决定（Swarm）。
-   **队列驱动**——工作者从共享队列中拉取；没有显式的下一个发言者（群体架构、Matrix）。

### 框架之间的变化

一旦原语固定下来，剩下的设计决策是：

-   **记忆策略**——临时 vs 持久检查点（LangGraph 检查点器）。
-   **安全边界**——谁可以批准交接（人机交互，human-in-the-loop）。
-   **成本核算**——每个智能体的 token 预算。
-   **可观测性**——跟踪交接，持久化状态以供重放。

所有这些都可以在原语之上实现。它们都不是新的原语。

## 构建它

`code/main.py` 用约 150 行标准库 Python 实现了四个原语。没有真正的 LLM——每个智能体是一个脚本化策略，因此重点保持在协调结构上。

该文件导出：

-   `Agent`——一个包含名称、系统提示词、工具、策略函数的数据类。
-   `Handoff`——一个返回新智能体的函数。
-   `SharedState`——一个线程安全的消息池。
-   `Orchestrator`——三种变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示通过所有三种编排器类型运行相同的三智能体流水线（研究 → 写作 → 审查），并在最后打印消息池。你可以看到输出仅在*谁选择下一个*上有所不同；智能体和共享状态在各次运行中是相同的。

运行它：

```
python3 code/main.py
```

预期输出：三次编排器运行，每种模式一次。每次打印最终的消息池。如果研究者提前决定完成，交接驱动的运行会到达更少的智能体——这是 LLM 路由权衡的缩影。

## 使用它

`outputs/skill-primitive-mapper.md` 是一个技能，它读取任何多智能体代码库或框架文档，并返回四原语映射。在新框架发布时运行它，以便在深入文档之前获得一个段落的理解。

## 交付它

在采用新框架之前，为其编写原语映射。如果你做不到，说明文档不完整，或者框架正在发明第五个原语（罕见——检查是否有你未见过的共享状态风格）。

将映射固定在架构文档中。当新团队成员加入时，先发送映射，再发送 API 文档。当框架版本变更时，对比映射，而不是变更日志。

## 练习

1.  用不同的智能体策略运行 `code/main.py` 三次。观察编排器的选择如何改变哪些智能体运行。
2.  实现第四种编排器类型：队列驱动的，智能体从共享状态中轮询工作。可能发生什么死锁，你如何检测它？
3.  阅读 LangGraph 快速入门（https://docs.langchain.com/oss/python/langgraph/workflows-agents）并将其重写为四个原语。LangGraph 的哪些抽象是 1:1 映射的，哪些是便利包装？
4.  阅读 OpenAI Swarm 手册（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 使四个原语中的哪一个最符合人体工程学，以及它将哪一个推给调用者。
5.  在此表中找到一个完全隐藏共享状态的框架。解释当智能体需要在不重新读取历史的情况下跨交接协调时，什么会出问题。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| 智能体（Agent） | "一个带工具的 LLM" | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| 交接（Handoff） | "控制转移" | 一个结构化调用，命名下一个智能体和可选负载。三种实现：函数返回、图边、发言者选择。 |
| 共享状态（Shared state） | "记忆" / "上下文" | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| 编排器（Orchestrator） | "协调器" | 决定谁下一个运行的角色。静态图、LLM 选择器、交接驱动或队列驱动。 |
| 原语（Primitive） | "抽象" | 每个框架参数化的四个轴之一。不是框架特性。 |
| 消息池（Message pool） | "共享聊天历史" | 全量历史共享状态。易于推理，扩展性差。 |
| 投影状态（Projected state） | "范围视图" | 共享状态的按角色视图。可扩展，需要模式设计。 |
| 发言者选择（Speaker selection） | "谁下一个发言" | 编排器模式，其中一个函数（通常是 LLM）从组中选择下一个智能体。 |

## 扩展阅读

-   [OpenAI 手册：编排智能体——例程与交接](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 交接驱动编排的最清晰阐述
-   [AutoGen 稳定版文档](https://microsoft.github.io/autogen/stable/) — GroupChat + 发言者选择是 LLM 选择编排的参考
-   [LangGraph 工作流与智能体](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 图边编排和基于归约器的共享状态
-   [CrewAI 介绍](https://docs.crewai.com/en/introduction) — 角色-目标-背景故事智能体，顺序/层级流程
-   [AG2（社区 AutoGen 延续）](https://github.com/ag2ai/ag2) — Microsoft 将 v0.4 移至维护模式后的活跃 AutoGen v0.2 分支
