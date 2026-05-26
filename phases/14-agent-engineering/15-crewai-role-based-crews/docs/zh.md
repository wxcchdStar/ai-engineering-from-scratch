# CrewAI：基于角色的 Crew 与 Flow (CrewAI: Role-Based Crews and Flows)

> CrewAI 是 2026 年基于角色的多 Agent 框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crew（自主、基于角色的协作）和 Flow（事件驱动、确定性）。文档直言："对于任何生产就绪的应用，从 Flow 开始。"

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 12 (工作流模式), Phase 14 · 14 (Actor 模型)
**时间：** 约 75 分钟

## 学习目标

- 列举 CrewAI 的四个原语（Agent、Task、Crew、Process）及其各自拥有的内容。
- 区分 Sequential、Hierarchical 和 Consensual 流程；为每种工作负载选择合适的。
- 区分 Crew（自主基于角色）和 Flow（事件驱动确定性），并解释文档的生产建议。
- 使用 `@tool` 装饰器和 `BaseTool` 子类接入工具；推理结构化输出 vs 自由文本。
- 列举四种 CrewAI 记忆类型及其各自的回报场景。
- 用标准库实现一个三 Agent crew（研究员、写手、编辑）来生成简报。
- 识别 CrewAI 的三种失败模式：prompt 膨胀、管理 LLM 税、脆弱的交接。

## 问题

采用多 Agent 框架的团队会撞上同一堵墙。"自主协作"在演示中听起来很棒。然后客户提交了一个 bug，你需要确定性重放。或者财务问 LLM 路由的 crew 每次运行成本多少。或者 on-call 需要知道凌晨 3 点哪个 Agent 卡住了。

自由形式的 LLM 路由 crew 无法干净地回答这些问题。纯 DAG 可以回答所有问题，但失去了头脑风暴 Agent 需要的探索形态。

CrewAI 的拆分诚实地面对了这个权衡。Crew 用于协作、基于角色、探索性的工作。Flow 用于事件驱动、代码拥有、可审计的生产。同一个框架，两种形态，按场景选择。

## 概念

### 四个原语

CrewAI 的接口很小。记住这些，其余都是配置。

- **Agent。** `role + goal + backstory + tools + (可选) llm`。backstory 是关键。它塑造语气、判断、Agent 何时停止。工具是 Agent 可以调用的函数（详见下文）。
- **Task。** `description + expected_output + agent + (可选) context + (可选) output_pydantic`。一个可复用的工作单元。`expected_output` 是契约。`context` 列出上游任务，其输出会被传入。`output_pydantic` 强制结构化输出。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensual。选择运行的形态。

Agent 不直接看到彼此。Task 引用 Agent。Crew 排序 Task。Process 决定谁选择下一个 Task。这就是整个心智模型。

### Sequential vs Hierarchical vs Consensual

- **Sequential。** Task 按声明顺序运行。Task N 的输出作为 `context` 对 Task N+1 可用。成本最低。最可预测。当顺序固定时使用。
- **Hierarchical。** 一个管理 Agent（单独的 LLM 调用）在专家之间路由。CrewAI 从你的 `manager_llm` 配置或默认值生成管理器。管理器每轮选择下一个 Task，可以拒绝或重新路由。当你有四个或更多专家且顺序真正取决于先前输出时使用。
- **Consensual。** Beta。Agent 对下一步投票。在研究之外很少值得额外的往返。

Hierarchical 在每次专家调用之上增加每轮一次 LLM 调用（管理器）。在五步运行中 token 成本可能翻三倍。仅在需要路由时才为此付费。

### Crew vs Flow

这是 2026 年文档开篇的框架。

- **Crew。** LLM 驱动的自主性。框架在运行时选择形态。适用于：研究、头脑风暴、初稿，任何路径本身就是答案一部分的场景。难以重放。难以测试。原型开发便宜。
- **Flow。** 你拥有的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记一个步骤，当另一个步骤发出该 topic 时触发。每个步骤是纯 Python（可以在内部调用 Crew）。适用于：生产。可观测。可测试。确定性。

文档 2026 年的生产建议：从 Flow 开始。当自主性值得其成本时，将 Crew 作为 Flow 步骤中的 `Crew.kickoff()` 调用折叠进去。Flow 给你审计追踪，Crew 给你探索。组合，不要二选一。

### 工具集成

三种给 Agent 工具的方式。选择适合的最简单方式。

1. **`@tool` 装饰器。** 纯函数变成工具。签名是 schema；docstring 是 LLM 看到的描述。最适合一次性辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 基于类的工具，带显式参数 schema、异步支持、重试。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 提供第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一次导入即可接入。

结构化输出使用 Pydantic。在 Task 上传递 `output_pydantic=MyModel`。CrewAI 根据模型验证 LLM 响应，要么强制转换要么重试。将其与紧凑的 `expected_output` 字符串配对。自由文本输出适合草稿；结构化输出是下游 Flow 可以消费的。

### 记忆钩子

CrewAI 开箱即用四种记忆类型。它们可以组合：一个 Crew 可以同时启用全部四种。

- **短期。** 单次运行内的对话缓冲区。运行结束时清除。
- **长期。** 跨运行持久化。存储在向量数据库中（默认 Chroma，可替换）。按与当前任务的相似度检索。
- **实体。** 每个实体的事实。"客户 X 在企业计划中。"按键为实体，而非按相似度。跨运行存活。
- **上下文。** 组装时检索。在 Agent 需要时拉取相关记忆，而非预加载。

在 Crew 上通过 `memory=True` 或按类型配置启用。由你配置的嵌入提供商支持（默认 OpenAI，可替换为本地）。记忆是 CrewAI 相对于更轻量框架的价值所在之一；纯 LangGraph 需要你自己接入每一个。

### CrewAI 适合的场景

- 三到六个 Agent，具有命名角色和协作工作流。起草、审查、规划、头脑风暴。
- LLM 对下一步的判断是价值一部分的路由（Hierarchical）。
- 任何团队更喜欢阅读 `role + goal + backstory` 而非阅读图定义的场景。

### CrewAI 不适合的场景

- 具有严格顺序的确定性 DAG。使用 LangGraph (第 13 课)。图形状是正确的抽象；CrewAI 的角色框架是摩擦。
- 亚秒级延迟预算。Hierarchical 增加往返。即使 Sequential 也会序列化包含 backstory 和先前输出的 prompt。
- 单 Agent 循环。跳过框架；一个 Agent 循环 (第 1 课) 加一个工具注册表更短。

第 17 课 (Agent 框架权衡) 以矩阵形式展开。简短版本：CrewAI 位于"协作基于角色"的角落。

### 依赖形态

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。2026 年初 30k+ GitHub stars。AWS Bedrock 集成有文档记录；其基准测试声称在 QA 任务上比 LangGraph 快 5.76 倍。将框架供应商的数字视为方向性的。

### 这个模式哪里会出错

- **backstory 导致的 prompt 膨胀。** 每个 Agent 2000 字的 backstory 和五个 Agent 的 crew 在第一次工具调用前就烧光了上下文预算。将 backstory 保持在 200 字以下。跨 Agent 复用短语；不要重复五遍风格指南。
- **管理 LLM token 税。** Hierarchical 流程在每次专家调用前增加一次管理 LLM 调用。在五任务 crew 中这是六次 LLM 调用而非五次，且管理调用携带完整任务列表加先前输出。除非路由取决于输出，否则切换到 Sequential。
- **脆弱的交接。** Task N 的 `expected_output` 是"一个大纲"。Task N+1 将其作为 `context` 读取并尝试解析三个部分。LLM 产生了四个。下游 Agent 即兴发挥。通过在 Task N 上使用 `output_pydantic` 修复，使 Task N+1 读取类型化对象而非自由文本。
- **Crew 直接上生产。** 自由形式的 Crew 在没有 Flow 包装的情况下发布到生产。输出变异性高；重放不可能；on-call 无法对比坏运行和好运行。用 Flow 包装。

## 构建

`code/main.py` 用标准库实现两种形态加一个三 Agent crew。

形态：

- `Agent`、`Task` 数据类，匹配 CrewAI 的接口。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，将输出作为 `context` 传递。
- `HierarchicalCrew.kickoff(topic)` 添加一个管理 Agent，每轮选择下一个专家，在 "done" 时停止。
- `Flow` 带 `@start` 和 `@listen(topic)` 装饰器、一个小型事件循环和追踪。
- `tool(name)` 装饰器，镜像 CrewAI 的 `@tool` 形态。
- `Memory` 带 `short_term`、`long_term`、`entity` 存储；模拟相似度使用 numpy。
- 模拟 LLM 响应是基于角色加输入前缀的硬编码字符串。无网络。确定性。

具体演示：研究员、写手、编辑 crew 生成关于 "agent engineering 2026" 的简报。研究员拉取（模拟的）来源。写手起草。编辑精简。同一个 crew 通过 Flow 运行以展示确定性形态。

运行：

```bash
python3 code/main.py
```

追踪覆盖：sequential crew 通过 `context` 传递输出，hierarchical crew 带管理选择（研究员、写手、编辑，然后 "done"），flow 以显式 topic（`researched`、`drafted`、`edited`）运行相同的三个步骤，通过 `@tool` 路由的工具调用，以及跨两次 kickoff 存活的长期记忆。

Crew 追踪是流动的；管理器原则上可以重新排序。Flow 追踪是固定的。这个选择就是本课的核心。

## 使用

- **CrewAI Flow** 用于生产。即使 Flow 只有一个步骤调用 `Crew.kickoff()`。Flow 提供审计边界。
- **CrewAI Crew (Sequential)** 用于顺序清晰的协作工作，特别是初稿和审查循环。
- **CrewAI Crew (Hierarchical)** 当路由取决于输出且有四个或更多专家时。
- **LangGraph** (第 13 课) 用于显式状态机、持久化恢复、严格排序。
- **AutoGen v0.4** (第 14 课) 用于 actor 模型并发和故障隔离。
- **OpenAI Agents SDK** (第 16 课) 用于 OpenAI 优先的产品，带交接和护栏。
- **Claude Agent SDK** (第 17 课) 用于 Claude 优先的产品，带子 Agent 和会话存储。

## 交付

`outputs/skill-crew-or-flow.md` 为任务选择 Crew 或 Flow 并搭建最小实现。硬拒绝：没有 backstory 的 Crew、没有显式 topic 的 Flow、少于三个专家的 Hierarchical。

## 陷阱

- **Backstory 作为调味料。** 它塑造输出。每个 Agent 测试三个变体；差异是真实的。选一个，冻结它。
- **跳过 `expected_output`。** 没有每个任务的契约，下游任务接收 LLM 产生的任何内容。Crew 运行了；审计失败了。
- **记忆始终开启。** 长期记忆每次运行都写入。向量数据库增长。检索变得嘈杂。将写入范围限定在事实是持久性的任务上。
- **管理 prompt 漂移。** Hierarchical 的管理 prompt 是隐式的。如果路由变得奇怪，在 verbose 模式下转储并阅读。
- **Crew 中的工具副作用。** 一个 Crew 可能比预期调用工具更多次。POST、DELETE、支付应放在 Flow 步骤中，永远不要放在 Crew 工具中。

## 练习

1. 将 Sequential crew 转换为 Flow。数一下变异性下降的接触点。注意可读性下降的地方。
2. 为 crew 添加实体记忆：关于客户的事实跨 kickoff 持久化。验证检索拉取了正确的实体。
3. 实现一个 Hierarchical 流程，其中管理器拒绝路由到编辑，直到写手的输出至少有三个段落。追踪重试。
4. 为一个（模拟的）网络搜索接入 `BaseTool` 子类。比较追踪形态与 `@tool` 装饰器版本。
5. 为编辑任务添加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让写手任务输出一次格式错误的 JSON；在追踪中验证 CrewAI 的重试行为。
6. 阅读 CrewAI 的文档介绍。将玩具移植到真实的 `crewai` API。标准库版本跳过了哪些保证？
7. 将 AgentOps 或 Langfuse (第 24 课) 接入真实运行。标准库版本中你错过了哪些追踪？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Agent | "角色" | Role + goal + backstory + tools |
| Task | "工作单元" | Description + expected output + 负责人 + 可选结构化输出 |
| Crew | "Agent 团队" | Agent + Task + Process 的容器 |
| Process | "执行策略" | Sequential / Hierarchical / Consensual |
| Flow | "确定性工作流" | 事件驱动、代码拥有、可测试 |
| Backstory | "角色 prompt" | 塑造 Agent 的语气和判断 |
| `@tool` | "函数工具" | 将函数变成 Agent 可调用工具的装饰器 |
| `BaseTool` | "类工具" | 基于类的工具，带参数 schema、重试、异步支持 |
| 实体记忆 (Entity memory) | "每个实体的事实" | 范围限定在客户/账户/问题的记忆 |
| 长期记忆 (Long-term memory) | "跨运行记忆" | 向量支持的记忆，在 kickoff 之间存活 |
| 上下文记忆 (Contextual memory) | "即时检索" | 在 Agent 需要时拉取的记忆 |
| 管理 LLM (Manager LLM) | "路由 Agent" | Hierarchical 流程中选择下一个 Task 的额外 LLM |
| `expected_output` | "任务契约" | 告诉 Agent（和审计）返回什么形状的字符串 |

## 扩展阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：概念和推荐的生产路径
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：事件驱动形态，`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具包
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：多 Agent 何时有帮助，何时没有
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案
