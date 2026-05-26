# 编排模式：监督者、Swarm、层级式 (Orchestration Patterns: Supervisor, Swarm, Hierarchical)

> 四种编排模式在 2026 年框架中反复出现：监督者-工作者、swarm / 点对点、层级式、辩论。Anthropic 的指导："关键在于为你的需求构建正确的系统。"从简单开始；仅在单个 Agent 加五种工作流模式不足时才添加拓扑。

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 12 (工作流模式), Phase 14 · 25 (多 Agent 辩论)
**时间：** 约 60 分钟

## 学习目标

- 列举四种反复出现的编排模式及其各自适合的场景。
- 描述 2026 年 LangChain 建议：基于工具调用的监督 vs 监督者库。
- 解释 Anthropic 的"构建正确系统"规则及其如何门控拓扑选择。
- 用标准库在通用脚本化 LLM 上实现全部四种模式。

## 问题

团队在需要之前就伸手去拿"多 Agent"。四种模式在框架中反复出现；一旦你能命名它们，你就能选择正确的——或完全跳过拓扑。

## 概念

### 监督者-工作者

- 一个中心路由 LLM 分发给专家 Agent。
- 决定：循环回自身、交接给专家、终止。
- 专家之间不互相交谈；所有路由通过监督者。

框架：LangGraph `create_supervisor`、Anthropic 编排器-工作者、CrewAI Hierarchical Process。

**2026 年 LangChain 建议：** 通过直接工具调用而非 `create_supervisor` 进行监督。提供更精细的上下文工程控制——你精确决定每个专家看到什么。

### Swarm / 点对点

- Agent 通过共享工具接口直接交接。
- 没有中心路由器。
- 比监督者延迟更低（更少的跳数）。
- 更难推理（没有单一控制点）。

框架：LangGraph swarm 拓扑、OpenAI Agents SDK 交接（当所有 Agent 可以交接给所有其他 Agent 时）。

### 层级式

- 监督者管理子监督者管理工作器。
- 在 LangGraph 中实现为嵌套子图；在 CrewAI 中实现为嵌套 crew。
- 以运维复杂性为代价扩展到大型 Agent 群体。

何时需要：当单个监督者的上下文预算无法容纳所有专家的描述时。

### 辩论

- 并行提议者 + 迭代交叉批评 (第 25 课)。
- 不完全是编排——更像是验证——但在框架中作为拓扑选择出现。

### CrewAI Crew vs Flow

CrewAI 形式化了两种部署模式：

- **Flow** 用于确定性事件驱动自动化（推荐的生产起点）。
- **Crew** 用于自主基于角色的协作。

这与上述四种模式正交，但映射到拓扑：Flow 通常是监督者或层级式；Crew 通常是带 LLM 路由器的监督者。

### Anthropic 的指导

"LLM 领域的成功不在于构建最复杂的系统。而在于为你的需求构建正确的系统。"

决策顺序：

1. 单个 Agent + 工作流模式 (第 12 课) — 从这里开始。
2. 监督者-工作者 — 当你有 2-4 个专家时。
3. Swarm — 当延迟比推理清晰度更重要时。
4. 层级式 — 仅当监督者上下文预算失败时。
5. 辩论 — 当准确率比成本更重要时。

### 这个模式哪里会出错

- **拓扑优先思维。** "我们需要多 Agent"在识别多 Agent 解决什么问题之前。
- **Swarm 中的弹跳交接。** A -> B -> A -> B。使用跳数计数器。
- **假层级。** 三层因为"企业级"；实际只有两个团队。折叠。

## 构建

`code/main.py` 用标准库在脚本化 LLM 上实现全部四种模式：

- `Supervisor` — 中心路由器。
- `Swarm` — 带直接交接的点对点。
- `Hierarchical` — 监督者的监督者。
- `Debate` — 并行提议者 + 批评。

每种模式处理相同的三意图任务（退款 / bug / 销售）。追踪形态不同。

运行：

```
python3 code/main.py
```

输出：每种模式的追踪 + 操作计数。监督者最干净；swarm 最短；层级式最深；辩论最昂贵。

## 使用

- **LangGraph** 用于监督者和层级式（嵌套子图）。
- **OpenAI Agents SDK** 用于交接即工具（监督者形态）。
- **CrewAI Flow** 用于生产确定性。
- **自定义** 用于辩论或当你需要精确控制时。

## 交付

`outputs/skill-orchestration-picker.md` 选择拓扑并实现它。

## 练习

1. 通过移除路由器将监督者-工作者转换为 swarm。什么坏了？什么改进了？
2. 为 swarm 添加跳数计数器：3 次交接后拒绝。它能捕获 A->B->A 弹跳吗？
3. 为 12 个专家领域构建两级层级系统。上下文预算在哪里没有嵌套就失败？
4. 在生产形态的工作负载上对四种模式进行性能分析。每种模式在哪个指标上获胜（延迟、成本、准确率、可调试性）？
5. 阅读 Anthropic 的"Building Effective Agents"文章。将你的每个生产流程映射到四种模式之一。有没有不能干净映射的？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 监督者-工作者 (Supervisor-worker) | "路由器 + 专家" | 中心 LLM 分发给专家；它们不互相交谈 |
| Swarm | "点对点" | 通过共享工具直接交接；无中心路由器 |
| 层级式 (Hierarchical) | "监督者的监督者" | 用于大型群体的嵌套子图 |
| 辩论 (Debate) | "提议者 + 批评" | 并行提议者，交叉批评 (第 25 课) |
| 基于工具调用的监督 (Tool-call-based supervision) | "无库监督者" | 将监督者实现为直接工具调用以控制上下文 |
| Crew | "自主团队" | CrewAI 的基于角色协作模式 |
| Flow | "确定性工作流" | CrewAI 的事件驱动生产模式 |

## 扩展阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 五种模式 + Agent vs 工作流
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 监督者、swarm、层级式
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — 辩论模式
