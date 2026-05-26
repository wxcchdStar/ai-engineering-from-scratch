# Anthropic 的工作流模式：简单优于复杂 (Anthropic's Workflow Patterns: Simple Over Complex)

> Schluntz 和 Zhang (Anthropic, 2024 年 12 月) 区分了工作流（预定义路径）和 Agent（动态工具使用）。五种工作流模式覆盖大多数场景。从直接 API 调用开始。仅在步骤无法预测时才添加 Agent。

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 01 (Agent 循环)
**时间：** 约 60 分钟

## 学习目标

- 列举 Anthropic 的五种工作流模式：prompt 链式调用、路由、并行化、编排器-工作者、评估器-优化器。
- 解释 Agent 与工作流的区别及其各自的工程成本。
- 识别何时选择工作流而非 Agent（反之亦然）。
- 用标准库针对脚本化 LLM 实现全部五种模式。

## 问题

团队为只需要一个函数调用的问题引入多 Agent 框架。成本是真实的：框架增加了层次，模糊了 prompt，隐藏了控制流，并引入了过早的复杂性。Schluntz 和 Zhang 2024 年 12 月的文章是引用最多的行业反驳：从简单开始，仅在复杂性值得时才添加。

## 概念

### 工作流 vs Agent

- **工作流。** LLM 和工具通过预定义的代码路径编排。工程师拥有图。
- **Agent。** LLM 动态指导自己的工具并采取自己的步骤。模型拥有图。

两者各有其位。工作流更便宜、更快、更容易调试。Agent 解锁开放式问题，但使失败模式更难推理。

### 增强型 LLM

所有五种模式的基础：一个 LLM 接入三种能力——搜索（检索）、工具（动作）、记忆（持久化）。任何 API 调用都可以使用这些。

### 五种模式

1. **Prompt 链式调用。** 调用 1 的输出是调用 2 的输入。当任务有清晰的线性分解时使用。步骤之间可选程序化门控。

2. **路由。** 一个分类器 LLM 选择调用哪个下游 LLM 或工具。当不同类别的输入需要不同处理时使用（一级支持 vs 退款 vs bug vs 销售）。

3. **并行化。** 同时运行 N 个 LLM 调用，聚合结果。两种形态：分段（不同块）和投票（相同 prompt，N 次运行，多数/综合）。

4. **编排器-工作者。** 一个编排器 LLM 动态决定运行哪些工作者（也是 LLM）并综合其输出。类似 Agent 循环，但编排器不会无限循环。

5. **评估器-优化器。** 一个 LLM 提出答案，另一个 LLM 评估它。迭代直到评估器通过。这是 Self-Refine (第 05 课) 的泛化。

### 工作流优于 Agent 的场景

- **可预测的任务。** 如果你能枚举步骤，就应该枚举。
- **成本受限的任务。** 工作流有有限的步骤数；Agent 可能螺旋失控。
- **合规受限的任务。** 审计员想读图，而不是从轨迹中推断。

### Agent 优于工作流的场景

- **开放式研究。** 当下一步取决于上一步返回了什么。
- **可变长度的任务。** 从几分钟到几小时的工作，步骤数未知。
- **新领域。** 当你还不知道正确的工作流时——先探索，后固化。

### 上下文工程伴侣

"面向 AI Agent 的有效上下文工程" (Anthropic 2025) 形式化了相邻的学科：200k 窗口是一个预算，而非容器。包含什么、何时压缩、何时让上下文增长。在本课程重新编号前的 Phase 14 上下文压缩课程中详细讨论。

## 构建

`code/main.py` 针对 `ScriptedLLM` 实现全部五种工作流模式：

- `prompt_chain(input, steps)` — 顺序执行。
- `route(input, classifier, handlers)` — 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` — N 次运行，聚合。
- `orchestrator_workers(task, workers)` — 编排器选择工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — 循环直到通过。

运行：

```
python3 code/main.py
```

每种模式打印其追踪。每种模式的代码行数约为 10-15 行；框架的成本以千行计。

## 使用

- 大多数任务使用直接 API 调用。
- 仅在模式真正需要持久状态 (LangGraph)、actor 模型并发 (AutoGen v0.4) 或角色模板 (CrewAI) 时才使用框架。
- 当你想要 Claude Code harness 形态而不想自己重建时，使用 Claude Agent SDK。

## 交付

`outputs/skill-workflow-picker.md` 为给定的任务描述选择正确的模式，包括决策理由和当工作流不足时重构为 Agent 的路径。

## 练习

1. 实现带置信度阈值的路由。低于阈值 → 升级给人工。对于一级支持用例，阈值应该设在哪里？
2. 为 `parallel_vote` 添加超时。当一个调用挂起时会发生什么？如何在有缺失投票的情况下聚合？
3. 将 `evaluator_optimizer` 改为 bandit：在迭代中保留 top-2 输出，这样后期好的结果不会被后期差的结果覆盖。
4. 将 prompt 链式调用与路由组合：路由器选择三条链中的一条。测量 token 成本与单个大 prompt 替代方案的对比。
5. 选择你的一个生产功能。画出工作流图。数步骤。Agent 在这里真的会更好吗？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 工作流 (Workflow) | "预定义流程" | 工程师拥有的 LLM 和工具调用图 |
| Agent | "自主 AI" | 模型拥有的图；动态工具指导 |
| 增强型 LLM (Augmented LLM) | "带工具的 LLM" | LLM + 搜索 + 工具 + 记忆；原子单元 |
| Prompt 链式调用 (Prompt chaining) | "顺序调用" | 调用 N 的输出是调用 N+1 的输入 |
| 路由 (Routing) | "分类器分发" | 选择哪个链/模型处理输入 |
| 并行化 (Parallelization) | "扇出" | N 个并发调用；通过分段或投票聚合 |
| 编排器-工作者 (Orchestrator-workers) | "分发器 Agent" | 编排器 LLM 动态选择专家 LLM |
| 评估器-优化器 (Evaluator-optimizer) | "提议者 + 评判者" | 迭代直到评估器通过；Self-Refine 的泛化 |

## 扩展阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 五种工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 伴侣学科
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 当有状态图值得其成本时
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 编排器-工作者模式的产品化
