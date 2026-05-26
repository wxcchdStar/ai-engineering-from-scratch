# ReWOO 与 Plan-and-Execute：解耦式规划

> ReAct 在一个流中将思考和行动交错进行。ReWOO 将它们分离：先一次性制定完整计划，然后执行。token 用量减少 5 倍，HotpotQA 准确率 +4%，且你可以将规划器蒸馏到 7B 模型。Plan-and-Execute 使其通用化；Plan-and-Act 将其扩展到网页导航。

**类型：** 构建
**语言：** Python（标准库）
**前置课程：** Phase 14 · 01（Agent 循环）
**时长：** 约 60 分钟

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 拆分比 ReAct 的交错循环节省 token 并提高鲁棒性。
- 实现一个计划 DAG、一个依赖排序执行器和一个组合 worker 输出的求解器——全部使用标准库。
- 决定任务何时应以先计划后执行 vs 交错 ReAct 的方式运行，使用 2026 年"五种工作流模式"框架（Anthropic）。
- 识别 Plan-and-Act 的合成计划数据何时是长视距网页或移动端任务所必需的。

## 问题背景

ReAct 的交错 thought-action-observation 循环简单灵活，但每个工具调用都必须携带完整的先前上下文——包括每一个先前的思考。token 使用量随深度呈二次增长。更糟的是：当工具在循环中途失败时，模型必须从错误观测中重新推导整个计划。

ReWOO（Xu 等人，arXiv:2305.18323，2023 年 5 月）注意到这一点并提出一个方案：一次性规划全部，并行获取证据，最后组合答案。一次 LLM 调用做规划，N 次工具调用获取证据（可并行），一次 LLM 调用做求解。代价是灵活性降低（计划是静态的），换来更好的 token 效率和更清晰的故障模式。

## 核心概念

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        （工具调用，可能并行）
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 生成一个 DAG。每个节点命名一个工具、其参数以及它依赖的哪些先前节点（如 `#E1`、`#E2` 的引用）。Workers 按拓扑顺序执行节点。Solver 将所有内容拼接在一起。

### 为什么节省 5 倍 token

ReAct 的提示词长度随步数线性增长。在第 10 步时，提示词包含：思考 1 + 行动 1 + 观测 1 + 思考 2 + 行动 2 + 观测 2，依此类推。每个中间步骤还冗余地包含原始提示词。

ReWOO 支付一次 planner 提示词（较大），N 次小型 worker 提示词（每次仅工具调用，无链），以及一次 solver 提示词。在 HotpotQA 上，论文测量到约 5 倍更少的 token，同时绝对准确率 +4。

### 为什么更鲁棒

如果 worker 3 在 ReAct 中失败，循环必须在中途从错误中推理恢复。在 ReWOO 中，worker 3 返回一个错误字符串；solver 在原始计划上下文中看到它，可以优雅降级。故障定位是逐节点的，而非逐步的。

### Planner 蒸馏

论文的第二个结果：因为 planner 不看到观测结果，你可以用 175B 教师模型的 planner 输出来微调一个 7B 模型。小模型处理规划；大模型在推理时不需要。这现在是标准做法——许多 2026 年生产 agent 使用小型 planner 加大型 executor，反之亦可。

### Plan-and-Execute（LangChain，2023）

LangChain 团队 2023 年 8 月的博文将 ReWOO 推广为一个模式名称：Plan-and-Execute。前置 planner 发出步骤列表，executor 运行每一步，可选的 replanner 可以在观测结果后修订计划。这比 ReWOO 更接近 ReAct（replanner 将观测结果带回规划中），但保留了 token 节省。

### Plan-and-Act（Erdogan 等人，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将该模式扩展到长视距网页和移动端 agent。关键贡献是合成计划数据：一个标注轨迹生成器产生计划是显式的训练数据。用于微调能够在 WebArena 类任务上持续工作 30-50 步以上的 planner 模型，而单条 ReAct 轨迹会失去连贯性。

### 如何选择

| 模式 | 适用场景 |
|------|----------|
| ReAct | 短任务、环境未知、需要反应式异常处理 |
| ReWOO | 结构化任务、已知工具、对 token 敏感、可并行化证据 |
| Plan-and-Execute | 类似 ReWOO 但在部分执行后有重新规划 |
| Plan-and-Act | 长视距（>30 步）、网页/移动端/computer-use |
| Tree of Thoughts | 搜索值得付出代价时（第 04 课） |

Anthropic 2024 年 12 月指南：从最简单的开始。如果任务是一个工具调用加一个摘要，不要构建 ReWOO。如果任务是一个 40 步的研究任务，不要只用 ReAct。

## 构建

`code/main.py` 实现了一个玩具 ReWOO：

- `Planner`——一个脚本化策略，从提示词发出计划 DAG。
- `Worker`——通过注册表分发每个节点的工具调用。
- `Solver`——读取证据并产生最终答案的脚本化组合。
- 依赖解析——如 `#E1` 的引用被替换为先前的 worker 输出。

演示回答了"法国的首都是什么，人口是多少，四舍五入到百万？"使用两步计划：(1) 查找首都，(2) 查找人口，然后求解。

运行：

```
python3 code/main.py
```

轨迹先显示完整计划，然后是 worker 结果，最后是 solver 组合。将 token 计数（我们打印粗略的字符计数）与 ReAct 风格交错运行比较——ReWOO 在这类结构化任务上胜出。

## 使用

LangGraph 将 Plan-and-Execute 作为一个配方提供（`create_react_agent` 用于 ReAct，自定义图用于 plan-execute）。CrewAI 的 Flows 直接编码该模式：你预先定义任务，Flow DAG 执行它们。Plan-and-Act 的合成数据方法仍主要属于研究领域；其运行时模式（显式计划 DAG）通过 LangGraph 和 CrewAI Flows 在生产中交付。

## 交付产物

`outputs/skill-rewoo-planner.md` 从用户请求生成一个 ReWOO 计划 DAG，给定工具目录。它在交给执行器之前验证计划（无环、每个引用可解析、每个工具存在）。

## 练习

1. 对独立的计划节点并行化 worker 执行。在具有 2 个并行组的 6 节点 DAG 上，它能带来什么好处？
2. 添加一个在任意 worker 返回错误时触发的 replanner 节点。将 ReWOO 变为 Plan-and-Execute 的最小改动是什么？
3. 将 `Planner` 替换为小模型（7B 级别），将 `Solver` 保留在 frontier 模型上。比较端到端质量——拆分在哪里失败？
4. 阅读 ReWOO 论文第 4 节关于 planner 蒸馏的内容。概念上复现 175B -> 7B 的结果：你需要什么训练数据，以及如何评分计划质量？
5. 将玩具移植到 Plan-and-Act 的轨迹形态：计划是一个序列，而非 DAG。哪些权衡发生了变化？

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| ReWOO | "没有观测的推理" | 先计划，然后并行获取证据，最后求解——规划提示词中没有观测结果 |
| Plan-and-Execute | "LangChain 的计划-执行模式" | 在 ReWOO 的基础上增加一个可选的执行后 replanner 节点 |
| Plan-and-Act | "规模化计划-执行" | 显式的 planner/executor 拆分，配有用于长视距任务的合成计划训练数据 |
| Evidence reference | "#E1, #E2, ..." | 计划节点占位符，在分发时替换为先前的 worker 输出 |
| Planner distillation | "小型规划器，大型执行器" | 在大型教师模型的 planner 轨迹上微调小模型 |
| Token efficiency | "更少的往返次数" | 论文中在 HotpotQA 上比 ReAct 节省 5 倍 token |
| DAG executor | "拓扑分发器" | 按依赖顺序运行计划节点；每层并行 |

## 延伸阅读

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — 经典论文
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — 带合成计划数据的规模化 planner-executor
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — 框架配方
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 选择最简单可行的模式