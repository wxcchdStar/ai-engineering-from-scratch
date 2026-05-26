# 评估驱动的 Agent 开发 (Eval-Driven Agent Development)

> Anthropic 的指导："从简单的 prompt 开始，通过全面的评估进行优化，仅在需要时添加多步 Agent 系统。"评估不是最后一步。它是驱动 Phase 14 中每个其他选择的外部循环。

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 全部。
**时间：** 约 60 分钟

## 学习目标

- 列举三个评估层——静态基准、自定义离线、在线生产——及其各自的用途。
- 解释评估器-优化器紧密循环。
- 描述 2026 年最佳实践：评估与代码共存、在 CI 中运行、门控 PR。
- 将每个 Phase 14 课程连接到它产生的评估用例。

## 问题

Agent 通过演示。它们以演示无法预测的方式在生产中失败。基准回答"这个模型在广泛意义上是否胜任？"而非"这个 Agent 是否为我的产品发布了正确的补丁？"答案：三层评估，持续运行，每个护栏和学习到的规则都映射到一个评估用例。

## 概念

### 三个评估层

1. **静态基准** — SWE-bench Verified 用于代码 (第 19 课)，WebArena/OSWorld 用于浏览/桌面 (第 20 课)，GAIA 用于通用 (第 19 课)，BFCL V4 用于工具使用 (第 06 课)。用于跨模型比较和回归门控。污染是真实的：SWE-bench+ 发现 32.67% 的解决方案泄露。始终报告 Verified / +-审计分数。

2. **自定义离线评估** — 你产品的形态：
   - LLM 作为评判者（Langfuse、Phoenix、Opik — 第 24 课）。
   - 基于执行（运行补丁，检查测试）。
   - 基于轨迹（将动作序列与黄金标准比较；OSWorld-Human 显示顶级 Agent 超过黄金标准 1.4-2.7 倍）。

3. **在线评估** — 生产：
   - 会话重放（Langfuse）。
   - 护栏触发的告警（第 16、21 课）。
   - 每步成本/延迟追踪（第 23 课 OTel span）。

### 评估器-优化器 (Anthropic)

紧密循环：

1. 提议者生成输出。
2. 评估者评判。
3. 优化直到评估者通过。

这是 Self-Refine (第 05 课) 的泛化。你关心的任何 Agent 流程都可以包装在评估器-优化器中以提高可靠性。

### 2026 年最佳实践

- 评估与代码共存。
- 在每个 PR 上通过 CI 运行。
- 按评估分数门控合并（例如"与 main 相比回归不超过 5%"）。
- 每个护栏映射到一个评估用例。
- 每个学习到的规则（Reflexion、pro-workflow learn-rule）映射到一个失败用例。

### 连接 Phase 14

Phase 14 中的每课都产生评估用例：

| 课程 | 它产生的评估用例 |
|------|-----------------|
| 01 Agent 循环 | 预算耗尽、无限循环守卫 |
| 02 ReWOO | 工具失败时规划器正确重新规划 |
| 03 Reflexion | 学习到的反思在重试时应用 |
| 05 Self-Refine/CRITIC | 评判者通过优化后的输出 |
| 06 工具使用 | 参数强制转换有效；未知工具被拒绝 |
| 07-10 记忆 | 检索引用匹配来源；过时事实失效 |
| 12 工作流模式 | 每种模式产生正确输出 |
| 13 LangGraph | 恢复精确复现状态 |
| 14 AutoGen Actor | DLQ 捕获崩溃的处理器 |
| 16 OpenAI Agents SDK | 护栏在正确的输入上触发 |
| 17 Claude Agent SDK | 子 Agent 结果返回给编排器 |
| 19-20 基准 | SWE-bench Verified 分数、WebArena 成功率、OSWorld 效率 |
| 21 计算机使用 | 每步安全捕获注入的 DOM |
| 23 OTel | Span 发出必需属性 |
| 26 失败模式 | 检测器标记已知失败 |
| 27 Prompt 注入 | PVE 拒绝投毒的检索 |
| 28 编排 | 监督者路由到正确的专家 |
| 29 运行时形态 | DLQ 处理 N% 失败 |

如果你的评估套件对每个都有用例，你就覆盖了 Phase 14。

### 评估驱动开发在哪里失败

- **没有基线。** 没有上次已知良好状态的评估是不可读的。存储基线。
- **没有锚定的 LLM 评判者。** 评判者也会幻觉。CRITIC 模式 (第 05 课) — 评判者锚定在外部工具上。
- **过度拟合评估。** 为评估优化会偏离生产实用性。轮换用例。
- **不稳定的评估。** 非确定性用例导致误报。固定种子，快照状态。

## 构建

`code/main.py` 是一个标准库评估 harness：

- 带类别（基准、自定义、在线）的用例注册表。
- 一个被测的脚本化 Agent。
- 评估器-优化器循环：提议、评判、优化直到通过或达到最大轮次。
- CI 门控：聚合通过率 + 相对于基线的回归。

运行：

```
python3 code/main.py
```

输出：每个用例的通过/失败、回归标志、CI 门控裁决。

## 使用

- 在与 Agent 代码相同的仓库中编写评估用例。
- 在每个 PR 上通过 CI 运行它们。
- 在回归时使构建失败。
- 随时间追踪通过率。
- 将每个生产失败绑定到一个新用例。

## 交付

`outputs/skill-eval-suite.md` 为 Agent 产品构建三层评估套件，带 CI 门控和回归追踪。

## 练习

1. 取一个你的生产失败。编写一个复现它的评估用例。你的 Agent 现在能通过吗？
2. 为你的领域构建一个三维度的 LLM 评判评分标准（事实性、语气、范围）。评分 50 个会话。
3. 将评估套件接入 CI。在 >=5% 回归时使构建失败。
4. 添加轨迹效率指标：Agent 与黄金轨迹相比用了多少步？
5. 将每个 Phase 14 课程映射到你的套件中的一个评估用例。有缺失的吗？那是需要弥补的差距。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 静态基准 (Static benchmark) | "现成评估" | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| 自定义离线评估 (Custom offline eval) | "领域评估" | 在你产品形态上的 LLM 评判 / 执行 / 轨迹 |
| 在线评估 (Online eval) | "生产评估" | 会话重放、护栏告警、成本/延迟追踪 |
| 评估器-优化器 (Evaluator-optimizer) | "提议-评判-优化" | 迭代直到评判者通过 |
| CI 门控 (CI gate) | "合并阻止器" | 在评估回归时使构建失败 |
| 基线 (Baseline) | "上次已知良好" | 用于检测回归的参考分数 |
| 轨迹效率 (Trajectory efficiency) | "步骤超过黄金" | Agent 步骤数除以人类专家最小值 |

## 扩展阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "从简单开始，用评估优化"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策展基准
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — 工具使用基准
- [Langfuse docs](https://langfuse.com/) — 实践中的评估 + 会话重放
