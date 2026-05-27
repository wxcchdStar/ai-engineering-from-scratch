# 自主编码智能体格局（2026）

> SWE-bench Verified 在不到三年内从 4% 上升到 80.9%。相同的 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline 自主模式下得分 59.8%——模型周围的脚手架现在与模型本身一样重要。OpenHands（前身为 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环在沙箱中直接执行 Python 操作，而不是 JSON 工具调用。头条数字隐藏了一个方法论问题：500 个 SWE-bench Verified 任务中有 161 个只需要 1-2 行更改，而 SWE-bench Pro（10+ 行任务）对相同的前沿模型仅为 23-59%。

**类型：** 学习
**语言：** Python（标准库，CodeAct vs JSON 工具调用比较）
**前置条件：** 第 14 阶段 · 07（工具使用），第 15 阶段 · 01（长周期智能体）
**时间：** 约 45 分钟

## 问题

"哪个编码智能体最好"是错误的问题。正确的问题是：在匹配我工作的任务分布上，使用我将在生产中运行的脚手架，我得到什么端到端可靠性？

在 2022 年到 2026 年之间，该领域认识到脚手架——检索层、规划器、沙箱、编辑-验证循环、反馈格式——是承重的。Claude Sonnet 4.5 在 SWE-agent v1 上在 SWE-bench Verified 上得分 43.2%；相同模型在 Cline 的自主脚手架内得分 59.8%。16.6 个绝对百分点的差异，相同的权重。基础模型是一个组件；循环是产品。

伴随的问题是基准饱和隐藏了回归。SWE-bench Verified 接近饱和，简单任务尾部（500 个任务中有 161 个需要 ≤2 行）拉高了顶部分数。真实世界质量更好地在像 SWE-bench Pro（10+ 行更改）这样的分布上测量，其中相同的领先者仍然在 23-59%。

## 概念

### SWE-bench，一段话讲清楚

SWE-bench（Jimenez 等人）取真实的 GitHub issue 及其 ground-truth 补丁，要求智能体生成一个使测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是一个人工策划的 500 任务子集，移除了模糊和损坏的任务。SWE-bench Pro 是更难的继任者——需要 10+ 行更改的任务，当前前沿智能体在 23-59%。

### 2022 → 2026 曲线实际显示了什么

- **2022**：研究模型在原始 SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin 风格脚手架约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 内推入 40-55% 范围。
- **2026**：Claude Sonnet 4.5 和前沿竞争者在 SWE-bench Verified 上达到 70-80%+。Epoch AI 的排行榜实时跟踪。

斜率来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）和更好的基准测试（Verified 移除噪声）。

### CodeAct vs JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前身为 OpenDevin）做了一个特定的架构赌注：模型不发出主机解码和执行的 JSON 工具调用，而是发出 Python 代码，Jupyter 风格的内核在沙箱中运行它。智能体可以在一个操作中循环文件、链式工具并捕获自己的异常。

权衡：

- **JSON 工具调用**：每个操作是一轮；易于审计；有限的组合性；默认安全，因为每个调用通过显式验证器。
- **CodeAct**：一个操作可以是整个程序；组合性强；需要加固的沙箱（OpenHands 使用 Docker 隔离）；失败模式包括沙箱运行时允许的任何内容。

两种架构都在生产中。CodeAct 在开放平台中占主导（OpenHands、smolagents）。JSON 工具调用在托管服务中仍然占主导（Anthropic Managed Agents、OpenAI Assistants），其中提供商控制执行器。

### 2026 年格局中的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著属性 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃的开放平台；事件流可重放 |
| SWE-agent | MIT | 智能体-计算机接口 (ACI) | 第一个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 本地仓库中的 diff 编辑 | 最小脚手架，强大的回归稳定性 |
| Cline | Apache-2 | 带工具策略的 VS Code 智能体 | Sonnet 4.5 上得分最高的开放脚手架 |
| Devin (Cognition) | 专有 | 托管 VM + 规划器 | 第一个"AI 软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细涵盖智能体循环 |

### 为什么脚手架占主导

编码运行是一个长周期轨迹（第 1 课）。可靠性在步骤间复利。脚手架获得分数的三个地方：

1. **检索**：找到正确的文件来阅读是静默瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的仓库地图都攻击这一点。
2. **验证器循环**：运行测试、读取堆栈跟踪并重新尝试在 SWE-bench 上是 10+ 点的增量。
3. **失败遏制**：在错误时回滚的沙箱防止复合损害。相同模型有和没有验证器循环看起来像两个不同的产品。

### 基准饱和与真实分布

OpenHands 作者和 Epoch AI 都指出 SWE-bench Verified 有一个简单尾部：500 个任务中有 161 个只需要 1-2 行更改。高分部分由这个尾部驱动。SWE-bench Pro 限制为 10+ 行更改，即使对前沿系统也返回 23-59% 范围的分数。你的生产分布几乎肯定更接近 Pro 而不是 Verified。

选择智能体的含义：在你自己的 bug 积压上运行一个类似 Pro 的子集。重要的分数是在代表你交付内容的任务上的分数。

## 使用它

`code/main.py` 在固定的迷你任务分布上比较两个玩具智能体脚手架：

1. 一个每轮执行一个操作的 **JSON 工具调用**脚手架。
2. 一个每个操作可以发出小段 Python 代码的 **CodeAct**脚手架。

两者都使用存根"模型"（确定性规则），因此比较将脚手架与模型质量隔离。输出显示 CodeAct 脚手架以更少的轮次解决更多任务，代价是更大的每次操作爆炸半径。

## 交付它

`outputs/skill-scaffold-audit.md` 帮助你在采用之前审计一个提议的编码智能体脚手架：检索质量、验证器存在、沙箱隔离和基准到分布的适配。

## 练习

1. 运行 `code/main.py`。每个脚手架在相同任务集上需要多少轮？每个的每次操作爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文论证 CodeAct 在复杂任务上优于 JSON 工具调用。确定论文承认的一个失败模式，并写一句话说明该模式在生产中何时会占主导。

3. 从你的 bug 积压中选择一个需要跨两个文件 10+ 行更改的任务。估计前沿模型在（a）JSON 工具调用和（b）CodeAct 下的端到端成功概率。证明差距。

4. SWE-bench Verified 有 161 个单文件、1-2 行任务。构造一个排除它们的分数。排行榜如何重新排列？

5. 阅读"Introducing SWE-bench Verified"（OpenAI）。解释用于移除模糊任务的具体方法论，并命名策划会错过的一个类别。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| SWE-bench | "编码基准" | 带有 ground-truth 补丁和测试套件的真实 GitHub issue |
| SWE-bench Verified | "清理后的子集" | 500 个人工策划任务，存在简单尾部 |
| SWE-bench Pro | "更难的子集" | 10+ 行更改；前沿在 23-59% |
| CodeAct | "代码即操作" | 智能体发出 Python；Jupyter 风格内核在沙箱中执行 |
| JSON 工具调用 | "函数调用" | 每个操作是在执行前验证的结构化 JSON 负载 |
| 脚手架 | "智能体框架" | 基础模型周围的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（智能体-计算机接口） | "SWE-agent 的格式" | 为 LLM 人机工程学设计的命令集，而非人类 shell |
| 验证器循环 | "测试并重试" | 运行测试，读取输出，修改补丁；最大的非模型可靠性增益 |

## 扩展阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始基准和方法论。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策划子集是如何构建的。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构和事件流设计。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时跟踪的分数。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 长周期编码智能体可靠性框架。
