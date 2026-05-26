# Tree of Thoughts 与 LATS：审慎搜索

> 单一的思维链轨迹没有回溯的空间。ToT（Yao 等人，2023）将推理转化为一棵树，每个节点带有自我评估。LATS（Zhou 等人，2024）在蒙特卡洛树搜索 (MCTS) 下将 ToT、ReAct 和 Reflexion 统一。Game of 24 从 4%（CoT）提升到 74%（ToT）；LATS 在 HumanEval 上达到 92.7% pass@1。

**类型：** 构建
**语言：** Python（标准库）
**前置课程：** Phase 14 · 01（Agent 循环），Phase 14 · 03（Reflexion）
**时长：** 约 75 分钟

## 学习目标

- 将推理框架化为搜索：节点是"思考"，边是"扩展"，价值是"有多有前途"。
- 实现一个带有自我评估评分的标准库 ToT 风格 BFS 树搜索。
- 扩展为一个带有 select / expand / simulate / backpropagate 的玩具 LATS MCTS 循环。
- 决定何时搜索值得额外的 token 倍率（Game of 24、代码生成）以及何时单条轨迹就足够（简单问答）。

## 问题背景

思维链是线性路径。如果第一步错了，随后的每一步都建立在错误的前提上。在 Game of 24（用四个数字通过 + − × ÷ 算出 24）上，GPT-4 CoT 仅达到 4% 准确率。模型在早期选错了子表达式且无法恢复。

推理所需要的，是能够提出多个候选方案、评估它们、选择有前途的、并在遇到死胡同时回溯的能力。这就是搜索。Tree of Thoughts 和 LATS 是两种经典表述。

## 核心概念

### Tree of Thoughts（Yao 等人，NeurIPS 2023）

每个节点是一个连贯的中间步骤（"一个思考"）。每个节点可以扩展到 K 个子思考。LLM 通过评分提示词自我评估每个节点。搜索探索这棵树——BFS、DFS 或束搜索。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- 分数：高
              /   \              |                  |
          ...    ...          ...                finish
```

自我评估是关键。论文展示了三种变体：`sure/likely/impossible` 分类、`1..10` 数字评分、以及候选方案投票。三种都在 Game of 24 上大幅超越 CoT（使用 GPT-4 从 4% 到 74%）。

### LATS（Zhou 等人，ICML 2024）

LATS 在 MCTS 下统一了 ToT、ReAct 和 Reflexion。LLM 扮演三种角色：

- **Policy**：提出候选下一步行动（ReAct 风格）。
- **Value function**：对部分轨迹评分（ToT 风格自我评估）。
- **Self-reflector**：失败时写一段自然语言反思（Reflexion 风格），并用它为未来的 rollout 重新播种。

环境反馈（观测结果）混合到价值函数中，因此搜索不仅由模型意见通知，还由真实工具结果通知。论文当时的成果：GPT-4 在 HumanEval pass@1 达到 92.7%（SOTA），GPT-3.5 在 WebShop 平均达到 75.9（接近基于梯度的微调）。

### MCTS 简述

每次迭代的四个阶段：

1. **Select**——使用 UCT（树的上置信界）从根走到叶子节点。
2. **Expand**——通过 policy 生成 K 个子节点。
3. **Simulate**——从子节点使用 policy 做 rollout，用价值函数（或环境奖励）对叶子评分。
4. **Backpropagate**——沿路径向上更新访问计数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是 exploitation；第二项是 exploration。c 按任务调整。

### 成本现实

搜索会大幅增加 token。ToT 在 Game of 24 上使用了 CoT 的 100-1000 倍 token。LATS 类似。这不是免费的；将搜索保留给：

- 单条轨迹明显不够的任务（Game of 24、复杂代码）。
- 墙上时钟时间不如正确性重要的任务。
- 具有廉价、可靠价值函数的任务（代码的单元测试、数学的明确目标）。

如果你的任务只有一个正确答案且评估器有噪声，搜索往往会使情况更糟——它会找到一个"得分高"的错误答案。

### 2026 年定位

大多数生产 agent 不运行 LATS。它们运行 ReAct 配以工具 grounded 验证（CRITIC，第 05 课）。搜索出现在专业领域：

- 以测试为价值函数的编码 agent（HumanEval 风格）。
- 探索多个查询路径的深度研究 agent。
- LangGraph 子图内的规划密集型工作流。

AlphaEvolve（第 11 课）是 2025 年的极致：对代码进行演化搜索，机器可检查的适应度，前沿进展（56 年来首次 4x4 矩阵乘法改进）。

## 构建

`code/main.py` 实现：

- 一个在风格化"选择算术运算"任务上的微型 ToT BFS。
- 一个在同一任务上带有 UCT selection 的玩具 LATS MCTS 循环（Select / Expand / Simulate / Backpropagate）。
- 一个组合符号分数加自我评估分数的价值函数。

运行：

```
python3 code/main.py
```

轨迹显示 ToT 使用 BFS 每个节点展开三个候选方案，与 LATS 通过 MCTS 收敛到最佳 rollout 进行比较。二者均打印 token 计数。

## 使用

LangGraph 将 ToT 风格探索作为子图模式提供；LangChain 团队关于 LATS 的博文（2024 年 5 月）是参考教程。LlamaIndex 提供了一个 `TreeOfThoughts` agent。对 2026 年大多数生产 agent 而言，此模式位于 `if task_complexity > threshold: use_search()` 门控之后——参见第 05 课的 evaluator-optimizer 模式。

## 交付产物

`outputs/skill-search-policy.md` 给定任务形态、预算和评估器保真度，在线性 ReAct、ToT、LATS 和演化搜索之间做出选择。

## 练习

1. 用 UCT c=0.1 和 c=2.0 分别运行玩具 LATS。轨迹中有什么变化？
2. 将价值函数替换为有更多噪声的评分器（添加随机扰动）。MCTS 是否仍然找到最佳叶子？它能容忍的最小信噪比是多少？
3. 实现束搜索 ToT（每层保留 top-k）并与 BFS 比较。在紧张的 token 预算下哪个更好？
4. 阅读 LATS 第 5.1 节。复现 HumanEval 轨迹计数：达到报告的 pass@1 需要多少次 rollout？
5. 阅读 LATS 论文关于"LATS 何时帮助较小"的讨论。写一个将任务形态映射到搜索策略的一段式决策规则。

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Tree of Thoughts | "分支 CoT" | Yao 等人——带有自我评估的思考节点树 |
| LATS | "面向 LLM 的 MCTS" | Zhou 等人——在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | "上置信界" | 平衡 exploitation (Q) 和 exploration (ln N / n) 的 select 公式 |
| Value function | "此状态有多好" | 提示词化的 LLM 评分或环境奖励；馈入 backprop |
| Policy | "行动提议器" | ReAct 风格生成器；发出候选的下一步思考/行动 |
| Rollout | "模拟轨迹" | 使用 policy 从一个节点走到叶子节点，用 value 评分 |
| Backpropagate | "更新祖先" | 将叶子的奖励向上推，更新访问计数和 Q |
| Search cost | "token 爆炸" | Game of 24 上是 CoT 的 100-1000 倍；在采用之前做好预算 |

## 延伸阅读

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — 经典论文
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — 带 Reflexion 反馈的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 搜索的子图模式
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — 带可编程评估器的演化搜索