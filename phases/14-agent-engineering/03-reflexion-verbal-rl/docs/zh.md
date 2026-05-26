# Reflexion：语言化强化学习

> 基于梯度的 RL 需要数千次试验和 GPU 集群来修复一个故障模式。Reflexion（Shinn 等人，NeurIPS 2023）用自然语言做到：每次失败的试验后，agent 写下一段反思 (reflection)，存储在情节记忆 (episodic memory) 中，并在下一次试验中基于该记忆调整。这就是 Letta 的 sleep-time compute、Claude Code 的 CLAUDE.md learnings 和 pro-workflow 的 learn-rule 背后的模式。

**类型：** 构建
**语言：** Python（标准库）
**前置课程：** Phase 14 · 01（Agent 循环），Phase 14 · 02（ReWOO）
**时长：** 约 60 分钟

## 学习目标

- 说出 Reflexion 的三个组成部分（Actor、Evaluator、Self-Reflector）以及情节记忆的角色。
- 用标准库实现一个带有二元评估器、反思缓冲区和全新重试的 Reflexion 循环。
- 为给定任务在标量、启发式和自我评估反馈源之间做出选择。
- 解释为什么语言化强化能捕捉到基于梯度的 RL 需要数千次试验才能修复的错误。

## 问题背景

一个 agent 任务失败。在标准 RL 中，你需要再运行数千次试验，计算梯度，更新权重。昂贵、缓慢，且大多数生产 agent 没有为每个故障准备的训练预算。

Reflexion（Shinn 等人，arXiv:2303.11366）提出了一个不同的问题：如果 agent 仅仅是思考为什么失败了，然后带着这个思考在其提示词中重试呢？没有权重更新。没有梯度。只有存储在试验之间的自然语言。

结果：在 ALFWorld 上它超越了 ReAct 和其他未经微调的基线。在 HotpotQA 上它改进了 ReAct。在代码生成（HumanEval/MBPP）上它当时达到了最优水平。全程没有一次梯度步骤。

## 核心概念

### 三个组成部分

```
Actor         : 产生一条轨迹（ReAct 风格循环）
Evaluator     : 给轨迹打分——二元、启发式或自我评估
Self-Reflector: 写一段关于失败的自然语言反思
```

外加一个数据结构：

```
Episodic memory（情节记忆）: 先前反思的列表，追加到下一次试验的提示词前
```

一次试验运行 Actor。Evaluator 评分。如果分数低，Self-Reflector 产生一段反思（"我选错了工具，因为我误读了问题，以为它问的是 X 而实际问的是 Y"）。反思进入情节记忆。下一次试验从头开始但看到这段反思。

### 三种评估器类型

1. **标量 (Scalar)**——外部二元信号。ALFWorld 成功或失败。HumanEval 测试通过或失败。最简单，信号最强。
2. **启发式 (Heuristic)**——预定义的失败特征。"如果 agent 连续产生相同的行动两次，标记为卡住。""如果轨迹超过 50 步，标记为低效。"
3. **自我评估 (Self-evaluated)**——LLM 给自己的轨迹打分。在无 ground truth 时使用。信号较弱；与工具 grounded 验证配合使用效果最好（第 05 课——CRITIC）。

2026 年默认方案是混合：有标量时用标量，没有时用自我评估，启发式作为安全护栏。

### 为什么这能泛化

Reflexion 与其说是一种新算法，不如说是一种命名模式。几乎每个生产级"自愈" agent 都运行着某种变体：

- Letta 的 sleep-time compute（第 08 课）：一个单独 agent 反思过去的对话并写入内存块。
- Claude Code 的 `CLAUDE.md` / "save memory" 模式：反思作为 learnings 捕获，追加到未来的会话之前。
- pro-workflow 的 `/learn-rule` 命令：修正作为显式规则捕获。
- LangGraph 的 reflection 节点：一个节点对输出评分，如果需要则路由到 refine。

所有这些都源于同一个洞见：自然语言是一种足够丰富的媒介，可以在运行之间传递"我从失败中学到了什么"。

### 何时有效，何时无效

Reflexion 在以下情况下有效：

- 有明确的失败信号（测试失败、工具错误、错误答案）。
- 任务类别是可复现的（相同类型的问题可以再次提出）。
- 反思有空间改进轨迹（足够的行动预算）。

Reflexion 在以下情况下无效：

- Agent 首次尝试就已经成功。
- 失败是外部的（网络断开、工具损坏）——"网络曾经断开过"的反思对未来的运行没有帮助。
- 反思变成了迷信——存储关于一次性不稳定运行的叙述。

2026 年陷阱：内存腐败 (memory rot)。反思不断累积；一些已过时或错误；随着情节缓冲区的增长，重跑变得更慢。缓解措施：定期压缩（第 06 课）、反思的 TTL、或单独的 sleep-time 清理 agent（Letta）。

## 构建

`code/main.py` 在一个玩具谜题上实现 Reflexion：产生一个总和为目标值的 3 元素列表。Actor 发出候选列表；Evaluator 检查和；Self-Reflector 写一行关于哪里出错的文字。反思进入情节记忆供下一次试验使用。

组件：

- `Actor`——一个在看到反思时会改进的脚本化策略。
- `Evaluator.binary()`——对目标和通过/失败。
- `SelfReflector`——产生一行关于故障的诊断。
- `EpisodicMemory`——具有 TTL 语义的有界列表。

运行：

```
python3 code/main.py
```

轨迹显示三次试验。试验 1 失败，记录反思，试验 2 看到反思并改进但仍失败，试验 3 成功。与基线运行（无反思）比较——它一直停留在试验 1 的答案上。

## 使用

LangGraph 将 reflection 作为一个节点模式提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 将情节缓冲区外部化为一个 markdown 文件。Letta 的 sleep-time compute 在停机时运行 Self-Reflector，因此主 agent 保持延迟受限。OpenAI Agents SDK 不直接提供 Reflexion；你需要通过自定义 Guardrail（按分数拒绝轨迹）和一个跨运行存活的 memory `Session` 来构建它。

## 交付产物

`outputs/skill-reflexion-buffer.md` 创建并维护一个带有反思捕获、TTL 和去重的情节缓冲区。给定任务类别和一次失败，它发出一个实际上能帮助下一次试验的反思（而非通用的"要更小心"）。

## 练习

1. 从二元评估器切换到返回距离度量（离目标多远）的标量评估器。它收敛更快吗？
2. 给反思添加一个 10 次试验的 TTL。在那之后，旧反思是帮助还是阻碍？
3. 实现启发式评估器：如果相同行动重复则标记试验为卡住。这与 Self-Reflector 如何交互？
4. 用一个忽略反思的对抗性 Actor 运行 Reflexion。迫使 Actor 注意到反思的最小反思提示词工程是什么？
5. 阅读 Reflexion 论文第 4 节关于 AlfWorld 的内容。概念上复现 130% 成功率提升：与普通 ReAct 相比的关键差异是什么？

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Reflexion | "自我纠正" | Shinn 等人 2023——Actor、Evaluator、Self-Reflector 加情节记忆 |
| Verbal reinforcement | "无需梯度的学习" | 自然语言反思追加到下一次试验的提示词之前 |
| Episodic memory | "逐任务反思" | 针对一个任务类别的先前反思的有界缓冲区 |
| Scalar evaluator | "二元成功信号" | 来自 ground truth 的通过/失败或数值分数 |
| Heuristic evaluator | "基于模式的检测器" | 预定义的失败特征（如卡住循环、步数过多） |
| Self-evaluator | "LLM 作为自身轨迹的评判者" | 无 ground truth 时的低信号备选方案——与工具 grounded 验证配对 |
| Memory rot | "过时反思" | 情节缓冲区填满过时条目；用压缩/TTL 修复 |
| Sleep-time reflection | "异步自我反思" | 在热路径之外运行 Self-Reflector，使主 agent 保持快速 |

## 延伸阅读

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 经典论文
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — 生产环境中的异步反思
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 将情节缓冲区作为上下文管理
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — reflection 节点模式