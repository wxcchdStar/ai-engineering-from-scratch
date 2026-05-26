# Self-Refine 与 CRITIC：迭代式输出改进

> Self-Refine（Madaan 等人，2023）在一个循环中使用一个 LLM 扮演三种角色——生成、反馈、精炼。平均提升：7 个任务上 +20 绝对值。CRITIC（Gou 等人，2023）通过将验证路由到外部工具来强化反馈步骤。2026 年此模式作为"evaluator-optimizer"（Anthropic）或 guardrail 循环（OpenAI Agents SDK）出现在每个框架中。

**类型：** 构建
**语言：** Python（标准库）
**前置课程：** Phase 14 · 01（Agent 循环），Phase 14 · 03（Reflexion）
**时长：** 约 60 分钟

## 学习目标

- 陈述 Self-Refine 的三个提示词（generate、feedback、refine），并解释为什么历史记录对 refine 提示词很重要。
- 解释 CRITIC 的关键洞察：LLM 在没有外部 grounding 的情况下自我验证不可靠。
- 实现一个带有历史记录和可选外部验证器的标准库 Self-Refine 循环。
- 将此模式映射到 Anthropic 的"evaluator-optimizer"工作流和 OpenAI Agents SDK 的输出 guardrails。

## 问题背景

一个 agent 产生的答案几乎正确。也许一行代码有语法错误。也许摘要太长了。也许计划遗漏了一个边缘情况。你想要的是：agent 审视自己的输出，然后修复它。

Self-Refine 表明这可以通过单一模型、无训练数据、无 RL 来实现。但有一个陷阱：LLM 在困难事实上的自我验证表现不佳。CRITIC 指明修复方案——将验证步骤路由到外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代式改进的默认方式：生成、验证（尽可能外部化）、精炼，在验证器通过时停止。

## 核心概念

### Self-Refine（Madaan 等人，NeurIPS 2023）

一个 LLM，三种角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
当 feedback 说"没有问题"或预算耗尽时停止。
```

关键细节：`refine` 看到完整历史——所有先前的输出和评判——因此它不会重复错误。论文对此做了消融实验：去掉历史记录后质量急剧下降。

亮点：包括 GPT-4 在内，在 7 个任务（数学、代码、缩写、对话等）上平均绝对提升 +20。无训练、无外部工具、单一模型。

### CRITIC（Gou 等人，arXiv:2305.11738，v4 2024 年 2 月）

Self-Refine 的弱点：反馈步骤是一个 LLM 给自己评分。对于事实性声明，这不可靠（一个幻觉对产生它的模型来说往往看起来很有说服力）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 一个用于事实性声明的搜索引擎。
- 一个用于代码正确性的代码解释器。
- 一个用于算术的计算器。
- 领域特定验证器（单元测试、类型检查器、linter）。

验证器产生一个基于工具结果的结构化评判 (critique)。精炼器然后基于此评判进行改进。

亮点：CRITIC 在事实性任务上优于 Self-Refine，因为评判是有 grounding 的。在没有外部验证器的任务上（创意写作、格式化），CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形态：

1. **验证器通过。** 外部测试返回成功。可用时优先（单元测试、类型检查器、guardrail 断言）。
2. **没有反馈发出。** 模型说"输出没问题。"更便宜但不可靠；配合最大迭代上限使用。

2026 年默认方案：组合使用。"如果验证器通过 OR（模型说没问题 AND iterations >= 2）OR iterations >= max_iterations 则停止。"

### Evaluator-Optimizer（Anthropic，2024）

Anthropic 2024 年 12 月的博文将此命名为五种工作流模式之一。两种角色：

- Evaluator：对输出评分并产生评判。
- Optimizer：根据评判修订输出。

循环直到 evaluator 通过。这在 Anthropic 的框架中就是 Self-Refine/CRITIC。Anthropic 增加的关键工程细节是：evaluator 和 optimizer 提示词应显著不同，以免模型只是做一个橡皮图章。

### OpenAI Agents SDK 输出 guardrails

OpenAI Agents SDK 将此模式作为"output guardrails"提供。guardrail 是在 agent 最终输出上运行的验证器。如果 guardrail 被触发（抛出 `OutputGuardrailTripwireTriggered`），输出被拒绝，agent 可以重试。Guardrails 可以调用工具（CRITIC 风格）或是纯函数（Self-Refine 风格）。

### 2026 年陷阱

- **橡皮图章循环。** 相同的模型使用相同提示词风格做生成和评判，会收敛于"看起来不错。"使用结构不同的提示词，或使用一个更小更便宜的模型做评判。
- **过度精炼。** 每次精炼增加延迟和 token。预算 1-3 次；之后升级到人工审查。
- **CRITIC 在琐碎任务上。** 如果没有外部验证器，CRITIC 退化为 Self-Refine；不要为存根验证器支付延迟成本。

## 构建

`code/main.py` 在玩具任务上实现 Self-Refine 和 CRITIC：给定主题产生一个简短的要点列表。验证器检查格式（3 个要点，每个 60 字符以内）。CRITIC 添加一个外部"事实验证器"，惩罚已知幻觉。

组件：

- `generate`——脚本化生成器。
- `feedback`——LLM 风格的自我评判。
- `verify_external`——CRITIC 风格的 grounded 验证器。
- `refine`——根据历史重写输出。
- 停止条件——验证器通过或最多 4 次迭代。

运行：

```
python3 code/main.py
```

比较 Self-Refine 和 CRITIC 的运行。CRITIC 捕捉到 Self-Refine 遗漏的一个事实性错误，因为外部验证器有自我评判器所没有的 grounding。

## 使用

Anthropic 的 evaluator-optimizer 是用 Claude 友好语言表述的此模式。OpenAI Agents SDK 的输出 guardrails 是 CRITIC 形态的（guardrails 可以调用工具）。LangGraph 提供了一个 reflection 节点，读起来像 Self-Refine。Google 的 Gemini 2.5 Computer Use 增加了一个每步安全评估器，这是 CRITIC 的变体：每个行动在提交前都要被验证。

## 交付产物

`outputs/skill-refine-loop.md` 给定任务形态、验证器可用性和迭代预算，配置一个 evaluator-optimizer 循环。发出生成器、评估器/验证器和优化器的提示词，外加停止策略。

## 练习

1. 用 max_iterations=1 运行玩具。CRITIC 仍然有帮助吗？
2. 将外部验证器替换为有噪声的验证器（随机 30% 误报）。循环会做什么？这是 2026 年大多数 guardrail 栈的现实。
3. 实现一个"生成器-评判器在不同模型上"的变体：大模型生成，小模型评判。它是否胜过同模型方式？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三种验证工具类别并各举一例。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。SDK 做错了什么？做对了什么？

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Self-Refine | "自我修复的 LLM" | 在一个模型中、带历史记录的 Generate -> feedback -> refine 循环 |
| CRITIC | "工具 grounded 验证" | 将 feedback 替换为外部验证器（搜索、代码、计算器、测试） |
| Evaluator-Optimizer | "Anthropic 工作流模式" | 两个角色——evaluator 评分，optimizer 修订——循环至收敛 |
| Output guardrail | "事后检查" | OpenAI Agents SDK 的验证器，在 agent 产生输出后运行 |
| Verify step | "评判阶段" | 关键的决策：grounded 还是 self-rated |
| Refine history | "模型已经尝试了什么" | 先前的输出 + 评判追加到 refine 提示词前；去掉则质量崩塌 |
| Rubber-stamp loop | "自我认同失败" | 相同提示词的评判返回"看起来不错"；用结构不同的提示词修复 |
| Stop condition | "收敛测试" | 验证器通过 OR 无反馈 AND 达到迭代上限；永不单条件 |

## 延伸阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 经典论文
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — 工具 grounded 验证
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — evaluator-optimizer 工作流模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 作为 CRITIC 形态验证器的 output guardrails