# OpenAI Preparedness Framework 与 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2（2025 年 4 月）引入了研究类别——长周期自主性、Sandbagging、自主复制与适应、破坏安全措施——与跟踪类别不同。跟踪类别触发能力报告加由安全咨询小组审查的安全措施报告。DeepMind 的 FSF v3（2025 年 9 月，跟踪能力级别于 2026 年 4 月 17 日添加）将自主性纳入 ML 研发和网络领域（ML 研发自主级别 1 = 以与人类 + AI 工具相比具有竞争力的成本完全自动化 AI 研发流水线）。FSF v3 通过自动化监控工具性推理滥用来明确处理欺骗性对齐。诚实的说明：PF v2 中的研究类别（包括长周期自主性）不会自动触发缓解措施；策略语言是"潜在的"。DeepMind 自己说，如果工具性推理加强，自动化监控"长期来看不会保持足够"。

**类型：** 学习
**语言：** Python（标准库，三框架决策表差异工具）
**前置条件：** 第 15 阶段 · 19（Anthropic RSP）
**时间：** 约 45 分钟

## 问题

第 19 课仔细阅读了 Anthropic 的扩展策略。本课通过阅读 OpenAI 和 DeepMind 的策略来完成图景。三份文档是解决同一问题的同类产物——前沿实验室何时应该暂停或门控模型——它们在一小组类别上收敛，在重要的特定地方分歧。

收敛：三者都将长周期自主性标记为值得跟踪的能力类别。三者都承认欺骗性行为（对齐伪装、sandbagging）作为特定的风险类别。三者都有一个内部审查机构。分歧：OpenAI 将类别分为"跟踪"（强制性缓解）和"研究"（无自动触发）。DeepMind 将自主性纳入两个领域而不是单独命名。实验室名称跟踪 vs 研究，或关键 vs 中等，或 Tier-1 vs Tier-2；一个能力位于哪个桶的操作后果在不同实验室之间是不同的。

一起阅读它们是有用的练习。相同的能力在 Anthropic 可能是"强制性缓解"，在 OpenAI 是"监控但不触发"，在 DeepMind 是"在特定领域中跟踪"。策略姿态很重要。

## 概念

### OpenAI Preparedness Framework v2（2025 年 4 月）

结构：

- **跟踪类别**：触发能力报告（模型能做什么）加安全措施报告（什么缓解措施到位）。在部署前由安全咨询小组审查。
- **研究类别**：实验室正在跟踪但尚未承诺特定缓解措施的模型能力。包括长周期自主性、Sandbagging、自主复制与适应、破坏安全措施。

v2 中的关键框架：研究类别不会自动触发缓解措施。策略语言是"潜在的"缓解措施。这是要关注的特定增量：类别在跟踪 vs 研究中的位置是"门控"和"观察"之间的铰链。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；跟踪能力级别于 2026 年 4 月 17 日添加）

结构：

- **关键能力级别（CCL）**：跨五个领域的能力阈值：网络、生物、ML 研发、CBRN、自主性（纳入 ML 研发和网络）。
- **跟踪能力级别**：2026 年 4 月添加的额外粒度。具体例子：ML 研发自主级别 1 = 以与人类 + AI 工具相比具有竞争力的成本完全自动化 AI 研发流水线。
- **欺骗性对齐监控**：对工具性推理滥用的自动化监控的明确承诺。

自主性框架与 OpenAI 的不同。DeepMind 不将"自主性"作为顶级领域；它被纳入自主性会造成伤害的领域（ML 研发和网络）。论点是：没有领域的自主性是没有风险的能力；反驳论点是：跨领域的自主性是框架应该命名的元风险。

### 三者都收敛于什么

- 内部安全咨询小组（命名为 Anthropic SAG、OpenAI SAG、DeepMind 内部委员会）。在高能力模型部署前审查。
- 明确提及欺骗性对齐 / 对齐伪装作为风险类别。
- 按声明节奏的常设产物（Anthropic：Frontier Safety Roadmap、Risk Report；OpenAI：能力和安全措施报告；DeepMind：FSF 更新周期）。
- 承认仅监控的防御有上限。DeepMind 是明确的："自动化监控长期来看不会保持足够。"

### 它们在哪里分歧

- **Anthropic**：v3.0 中移除暂停承诺；AI R&D-4 阈值是命名的下一个门控。
- **OpenAI**：跟踪 vs 研究分割；研究类别（包括长周期自主性）不自动门控。
- **DeepMind**：自主性纳入其他领域；跟踪能力级别在 2026 年 4 月添加粒度。

### Sandbagging：一个使三者都复杂化的特定能力

Sandbagging（模型在评估上战略性表现不佳）在 OpenAI 的研究类别中。Anthropic 的 RSP v3.0 通过评估上下文差距（第 1 课）处理它。DeepMind 通过 FSF v3 中的欺骗性对齐监控处理它。

如果模型在评估上 sandbag，每个框架的能力阈值都被低估。框架只有在测量有效时才有效。这就是为什么外部测量（第 21 课，METR）和对抗性评估是实验室自我评估之外的必要条件。

### 策略阅读技能

- **定位**：你关心的每个能力都应该在策略中可找到。如果不可找到，策略不涵盖它。
- **分类**：它是跟踪（触发缓解）还是研究（跟踪但不触发）？OpenAI 命名了这一点；Anthropic 和 DeepMind 有自己的等价物。
- **节奏**：策略是按声明的时间表更新，还是仅在特定事件后更新？声明的节奏更强。
- **独立性**：外部审查是强制性的还是可选的？Anthropic 与 Apollo 和美国 AI 安全研究所合作；OpenAI 与 METR；DeepMind 主要与内部 SAG。

## 使用它

`code/main.py` 实现一个小型决策表差异工具。给定一个能力（自主性、欺骗性对齐、研发自动化、网络提升等），它输出三个策略各自如何分类该能力，以及触发什么缓解措施。它是一个阅读辅助工具，不是策略工具。

## 交付它

`outputs/skill-cross-policy-diff.md` 使用三个框架作为参考，为特定能力生成跨策略比较。

## 练习

1. 运行 `code/main.py`。确认差异工具的输出对至少两个你可以对照源文档验证的能力匹配策略。

2. 完整阅读 OpenAI Preparedness Framework v2。确定每个研究类别。对每个，写一句话说明为什么它在研究而不是跟踪中。

3. 完整阅读 DeepMind FSF v3，加上 2026 年 4 月的跟踪能力级别更新。确定 ML 研发自主级别 1 的具体评估标准。你将如何在外部测量它？

4. Sandbagging 在 OpenAI 的研究类别中。设计一个能迫使 sandbagging 模型揭示其实际能力的评估。参考第 1 课的评估上下文博弈讨论。

5. 在特定能力（你选择）上比较三个策略。命名你认为最严格的策略分类和最不严格的。用源文本证明。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Preparedness Framework | "OpenAI 的扩展策略" | PF v2（2025 年 4 月）；跟踪 vs 研究类别 |
| 跟踪类别 | "强制性缓解" | 触发能力 + 安全措施报告；SAG 审查 |
| 研究类别 | "仅监控" | 跟踪但不自动缓解；包括长周期自主性 |
| Frontier Safety Framework | "DeepMind 的扩展策略" | FSF v3（2025 年 9 月）+ 跟踪能力级别（2026 年 4 月） |
| CCL | "关键能力级别" | DeepMind 每个领域的阈值（网络、生物、ML 研发、CBRN） |
| ML 研发自主级别 1 | "研发自动化" | 以竞争成本完全自动化 AI 研发流水线 |
| Sandbagging | "战略性表现不佳" | 模型在评估上表现不佳；在 OpenAI 研究类别中 |
| 工具性推理 | "手段-目的推理" | 关于如何实现目标的推理；DeepMind 监控的目标 |

## 扩展阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 公告。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 公告。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 跟踪能力级别添加。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 格式 Risk Report 的示例。
