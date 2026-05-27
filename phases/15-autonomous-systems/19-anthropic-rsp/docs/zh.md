# Anthropic 负责任扩展策略 v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代了 2023 年的策略。双层缓解措施：Anthropic 将单方面做什么 vs 什么被框架化为行业范围的建议（包括 RAND SL-4 安全标准）。将 Frontier Safety Roadmaps 和 Risk Reports 添加为常设文档，而非一次性交付物。放弃了 2023 年的暂停承诺。引入了 AI R&D-4 阈值：一旦跨越，Anthropic 必须发布确认不对齐风险和缓解措施的肯定性案例。Claude Opus 4.6 未跨越它。Anthropic 在 v3.0 公告中声明"自信地排除这一点正变得困难。"SaferAI 将 2023 年 RSP 评为 2.2；他们将 v3.0 降级为 1.9，将 Anthropic 与 OpenAI 和 DeepMind 一起归入"弱"RSP 类别。定性阈值取代了 2023 年的定量承诺；移除暂停条款是最尖锐的倒退。

**类型：** 学习
**语言：** Python（标准库，RSP 阈值决策引擎）
**前置条件：** 第 15 阶段 · 06（AAR），第 15 阶段 · 07（RSI）
**时间：** 约 45 分钟

## 问题

前沿实验室发布扩展策略，这些策略部分是技术文档，部分是治理文档，部分是给监管机构的信号。RSP v3.0 是当前的 Anthropic 文档。仔细阅读它很重要，不是因为遵守它是强制性的（它不是），而是因为框架塑造了实验室如何构想灾难性风险以及他们如何向公众传达权衡。

v3.0 vs v2.0 的差异是有用的单元。添加了什么：Frontier Safety Roadmaps、Risk Reports、AI R&D-4 阈值。移除了什么：2023 年的暂停承诺。重新框架了什么：分为 Anthropic 单方面和行业建议的双层缓解计划。外部审查——SaferAI——将分数从 2.2（v2）降级为 1.9（v3.0）。这就是扩展策略如何在看起来更精致的同时变得不那么严格。

## 概念

### 双层缓解计划

- **Anthropic 单方面行动**：无论其他实验室做什么，Anthropic 将做什么。训练在阈值以上停止，特定的安全措施，特定的部署门控。
- **行业范围建议**：Anthropic 认为行业应该集体做什么。包括 RAND SL-4 安全标准。这些不是 Anthropic 方面的承诺；它们是政策倡导。

双层结构不在 v2 中。这意味着读者需要查看每个承诺位于哪一列。"行业范围建议"列中的安全措施不是 Anthropic 的承诺；它是 Anthropic 的希望。

### AI R&D-4 阈值

这是 RSP v3.0 命名为重要下一个阈值的能力级别。具体来说：一个能够以竞争成本自动化 AI 研究相当大一部分的模型。一旦 Anthropic 认为模型跨越了它，他们必须在继续扩展之前发布确认不对齐风险和缓解措施的肯定性案例。

根据 v3.0 公告，Claude Opus 4.6 未跨越它。文档补充道："自信地排除这一点正变得困难。"这种措辞很重要；它承认阈值足够近，是一个活的问题，而不是一个推测性的限制。

第 6 课（自动化对齐研究）和第 7 课（递归自改进）直接输入此阈值。跨越研究质量标准的自动化对齐研究人员是 AI R&D-4 阈值正在接近的证据。

### Frontier Safety Roadmaps 和 Risk Reports

v3.0 将两种产物类型提升为常设文档：

- **Frontier Safety Roadmap**：前瞻性文档，描述计划的安全工作、能力预期和缓解研究。
- **Risk Report**：发布后特定模型的回顾性文档，描述观察到的能力和残余风险。

两者都是公开的。两者都按声明的节奏更新。效用是：读者可以跟踪 Anthropic 在 Roadmap 中说他们会做什么与他们在一份 Risk Report 中报告的内容的比较。

### 移除暂停条款

2023 年 RSP 包含一个明确的暂停承诺：如果模型跨越特定能力阈值，训练将暂停直到缓解措施到位。v3.0 用更柔和的表述取代了明确的暂停（发布肯定性案例，如果缓解措施充分则继续）。SaferAI 和其他分析师直接指出这是新文档中最强的倒退。

变更的政策论点：2023 年的定量阈值在 2026 年时代的能力基准下变得无法达到，因为基准本身被重新缩放。反驳论点：扩展策略中的暂停条款是一个承诺装置；移除它移除了策略的可信度。

### SaferAI 的降级

SaferAI 是一个对 RSP 风格文档进行评级的独立组织。他们的公开评级：2023 年 Anthropic RSP 得分 2.2（满分 4.0 是最佳当前 RSP，1.0 是名义上的）。v3.0 得分 1.9。这将 Anthropic 从"中等"移至"弱"，与 OpenAI 和 DeepMind 一起进入弱类别。

根据 SaferAI 的降级因素：
- 定性阈值取代了定量阈值。
- 暂停承诺被移除。
- AI R&D-4 阈值缓解措施被描述为"肯定性案例"而非具体措施。
- 审查机制依赖于 Anthropic 的安全咨询小组，独立监督有限。

### 本课不是什么

这不是合规课。RSP v3.0 不是法规；没有什么强制 Anthropic 遵循它。本课在于以应有的具体性和怀疑态度阅读文档。扩展策略是前沿实验室发出的关于灾难性风险姿态的主要公共信号。好好阅读它们对于任何工作依赖于前沿能力的人来说是一项实用技能。

## 使用它

`code/main.py` 实现一个反映 RSP 阈值评估形状的小型决策引擎：给定一个候选模型和一组能力测量，返回 AI R&D-4 阈值是否被跨越、所需的肯定性案例部分以及部署是否可以继续。它故意简单；重点是使文档的逻辑明确。

## 交付它

`outputs/skill-scaling-policy-review.md` 对照 v3.0 参考审查扩展策略（Anthropic、OpenAI、DeepMind 或内部）：双层结构、阈值、暂停承诺、独立审查。

## 练习

1. 运行 `code/main.py`。输入三个不同能力级别的合成模型。确认阈值评估器按预期行为并产生正确的肯定性案例模板。

2. 完整阅读 RSP v3.0（32 页）。确定每个位于"行业范围建议"层的承诺。这些承诺中哪些在 v2 中会是"Anthropic 单方面"的？

3. 阅读 SaferAI 的 RSP 评分方法论。通过将他们的评分标准应用于文档来重现他们对 v3.0 的 1.9 分。哪个评分标准行驱动了最多的降级？

4. 2023 年的暂停承诺被移除。提出一个替代承诺，在承认 2026 年基准重新缩放问题的同时保持策略的可信度。

5. 将 RSP v3.0 与 OpenAI Preparedness Framework v2（第 20 课）进行比较。选择一个 v3.0 更强的领域。选择一个 Preparedness Framework 更强的领域。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| RSP | "Anthropic 的扩展策略" | 负责任扩展策略；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | "研究自动化阈值" | 以竞争成本自动化大量 AI 研究的能力 |
| 肯定性案例 | "安全论证" | 已发布的论证，表明风险已识别且缓解措施充分 |
| Frontier Safety Roadmap | "前瞻计划" | 关于计划的安全工作和预期能力的常设文档 |
| Risk Report | "模型的回顾性报告" | 发布后关于观察到的能力和残余风险的常设文档 |
| 双层缓解 | "单方面 vs 行业" | Anthropic 承诺 vs 行业建议，分开 |
| 暂停承诺 | "2023 年条款" | 暂停训练的明确承诺；在 v3.0 中移除 |
| SaferAI 评级 | "独立 RSP 等级" | 第三方评分标准；v3.0 得分 1.9（v2 为 2.2） |

## 扩展阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整的 32 页策略。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 从 v2 变更的摘要。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — 从 RSP v3.0 链接的常设文档。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 当前前沿模型的回顾性报告。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 连接到测量的自主性。
