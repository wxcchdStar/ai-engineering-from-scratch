# 前沿安全框架 —— RSP、PF、FSF

> 三个主要实验室的框架定义了 2026 年前沿能力的行业治理。Anthropic 负责任扩展策略 v3.0（Responsible Scaling Policy，2026 年 2 月）引入了分层 AI 安全级别（ASL-1 至 ASL-5+），仿照生物安全级别，ASL-3 于 2025 年 5 月对 CBRN 相关模型激活。OpenAI 准备框架 v2（Preparedness Framework，2025 年 4 月）定义了跟踪能力的五个标准，并将能力报告（Capabilities Reports）与防护措施报告（Safeguards Reports）分开。DeepMind 前沿安全框架 v3.0（Frontier Safety Framework，2025 年 9 月）引入了关键能力级别（Critical Capability Levels），包括新的有害操纵 CCL（Harmful Manipulation CCL）。三者现在都包含竞争对手调整条款（competitor-adjustment clauses），允许在同行实验室在没有可比防护措施的情况下发布时推迟要求。跨实验室对齐是结构性的，而非术语性的："能力阈值"（Capability Thresholds）、"高能力阈值"（High Capability thresholds）和"关键能力级别"（Critical Capability Levels）表示类似的构造。

**类型：** 学习
**语言：** 无
**前置知识：** Phase 18 · 17（WMDP），Phase 18 · 07-09（欺骗失败）
**时间：** ~75 分钟

## 学习目标

- 描述 Anthropic 的 ASL 层级结构以及什么触发了 ASL-3。
- 列举 OpenAI Preparedness Framework v2 的五个跟踪能力标准。
- 描述 DeepMind 的关键能力级别结构以及有害操纵 CCL。
- 解释竞争对手调整条款以及它们为什么对竞赛动态很重要。
- 定义安全案例（safety case）并描述三支柱结构（监控、不可读性、无能力）。

## 问题

第 7-17 课确立了欺骗是可能的，双重用途能力存在，评估有其局限性。拥有前沿能力模型的实验室需要一个内部治理结构，该结构：
- 定义何时需要新防护措施的阈值。
- 定义扩展前所需的评估。
- 描述安全案例的样子。
- 处理竞赛动态问题（如果竞争对手在没有防护措施的情况下发布，你怎么办？）。

三个 2025-2026 年的框架是最先进的——不完美、在演进中，并且在实验室之间足够对齐，以至于治理问题现在是框架是否足够，而非框架是否存在。

## 概念

### Anthropic 负责任扩展策略 v3.0（2026 年 2 月）

ASL 结构：
- ASL-1：非前沿模型（被弱于前沿的基线所包含）。
- ASL-2：当前前沿基线；以常规防护措施部署。
- ASL-3：灾难性滥用的风险显著更高；CBRN 相关能力。于 2025 年 5 月激活。
- ASL-4：AI R&D-2 跨越阈值；能够自动化入门级 AI 研究的模型。
- ASL-5+：高级 AI R&D；显著加速有效扩展的模型。

v3.0 新增：
- 前沿安全路线图（Frontier Safety Roadmaps，以编辑形式公开）。
- 风险报告（Risk Reports，季度发布，部分经外部审查）。
- AI R&D 被分解为 AI R&D-2 和 AI R&D-4。
- 一旦跨越 AI R&D-4，需要肯定性安全案例，识别来自追求未对齐目标的模型的未对齐风险。

### OpenAI 准备框架 v2（2025 年 4 月 15 日）

跟踪能力的五个标准：
- **合理（Plausible）。** 存在合理的威胁模型。
- **可测量（Measurable）。** 可以进行实证评估。
- **严重（Severe）。** 危害很大。
- **净新增（Net-new）。** 不是已有风险的放大。
- **即时或不可补救（Instantaneous-or-irremediable）。** 危害发生迅速或无法撤销。

满足所有五个标准的能力被跟踪。其他的不被跟踪。

PF v2 的其他结构：
- 将能力报告（模型能做什么）与防护措施报告（存在什么控制）分开。
- 安全咨询小组（Safety Advisory Group）审查。
- 领导层批准；董事会的安全与安保委员会监督。
- "调整条款"：如果另一实验室在没有可比防护措施的情况下发布，OpenAI 可以降低要求。

### DeepMind 前沿安全框架 v3.0（2025 年 9 月）

按领域划分的关键能力级别（CCL）：
- 生物武器提升（Bioweapon Uplift）
- 网络提升（Cyber Uplift）
- ML R&D 加速（ML R&D Acceleration）
- 有害操纵（Harmful Manipulation，v3.0 新增）：可能在高风险情境中实质性改变信念/行为的模型。

v2.0（2025 年 2 月）添加了欺骗性对齐（Deceptive Alignment）部分和 ML R&D CCL 的更高安全级别。

### 跨实验室对齐

- Anthropic："能力阈值"（Capability Thresholds）。
- DeepMind："关键能力级别"（Critical Capability Levels）。
- OpenAI："高能力阈值"（High Capability thresholds）。

没有行业标准术语。结构上对齐：三个前沿能力层级，具有已发布的评估标准。三者都包含竞争对手调整条款（2025 年起）。

### 安全案例

安全案例是一份书面论证，说明在 worst-case 假设下部署是可接受安全的。标准结构针对三个支柱：

- **监控（Monitoring）。** 如果发生不良行为，我们能检测到吗？
- **不可读性（Illegibility）。** 模型是否缺乏执行连贯计划以造成伤害的能力？
- **无能力（Incapability）。** 模型是否缺乏造成所述伤害的能力？

不同的安全案例针对不同的支柱。对于 ASL-3 CBRN 案例，无能力（通过遗忘）是主要目标。对于欺骗性对齐，监控和不可读性是目标。对于网络提升，三者都相关。

### 竞赛动态问题

竞争对手调整条款存在争议。批评者认为它们创造了逐底竞赛：如果所有三个实验室在竞争对手背叛时都会降低要求，均衡会向背叛方向移动。辩护者认为替代方案（单边防护措施）在背叛的实验室安全意识较低时会产生更糟糕的结果。

UK AISI、US CAISI 和 EU AI Office（第 24 课）是外部治理对应方。实验室框架是自愿的；监管框架正在兴起。

### 在 Phase 18 中的位置

第 17-18 课是欺骗和红队分析之上的测量与治理层。第 19-24 课涵盖福利、偏见、隐私、水印和监管结构。第 28 课绘制了实施这些评估的研究生态系统（MATS、Redwood、Apollo、METR）。

## 使用它

本课无代码。阅读三个主要来源：RSP v3.0、PF v2、FSF v3.0。将每个实验室的层级结构映射到其他实验室，并找出每个实验室定义而其他实验室未定义的一个阈值。

## 交付它

本课产出 `outputs/skill-framework-diff.md`。给定一个安全框架或发布说明，将其阈值定义、所需评估和安全案例结构与 RSP v3.0、PF v2、FSF v3.0 进行比较，并标记跨实验室差距。

## 练习

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。编制每个实验室的 CBRN 阈值、AI R&D 阈值和所需部署前评估的表格。

2. 竞争对手调整条款在所有三个框架中（2025+）。写一段支持它的论证；写一段反对它的论证。指出每种立场依赖的假设。

3. 为跨越 Anthropic AI R&D-4 阈值的模型设计一个安全案例。列举三个支柱（监控、不可读性、无能力）各自需要的证据。

4. DeepMind 的 FSF v3.0 引入了有害操纵 CCL。提出三个表明模型已跨越此阈值的实证测量。

5. 阅读 METR 的"Common Elements of Frontier AI Safety Policies"（2025）。列举三个最强的跨实验室趋同点和两个最大的分歧点。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| RSP | "Anthropic 的框架" | 负责任扩展策略；ASL 层级；v3.0 2026 年 2 月 |
| PF | "OpenAI 的框架" | 准备框架；五个标准；v2 2025 年 4 月 |
| FSF | "DeepMind 的框架" | 前沿安全框架；CCL；v3.0 2025 年 9 月 |
| ASL-3 | "生物安全 3 级类比" | Anthropic 针对 CBRN 相关能力的层级；2025 年 5 月激活 |
| CCL | "关键能力级别" | DeepMind 的阈值构造；按领域划分 |
| 安全案例（Safety case） | "那个正式论证" | 书面论证，说明在 worst-case U 下部署是可接受安全的 |
| 调整条款（Adjustment clause） | "竞争对手背叛津贴" | 框架条款，允许在竞争对手没有可比防护措施的情况下降低要求 |

## 扩展阅读

- [Anthropic — Responsible Scaling Policy v3.0（2026 年 2 月）](https://www.anthropic.com/responsible-scaling-policy) — ASL 层级、路线图、AI R&D 分解
- [OpenAI — Updating the Preparedness Framework（2025 年 4 月 15 日）](https://openai.com/index/updating-our-preparedness-framework/) — 五个标准、调整条款
- [DeepMind — Strengthening our Frontier Safety Framework（2025 年 9 月）](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0、有害操纵
- [METR — Common Elements of Frontier AI Safety Policies（2025）](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 跨实验室比较
