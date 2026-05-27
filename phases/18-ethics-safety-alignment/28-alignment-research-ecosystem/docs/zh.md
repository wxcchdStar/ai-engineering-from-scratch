# 对齐研究生态系统 —— MATS、Redwood、Apollo、METR

> 五个组织定义了 2026 年非实验室对齐研究层。MATS（ML Alignment & Theory Scholars）：自 2021 年底以来 527+ 名研究员，180+ 篇论文，10K+ 引用，h-index 47；2024 年夏季 cohort 注册为 501(c)(3)，约 90 名学者和 40 名导师；80% 的 2025 年前校友从事安全/安保工作，200+ 人在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的应用对齐实验室；引入了 AI Control（第 10 课）；与 UK AISI 在控制安全案例上合作。Apollo Research：为前沿实验室进行部署前策划评估；撰写了 In-Context Scheming（第 8 课）和 Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research）：基于任务的能力评估，自主任务时间跨度研究；"Common Elements of Frontier AI Safety Policies"比较了实验室框架。Eleos AI Research：模型福利部署前评估（第 19 课）；进行了 Claude Opus 4 福利评估。

**类型：** 学习
**语言：** 无
**前置知识：** Phase 18 · 01-27（Phase 18 之前的课程）
**时间：** ~45 分钟

## 学习目标

- 识别非实验室对齐研究生态系统的五个组织及其核心产出。
- 描述 MATS 的规模（学者、论文、h-index）及其作为人才管道的角色。
- 描述 Redwood 的 AI Control 议程及其与 UK AISI 的合作。
- 描述 METR 的基于任务的评估方法论。

## 问题

前沿实验室（第 18 课）在内部产生安全评估并发布选定结果。实验室之外的生态系统是评估被验证的地方，是新颖失败模式首次被发现的地方，也是人才被训练的地方。理解生态系统有助于解释哪些研究发现被谁信任。

## 概念

### MATS（ML Alignment & Theory Scholars）

始于 2021 年底。研究导师计划；学者与一位资深研究员在特定对齐问题上共度 10-12 周。

规模（2026）：
- 自成立以来 527+ 名研究员。
- 发表 180+ 篇论文。
- 10K+ 引用。
- h-index 47。
- 2024 年夏季：90 名学者 + 40 名导师；注册为 501(c)(3)。

职业成果：约 80% 的 2025 年前校友从事安全/安保工作。200+ 人在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。引入了 AI Control 议程（第 10 课）。与 UK AISI 在控制安全案例上合作。为 DeepMind 和 Anthropic 提供评估设计建议。

经典论文：Greenblatt、Shlegeris 等人，"AI Control"（arXiv:2312.06942，ICML 2024）；Alignment Faking（Greenblatt、Denison、Wright 等人，arXiv:2412.14093，与 Anthropic 联合）。

风格：具体的威胁模型、worst-case 对手、可被压力测试的具体协议。

### Apollo Research

为前沿实验室进行部署前策划评估。撰写了 In-Context Scheming（第 8 课，arXiv:2412.04984）。2025 年 OpenAI 反策划训练合作的合作伙伴。产出 Towards Safety Cases for AI Scheming（2024）。

风格：欺骗可能出现的智能体设置评估；三支柱分解（未对齐、目标导向性、情境意识）。

### METR（Model Evaluation and Threat Research）

基于任务的能力评估。自主任务完成时间跨度研究。"Common Elements of Frontier AI Safety Policies"（metr.org/common-elements，2025）比较了实验室框架。

与 Apollo 共同撰写 AI Scheming 安全案例草稿。

风格：长时间跨度任务评估、经验能力测量、框架综合。

### Eleos AI Research

模型福利部署前评估。进行了系统卡第 5.3 节记录的 Claude Opus 4 福利评估。为第 19 课的福利相关声明提供外部方法论检查。

### 流程

MATS 训练研究员。毕业生前往 Anthropic、DeepMind、OpenAI（实验室安全团队）或 Redwood、Apollo、METR、Eleos（外部评估）。外部评估者与实验室以及 UK AISI / CAISI 合作。出版物将生态系统反馈给 MATS 以供下一 cohort。

### 为什么这一层重要

单一来源的评估不可靠：评估自己模型的实验室存在结构性利益冲突。外部评估者可以提出并验证实验室可能低报的失败模式。2024 年 Sleeper Agents 论文（第 7 课）是 Anthropic + Redwood；Alignment Faking 是 Anthropic + Redwood；In-Context Scheming 是 Apollo；Anti-Scheming 是 Apollo + OpenAI。多组织结构是质量控制。

### 在 Phase 18 中的位置

第 7-11 课引用了 Redwood 和 Apollo 的工作；第 18 课引用了 METR 的框架比较；第 19 课引用了 Eleos。第 28 课是 Phase 其余部分所依赖的生态系统的显式组织地图。

## 使用它

无代码。阅读 METR 的"Common Elements of Frontier AI Safety Policies"作为外部综合如何为实验室内部政策工作增加价值的例子。

## 交付它

本课产出 `outputs/skill-ecosystem-map.md`。给定一个对齐声明或评估，它识别组织、发表场所和方法论风格，并与已知对应组织进行交叉检查。

## 练习

1. 从第 7-15 课中挑选一篇论文，识别涉及的组织。将作者与 MATS 校友和当前生态系统隶属关系进行交叉检查。

2. 阅读 METR 的"Common Elements of Frontier AI Safety Policies"。找出他们强调的三个跨实验室趋同点和两个最大的分歧点。

3. MATS 职业成果约 80% 是安全/安保。论证这种选择压力是适应性的（训练该领域）还是有偏的（过滤掉非正统立场）。

4. Redwood 和 Apollo 都做控制/策划工作，但风格不同。挑选一个失败模式，描述每个组织将如何调查它。

5. Eleos AI 是唯一的纯模型福利组织。设计一个假设的第二个组织，专注于不同的福利相关问题（认知自由、机器人具身等），并阐明其方法论。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MATS | "那个导师计划" | ML Alignment & Theory Scholars；自 2021 年以来 527+ 名研究员 |
| Redwood Research | "那个控制实验室" | 应用对齐；AI Control 作者；UK AISI 合作伙伴 |
| Apollo Research | "那个策划评估" | 为前沿实验室进行部署前策划评估 |
| METR | "那个任务跨度评估" | 基于任务的能力评估；框架综合 |
| Eleos AI | "那个福利实验室" | 模型福利部署前评估 |
| 人才管道（Talent pipeline） | "MATS -> 实验室" | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外部评估（External evaluation） | "非实验室检查" | 非模型生产者进行的评估；增加可信度 |

## 扩展阅读

- [MATS（ML Alignment & Theory Scholars）](https://www.matsprogram.org/) — 导师计划
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control 论文
- [Apollo Research](https://www.apolloresearch.ai/) — 策划评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法论
