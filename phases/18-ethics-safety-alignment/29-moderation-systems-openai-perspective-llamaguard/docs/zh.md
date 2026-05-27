# 审核系统 —— OpenAI、Perspective、Llama Guard

> 生产审核系统将第 12-16 课中定义的安全策略操作化。OpenAI Moderation API：`omni-moderation-latest`（2024）基于 GPT-4o，在一次调用中分类文本 + 图像；在多语言测试集上比先前版本好 42%；响应模式返回 13 个类别布尔值——harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用隐藏延迟；标记时显示占位响应。Llama Guard 3/4（第 16 课）：14 个 MLCommons 危害类别、代码解释器滥用、8 种语言（v3）、多图像（v4）。Perspective API（Google Jigsaw）：早于 LLM 作为审核者浪潮的毒性评分；主要是单维度毒性，具有 severe-toxicity/insult/profanity 变体；内容审核研究的基线。弃用：Azure Content Moderator 于 2024 年 2 月弃用，2027 年 2 月退役，由 Azure AI Content Safety 取代。

**类型：** 构建
**语言：** Python（标准库，三层审核工具）
**前置知识：** Phase 18 · 16（Llama Guard / Garak / PyRIT）
**时间：** ~60 分钟

## 学习目标

- 描述 OpenAI Moderation API 的类别分类法以及它与 Llama Guard 3 的 MLCommons 集合有何不同。
- 描述三层审核模式（输入、输出、自定义）并列举每层的一个失败模式。
- 描述 Perspective API 作为 LLM 时代前基线的地位以及为什么它仍在研究中使用。
- 陈述 Azure 弃用时间线。

## 问题

第 12-16 课描述了攻击和防御工具。第 29 课涵盖了在用户接触产品的表面将防御操作化的已部署审核系统。三层模式是 2026 年的默认配置。

## 概念

### OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o。在一次调用中分类文本 + 图像。对大多数开发者免费。

类别（响应模式中的 13 个布尔值）：
- harassment、harassment/threatening
- hate、hate/threatening
- self-harm、self-harm/intent、self-harm/instructions
- sexual、sexual/minors
- violence、violence/graphic
- illicit、illicit/violent

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不包括 `sexual/minors`；其余仅限文本。

对于 `code/main.py` 中的代码工具，我们将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别折叠到其顶级父类别中，以简化教学。生产代码应使用完整的 13 类别模式。

在多语言测试集上比前代审核端点好 42%。按类别评分；应用程序设置阈值。

### Llama Guard 3/4

在第 16 课中涵盖。14 个 MLCommons 危害类别（组织方式与 OpenAI 的 13 个响应模式布尔值不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）是原生多模态的，12B。

OpenAI 和 Llama Guard 分类法有重叠但存在分歧。OpenAI 将"illicit"作为一个广泛类别；Llama Guard 将"violent crimes"和"non-violent crimes"分开。部署根据其策略-分类法匹配度进行选择。

### Perspective API（Google Jigsaw）

早于 LLM 作为审核者浪潮（2020 年前）的毒性评分系统。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。单维度主要分数（TOXICITY）具有子维度变体。

广泛用作内容审核研究基线，因为 API 稳定、有文档记录且有多年的校准数据。对于现代 LLM 相关用例，Llama Guard 或 OpenAI Moderation 通常是更好的选择。

### 三层模式

1. **输入审核。** 在生成前分类用户提示词。如果标记则拒绝。延迟：一次分类器调用。
2. **输出审核。** 在交付前分类模型输出。如果标记则替换为拒绝。延迟：生成后一次分类器调用。
3. **自定义审核。** 领域特定规则（正则表达式、允许列表、业务策略）。在输入或输出时运行。

三层按设计是顺序的：输入审核必须在生成前完成，输出审核在生成后运行。并行性适用于层内——在同一文本上同时运行多个分类器（例如 OpenAI Moderation + Llama Guard + Perspective）隐藏了每个分类器的延迟。作为可选的优化，可以在输入审核完成且 token-1 流式传输延迟时显示占位响应（"请稍候，正在检查……"）。标记行为是可配置的：拒绝、净化、升级到人工审查。

### 失败模式

- **仅输入。** 不捕获输出幻觉（第 12-14 课的编码攻击绕过输入分类器）。
- **仅输出。** 允许任何输入到达模型；增加成本；向攻击者暴露内部推理。
- **仅自定义。** 跨类别不鲁棒；正则表达式脆弱。

分层是默认的。双保险。

### Azure 弃用

Azure Content Moderator：2024 年 2 月弃用，2027 年 2 月退役。由 Azure AI Content Safety 取代，后者基于 LLM 并与 Azure OpenAI 集成。迁移是 Azure 部署的 2024-2027 年级别项目。

### 在 Phase 18 中的位置

第 16 课涵盖红队上下文中的审核工具。第 29 课涵盖已部署的审核。第 30 课以当前双重用途能力证据收尾。

## 使用它

`code/main.py` 构建一个三层审核工具：输入审核器（关键词 + 类别分数）、输出审核器（对输出使用相同分类器）、自定义审核器（领域规则）。你可以通过它运行输入并观察哪一层捕获了什么。

## 交付它

本课产出 `outputs/skill-moderation-stack.md`。给定一个部署，它推荐审核栈配置：输入使用哪个分类器，输出使用哪个，哪些自定义规则，以及边缘情况使用什么评判者。

## 练习

1. 运行 `code/main.py`。通过所有三层运行良性、边界和有害输入。报告每种情况下哪一层触发。

2. 用 Perspective API 风格的特定类别毒性评分扩展工具。将其阈值行为与类别分数进行比较。

3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。找出三个不能干净映射的类别。

4. 为代码助手部署（例如 GitHub Copilot）设计审核栈。找出最相关和最不相关的类别，并提出自定义规则。

5. Azure Content Moderator 于 2027 年 2 月退役。规划向 Azure AI Content Safety 的迁移。找出迁移中风险最高的元素。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| OpenAI Moderation | "omni-moderation-latest" | 基于 GPT-4o 的 13 类别（文本）分类器，具有部分多模态支持 |
| Perspective API | "Google Jigsaw 毒性" | LLM 时代前的毒性评分基线 |
| Llama Guard | "MLCommons 14 类别" | Meta 的危害分类器（v3：8B 文本，8 种语言；v4：12B 多模态） |
| 输入审核（Input moderation） | "生成前过滤器" | 模型调用前对用户提示词的分类器 |
| 输出审核（Output moderation） | "生成后过滤器" | 交付前对模型输出的分类器 |
| 自定义审核（Custom moderation） | "领域规则" | 部署特定规则（正则表达式、允许列表、策略） |
| 分层审核（Layered moderation） | "所有三层" | 标准生产部署模式 |

## 扩展阅读

- [OpenAI Moderation API 文档](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 替代方案
