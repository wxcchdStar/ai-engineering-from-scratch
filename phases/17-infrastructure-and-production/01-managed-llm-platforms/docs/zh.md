# 托管 LLM 平台 — AWS Bedrock、Azure AI Foundry、GCP Vertex AI、Databricks Mosaic AI、Together AI、Fireworks AI、Groq

> 2026 年的托管 LLM 平台格局已分化为三个层级。**超大规模云平台（Hyperscalers）** — AWS Bedrock、Azure AI Foundry、GCP Vertex AI — 提供最广泛的模型目录、SOC 2/HIPAA/ISO 合规认证以及与现有云基础设施的原生集成；代价是每 token 价格通常比专业平台高 15-30%，且模型发布滞后 2-6 周。**数据/AI 平台** — Databricks Mosaic AI — 在现有数据湖仓（data lakehouse）之上提供统一的治理、特征存储和模型服务；适合已经在 Databricks 上做 ETL 和训练的团队。**专业推理平台（Specialist Inference Platforms）** — Together AI、Fireworks AI、Groq — 在原始推理吞吐量和延迟方面领先；Together 和 Fireworks 在开源模型上提供比超大规模云平台低 30-50% 的每 token 价格；Groq 的 LPU（语言处理单元，Language Processing Unit）硬件在 Llama 3.1 70B 上实现 < 50ms TTFT（首 token 时间，Time To First Token）。选择框架：延迟优先 → Groq；成本优先 + 开源模型 → Together/Fireworks；合规优先 + 现有云支出承诺 → Bedrock/Azure/Vertex；数据湖仓统一 → Databricks。多平台策略正在成为标准 — 在超大规模云平台上运行基础模型以满足合规要求，在专业平台上运行微调后的开源模型以控制成本。

**类型：** 学习
**语言：** Python（标准库，平台选择决策树演练）
**前置要求：** 无
**时间：** 约 45 分钟

## 学习目标

- 列举 2026 年三个平台层级（超大规模云平台、数据/AI 平台、专业推理平台）并说出每个层级中至少两家提供商。
- 根据延迟、成本、模型广度、合规和生态系统锁定五个维度选择平台。
- 解释为什么多平台策略正在成为标准，以及超大规模云平台与专业平台之间的权衡。
- 说出 Groq LPU 在 Llama 3.1 70B 上的 TTFT 基准数据（< 50ms）。

## 问题

你的团队需要为面向客户的产品选择一个 LLM 平台。AWS 的客户经理推销 Bedrock；你的数据团队已经在 Databricks 上；一位工程师在 Hacker News 上读到 Groq 是最快的。三个选项都是对的，但针对的是不同的约束条件。

错误的选择会让你付出代价。选择 Bedrock 用于延迟敏感型聊天机器人，你会比 Groq 多支付 30% 的费用且 TTFT 慢 3 倍。选择 Groq 用于需要 HIPAA BAA（业务伙伴协议，Business Associate Agreement）的医疗应用，你根本无法上线。选择 Together AI 而你的安全团队要求 SOC 2 Type II，你会在合规审查上卡住三个月。

平台选择是基础设施决策中最具后果性的之一 — 它决定了成本曲线、延迟特性、合规边界以及未来三年的迁移成本。

## 概念

### 三个层级

**超大规模云平台**：AWS Bedrock、Azure AI Foundry（原 Azure OpenAI Service）、GCP Vertex AI。最广泛的模型目录（每个平台 20-50+ 个模型）。原生 IAM（身份与访问管理，Identity and Access Management）、VPC（虚拟私有云，Virtual Private Cloud）网络、审计日志。SOC 2、HIPAA、ISO 27001、PCI-DSS 合规认证。每 token 价格比专业平台高 15-30%。模型发布滞后 2-6 周（超大规模云平台在 OpenAI/Anthropic 发布后进行集成测试）。适合：受监管行业、已有云支出承诺的企业、需要 VPC 内数据驻留的场景。

**数据/AI 平台**：Databricks Mosaic AI。在现有数据湖仓之上提供统一的治理目录、特征存储和模型服务。模型在 Databricks 工作流中原生提供。适合：已经在 Databricks 上做 ETL（提取-转换-加载，Extract-Transform-Load）和训练的团队；需要模型服务与特征存储之间紧密集成的场景。

**专业推理平台**：Together AI、Fireworks AI、Groq。在原始推理吞吐量和延迟方面领先。Together 和 Fireworks 在开源模型（Llama、Mixtral、DeepSeek）上提供比超大规模云平台低 30-50% 的每 token 价格。Groq 的 LPU 硬件专为 transformer 推理设计 — 确定性计算调度消除了 KV 缓存（Key-Value 缓存）碎片化。基准数据：Llama 3.1 70B 上 TTFT < 50ms。适合：延迟敏感型应用、成本敏感型大批量工作负载、开源模型优先策略。

### 选择框架

| 约束 | 首选平台 | 原因 |
|------|---------|------|
| 延迟 < 50ms TTFT | Groq | LPU 硬件，确定性调度 |
| 成本最低（开源模型） | Together / Fireworks | 比超大规模云平台低 30-50% |
| HIPAA / SOC 2 / VPC | Bedrock / Azure / Vertex | 原生合规 + 网络 |
| 数据湖仓统一 | Databricks Mosaic AI | 治理 + 特征存储 + 服务一体化 |
| 最广泛的模型目录 | Bedrock / Vertex | 每个平台 20-50+ 个模型 |
| 最新模型发布 | 直接 API（OpenAI、Anthropic） | 零日可用；无超大规模云平台延迟 |

### 多平台策略

2026 年的标准模式：在 Bedrock/Azure/Vertex 上运行基础模型以满足合规和 VPC 网络要求；在 Together/Fireworks 上运行微调后的开源模型以控制成本；在 Groq 上运行延迟关键路径。统一它们的是 AI 网关（第 17 阶段 · 19）— 一个 OpenAI 兼容的 API，按策略路由到不同平台。

### 你应该记住的数字

- 超大规模云平台价格溢价：比专业平台高 15-30%。
- 超大规模云平台模型发布滞后：2-6 周。
- Groq LPU TTFT：Llama 3.1 70B 上 < 50ms。
- Together/Fireworks 成本优势：比超大规模云平台低 30-50%。
- 三个层级：超大规模云平台、数据/AI 平台、专业推理平台。

## 动手实践

`code/main.py` 是一个平台选择决策树：给定延迟预算、合规要求和成本约束，输出推荐平台及理由。

## 交付物

本课产出 `outputs/skill-platform-picker.md`。给定产品约束，选择平台并说明多平台策略。

## 练习

1. 运行 `code/main.py`，输入延迟预算 100ms、HIPAA 合规、成本敏感。输出是什么？
2. 你的产品需要 SOC 2 + 欧洲数据驻留 + < 200ms TTFT。选择平台并说明理由。
3. 比较 Bedrock 上的 Claude 3.5 Sonnet 与直接 Anthropic API：什么情况下超大规模云平台的溢价是合理的？
4. 设计一个多平台策略：面向客户的聊天（低延迟）、内部文档处理（大批量）、合规归档（审计日志）。
5. 论证 Groq 的 LPU 硬件锁定是否值得用它换取 10 倍的云积分灵活性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 超大规模云平台（Hyperscaler） | "云提供商" | AWS/Azure/GCP 的托管 AI 平台 |
| Bedrock | "AWS 的 AI" | AWS 托管模型服务，VPC 原生 |
| Azure AI Foundry | "Azure 的 AI" | 原 Azure OpenAI Service；企业 AI 平台 |
| Vertex AI | "GCP 的 AI" | Google Cloud 的托管 AI 平台 |
| Together AI | "开源推理" | 专业推理平台，开源模型 |
| Fireworks AI | "快速推理" | 专业推理平台，低延迟 |
| Groq | "LPU 公司" | 定制硬件（LPU）用于 transformer 推理 |
| LPU | "Groq 芯片" | 语言处理单元（Language Processing Unit），确定性调度 |
| 多平台（Multi-platform） | "混合策略" | 跨多个提供商使用模型 |
| 数据驻留（Data residency） | "数据留在本地" | 数据不离开特定区域或 VPC |
