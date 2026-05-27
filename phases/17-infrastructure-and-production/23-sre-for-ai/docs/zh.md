# AI 的 SRE — 多 Agent 事件响应、Runbook 与预测性检测

> AI SRE（站点可靠性工程，Site Reliability Engineering）使用基于基础设施数据（日志、runbook、服务拓扑）通过 RAG（检索增强生成，Retrieval-Augmented Generation）接地的 LLM 来自动化调查、文档和协调阶段。2026 年的架构模式是多 agent 编排 — 由监督者（supervisor）协调的专业化 agent（日志、指标、runbook）；AI 提出假设和查询，人类批准判断性决策。Datadog Bits AI 和 Azure SRE Agent 将其作为托管产品提供。Runbook 正在演进：NeuBird Hawkeye 使用对抗性评估（adversarial evaluation）（两个模型分析同一事件；一致 = 置信度，不一致 = 不确定性）；运维记忆（operational memory）在团队变动中持久化。自动修复（Auto-remediation）保持谨慎：AI 建议，人类批准。完全自主的行动范围很窄（重启 Pod、回滚特定部署），具有严格的 guardrail — 任何推销"设置后就不用管"的人都在过度承诺。新兴前沿：事件前预测（pre-incident prediction）。MIT 研究报告称，在历史日志 + GPU 温度 + API 错误模式上训练的 LLM 预测了 89% 的中断，提前 10-15 分钟。预测：到 2026 年底，95% 的企业 LLM 具有自动故障转移能力。

**类型：** 学习
**语言：** Python（标准库，多 agent 事件分类模拟器）
**前置要求：** 第 17 阶段 · 13（可观测性）、第 17 阶段 · 24（混沌工程）
**时间：** 约 60 分钟

## 学习目标

- 绘制多 agent AI SRE 架构：监督者 + 专业化 agent（日志、指标、runbook）+ 人类批准关口。
- 解释为什么自动修复范围很窄（重启 Pod、回滚部署）而非广泛（重新架构服务）。
- 说出对抗性评估模式（NeuBird Hawkeye）：两个模型一致 = 置信度；不一致 = 升级。
- 引用 MIT 89% 早期检测结果以及运维约束：没有执行的预测只是仪表盘。

## 问题

值班工程师在凌晨 3 点被呼叫。"结账错误率高。"他们检查 Datadog、Loki、三份 runbook、部署日志。30 分钟后，他们意识到根因是 KV 缓存（Key-Value 缓存）尖峰导致的 vLLM OOM（内存溢出，Out Of Memory）。他们重启 Pod；错误清除。

在 2026 年，调查的前 20 分钟是可自动化的。按服务分组日志、与最近部署关联、与 runbook 匹配 — 所有这些都是 RAG + 工具使用。一个受监督的 agent 可以在人类打开 Datadog 之前进行第一轮分类并提出假设。

完全自主的修复是另一个问题。重启 Pod：安全。扩展 GPU 池：如果策略允许则安全。重新架构服务：绝对不行。这门学科就是划定那条窄线。

## 概念

### 多 agent 架构

```
          事件
           │
           ▼
        监督者
        /    |    \
       ▼     ▼     ▼
  日志 agent  指标 agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        假设 + 证据
             │
             ▼
        人类批准
             │
             ▼
        行动（窄范围）
```

监督者将事件分解为子查询。专业化 agent 具有工具访问权限（日志搜索、PromQL、文档检索）。监督者综合，向人类呈现假设 + 证据。人类批准或重定向。

### 自动修复范围

**安全（窄范围）**：重启 Pod、回滚特定部署、在预先批准的范围内扩展池、启用预先批准的功能标志。

**不安全（广泛）**：更改服务拓扑、修改资源限制、部署新代码、更改 IAM（身份与访问管理，Identity and Access Management）、修改数据库。

任何推销"设置后就不用管"的人都在过度承诺。安全集合随着 AI SRE 的成熟而增长，但边界是真实的。

### 对抗性评估（NeuBird Hawkeye）

两个模型独立分析同一事件。如果它们在根因上一致，置信度很高。如果不一致，升级到人类，两个假设都可见。简单的模式，有效的过滤器，防止幻觉根因。

### 运维记忆

团队流动是传统 SRE 的无声杀手 — 部落知识流失。AI SRE 将 runbook + 事后分析存储在向量数据库中；agent 在每个新事件上检索。当新工程师加入时，AI 拥有完整的历史记录。

### 事件前预测

MIT 2025 年研究：在历史日志、GPU 温度、API 错误模式上训练的 LLM 在测试集上预测了 89% 的中断，提前 10-15 分钟。

现实检查：没有执行的预测只是仪表盘。运维问题是"当我们预测时，我们做什么？" 预先排空？呼叫器？自动扩缩容？答案是策略特定的。

### 2026 年的产品

- **Datadog Bits AI** — Datadog 内的托管 SRE 副驾驶。
- **Azure SRE Agent** — Azure 原生。
- **NeuBird Hawkeye** — 对抗性评估 + 运维记忆。
- **PagerDuty AIOps** — 分类 + 去重。
- **Incident.io Autopilot** — 事件指挥官 + 协调。

### Runbook 即代码

Runbook 从 Confluence 页面演进为带有结构化部分的版本化 markdown（症状、假设、验证、行动）。结构化 runbook 提供更好的 RAG 检索。通过将非结构化 runbook 转化为结构化来开始任何 AI-SRE 推广。

### 你应该记住的数字

- MIT 早期检测：89% 的中断，10-15 分钟提前时间。
- 多 agent 分类：监督者 +（日志、指标、runbook）+ 人类。
- 安全自动修复集合：重启 Pod、回滚部署、在范围内扩展。
- 对抗性评估：两个模型独立；一致 = 置信度。

## 动手实践

`code/main.py` 模拟多 agent 分类：日志 agent 发现错误，指标 agent 发现 CPU 尖峰，runbook agent 匹配已知问题。监督者对假设进行排序。

## 交付物

本课产出 `outputs/skill-ai-sre-plan.md`。给定当前值班、事件量、团队成熟度，设计 AI SRE 推广方案。

## 练习

1. 运行 `code/main.py`。如果日志和指标 agent 不一致怎么办？监督者如何解决？
2. 为你的服务定义三个"安全"的自动修复行动。为每个提供理由。
3. 编写结构化 runbook 模板：部分、必填字段、验证命令。
4. 预测性检测在 12 分钟提前时间触发。你的策略是什么 — 呼叫器、预先排空，还是两者？
5. 论证 3 人团队在 2026 年是否应该采用 AI SRE 还是等待。考虑成熟度、量、风险。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| AI SRE | "值班 agent" | LLM 支持的事件调查 + 协调 |
| 监督者 agent（Supervisor agent） | "编排器" | 将事件分解为子查询的顶层 agent |
| 专业化 agent（Specialized agent） | "领域 agent" | 具有工具访问权限的子 agent（日志、指标、runbook） |
| 自动修复（Auto-remediation） | "AI 修复它" | 窄范围预先批准的行动；非广泛重新架构 |
| 运维记忆（Operational memory） | "向量 runbook" | 事后分析 + runbook 在向量数据库中用于 RAG |
| 对抗性评估（Adversarial eval） | "双模型检查" | 独立分析；一致 = 置信度 |
| NeuBird Hawkeye | "对抗性那个" | 具有对抗性评估 + 记忆模式的产品 |
| Bits AI | "Datadog 的 SRE agent" | Datadog 托管的 AI SRE |
| 事件前预测（Pre-incident prediction） | "早期检测" | 中断预测提前 10-15 分钟 |
