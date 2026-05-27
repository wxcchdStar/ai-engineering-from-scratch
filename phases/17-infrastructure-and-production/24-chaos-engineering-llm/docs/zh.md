# LLM 生产混沌工程

> LLM 的混沌工程（Chaos engineering）在 2026 年已成为一门独立的学科。在生产环境中运行实验的前提条件：已定义的 SLI（服务等级指标，Service Level Indicator）/SLO（服务等级目标，Service Level Objective）、追踪+指标+日志可观测性、自动回滚、runbook、值班。架构有四个平面：控制平面（control plane，实验调度器）、目标平面（target plane，服务、基础设施、数据存储）、安全平面（safety plane，guard + 中止 + 流量过滤器）、可观测平面（observability plane，指标 + 追踪 + 日志）、反馈（feedback，输入 SLO 调整）。Guardrail 是强制性的：燃烧率告警（burn-rate alerts）在每日错误预算燃烧 > 预期 2 倍时暂停实验；抑制窗口（suppression windows）+ trace-ID 关联去重告警噪声。节奏：每周小型金丝雀 + SLO 审查；每月游戏日（game day）+ 事后分析；每季度跨团队韧性审计 + 依赖映射。LLM 特定实验：内存过载、网络故障、供应商中断、畸形提示词、KV 缓存（Key-Value 缓存）驱逐风暴。工具：Harness Chaos Engineering（LLM 衍生建议、爆炸半径缩减、MCP 工具集成）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes 原生）。

**类型：** 学习
**语言：** Python（标准库，混沌实验运行器）
**前置要求：** 第 17 阶段 · 23（AI 的 SRE）、第 17 阶段 · 13（可观测性）
**时间：** 约 60 分钟

## 学习目标

- 说出五个混沌工程前提条件（SLI/SLO、可观测性、回滚、runbook、值班）并解释为什么跳过任何一个都会破坏实践。
- 绘制四个平面（控制、目标、安全、可观测）以及输入 SLO 的反馈循环。
- 列举五个 LLM 特定实验（内存过载、网络故障、供应商中断、畸形提示词、KV 驱逐风暴）。
- 根据堆栈选择工具 — Harness、LitmusChaos、Chaos Mesh。

## 问题

传统堆栈中的混沌测试已经成熟。LLM 堆栈增加了新的失败模式。一个带有毒字符的 4K token 提示词使 tokenizer 停滞 12 秒。上游供应商返回 429；你的网关重试；你的服务在重试放大的并发下 OOM（内存溢出，Out Of Memory）。突发负载下的 KV 缓存驱逐风暴导致重新预填充级联，使计算饱和。

这些都不会在单元测试中出现。混沌工程是你在用户之前发现它们的方式。

## 概念

### 前提条件

没有以下条件，不要在生产环境中运行混沌实验：

1. **SLI/SLO** — 已定义的服务等级指标和目标。
2. **可观测性（Observability）** — 追踪、指标、日志，连接到仪表盘。
3. **自动回滚（Automated rollback）** — 第 17 阶段 · 20 策略标志回滚。
4. **Runbook** — 结构化的，第 17 阶段 · 23。
5. **值班（On-call）** — 有人响应。

缺少任何一个都意味着混沌变成真实事件。

### 四个平面 + 反馈

**控制平面（Control plane）** — 实验调度器（Litmus 工作流、Chaos Mesh 调度、Harness UI）。

**目标平面（Target plane）** — 服务、Pod、节点、负载均衡器、数据存储。

**安全平面（Safety plane）** — 终止开关、抑制窗口、爆炸半径限制、错误预算关口。

**可观测平面（Observability plane）** — 正常指标 + trace-ID 关联，以区分混沌引起的故障和自然故障。

**反馈循环（Feedback loop）** — 发现结果反馈到 SLO 调整、runbook 更新、代码修复。

### Guardrail 是强制性的

- **燃烧率告警（Burn-rate alert）**：如果每日错误预算燃烧超过预期的 2 倍，暂停实验。
- **抑制窗口（Suppression windows）**：在实验期间静默爆炸半径内的非实验告警。
- **Trace-ID 关联（Trace-ID correlation）**：所有实验引起的错误携带一个标签，以便值班人员可以去重。

### 五个 LLM 特定实验

1. **内存过载（Memory overload）** — 通过以高并发发送长上下文请求来强制 KV 缓存抢占风暴。观察：服务是优雅降级还是崩溃？

2. **网络故障（Network failure）** — 切断推理网关和供应商之间的连接。观察：回退是否在 SLA 内启动？（第 17 阶段 · 19）

3. **供应商中断模拟（Provider outage simulation）** — OpenAI 100% 429。观察：路由是否故障转移到 Anthropic？（第 17 阶段 · 16、19）

4. **畸形提示词（Malformed prompt）** — 注入使 tokenizer 停滞的负载（例如深度嵌套的 Unicode、巨大的 UTF-8 码点）。观察：单个请求是否会锁定一个 worker？

5. **KV 驱逐风暴（KV eviction storm）** — 通过饱和 vLLM 块预算来强制驱逐。观察：LMCache 是否恢复，还是服务降级？

### 节奏

- **每周** — 在预发布环境中进行小型金丝雀实验，可能 5% 生产流量。
- **每月** — 针对特定场景安排游戏日；跨团队参与；事后分析。
- **每季度** — 跨团队韧性审计；依赖映射更新。

### 工具

- **Harness Chaos Engineering** — 商业；AI 衍生实验建议；爆炸半径缩减；MCP 工具集成。
- **LitmusChaos** — CNCF 毕业；基于 Kubernetes 工作流。
- **Chaos Mesh** — CNCF 沙箱；Kubernetes 原生 CRD（自定义资源定义，Custom Resource Definition）风格。
- **Gremlin** — 商业；广泛支持。
- **AWS FIS** / **Azure Chaos Studio** — 托管云产品。

### 从小处开始

第一个实验：在稳定流量下杀死一个解码副本。观察重新路由和恢复。如果这有效且看起来安全，逐步升级到网络混沌。

第一个 LLM 特定实验：注入一个供应商 429 持续 5 分钟。观察回退。大多数团队发现他们的回退没有经过充分测试。

### 你应该记住的数字

- 四个平面：控制、目标、安全、可观测。
- 燃烧率暂停：预期每日预算燃烧的 2 倍。
- 节奏：每周金丝雀、每月游戏日、每季度审计。
- 五个 LLM 实验：内存、网络、供应商、畸形提示词、KV 风暴。

## 动手实践

`code/main.py` 模拟三个带有安全平面关口的混沌实验。报告哪些实验会触发燃烧率中止。

## 交付物

本课产出 `outputs/skill-chaos-plan.md`。给定堆栈和成熟度，选择前三个实验和工具。

## 练习

1. 运行 `code/main.py`。哪个实验触发了燃烧率关口，为什么？
2. 为基于 vLLM 的 RAG（检索增强生成，Retrieval-Augmented Generation）服务设计前五个混沌实验。包括成功标准。
3. 你的燃烧率告警暂停了一个实验。你如何确定根因 — 混沌还是自然？
4. 论证混沌应该在生产环境还是仅在预发布环境中运行。什么时候生产环境是正确的答案？
5. 说出三个通用网络混沌无法复现的 LLM 特定失败模式。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| SLI / SLO | "服务目标" | 指标 + 目标；必需的前提条件 |
| 爆炸半径（Blast radius） | "范围" | 受实验影响的服务/用户集合 |
| 燃烧率告警（Burn-rate alert） | "预算关口" | 当错误预算燃烧率 > 预期 2 倍时触发 |
| 游戏日（Game day） | "每月演练" | 安排的跨团队混沌练习 |
| LitmusChaos | "CNCF 工作流" | 毕业的 CNCF Kubernetes 混沌工具 |
| Chaos Mesh | "CNCF CRD" | CNCF 沙箱 Kubernetes 原生混沌 |
| Harness CE | "商业 AI 辅助" | 具有 AI 建议的 Harness 混沌 |
| 畸形提示词（Malformed prompt） | "tokenizer 炸弹" | 使 tokenization 停滞的输入 |
| KV 驱逐风暴（KV eviction storm） | "抢占级联" | 大规模驱逐触发重新预填充 |
