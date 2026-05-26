# LLM 路由层 — LiteLLM、OpenRouter、Portkey

> 提供商锁定代价高昂。不同的工具调用工作负载适配不同的模型。路由网关提供统一的 API 表面、重试、故障转移、成本追踪和护栏 (guardrails)。三种原型主导 2026 年：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。本课列出决策标准并走读一个标准库路由网关。

**类型：** 学习
**语言：** Python（标准库，路由 + 故障转移 + 成本追踪器）
**前置课程：** Phase 13 · 02（函数调用），Phase 13 · 17（网关）
**时长：** 约 45 分钟

## 学习目标

- 区分自托管、托管式和生产级路由选项。
- 实现按定义优先级顺序在提供商故障时重试的故障转移链。
- 追踪跨提供商的每次请求成本和令牌使用量。
- 针对给定的生产约束在 LiteLLM、OpenRouter 和 Portkey 之间做出选择。

## 问题背景

提供商路由至关重要的场景：

1. **成本。** Claude Sonnet 的成本是 Haiku 的 3 倍。对于分类任务，Haiku 足够；对于综合任务，Sonnet 值这个价。按请求路由。

2. **故障转移。** OpenAI 有一个小时不正常。每个请求都失败。你希望自动回退到 Anthropic 而无需重新部署。

3. **延迟。** 实时聊天 UI 需要快速首字延迟。批量摘要器不需要。按延迟 SLA 路由。

4. **合规。** 欧盟用户必须留在欧盟区域。按区域路由。

5. **实验。** 在同一工作负载上 A/B 测试两个模型。按测试分组路由。

每次集成都手写这些逻辑很繁琐。一个路由网关提供一个 OpenAI 兼容的 API 并处理其余事宜。

## 核心概念

### OpenAI 兼容代理形态

所有人都使用 OpenAI 形态。路由网关暴露 `/v1/chat/completions`，接受 OpenAI schema，内部代理到 Anthropic / Gemini / Cohere / Ollama / 任何。客户端不关心。

### 模型别名

你的代码用 `our_smart_model` 而不是 `claude-3-5-sonnet-20251022`。网关将别名映射到真实模型。当 Anthropic 发布 Claude 4 时，你在服务端修改别名；代码无需改动。

### 故障转移链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置中定义这些。重试有预算限制，因此故障转移级联不会使成本暴涨。

### 语义缓存

相同或近乎相同的提示词命中缓存而不是提供商。重复 agent 循环的节省可达 30% 到 60%。键值基于向量嵌入；近乎相同的提示词共享缓存槽。

### Guardrails（护栏）

网关级别：

- **PII 脱敏。** 在发送提示词之前进行基于正则或 ML 的脱敏处理。
- **策略违规。** 拒绝包含被禁止内容的提示词。
- **输出过滤。** 清洗补全结果防止泄露。

Portkey 和 Kong 都提供了内置护栏。LiteLLM 将其作为可选项。

### 逐密钥限流

一个 API 密钥 = 一个团队。逐密钥预算防止一个团队消耗共享配额。大多数网关支持此功能。

### 自托管 vs 托管式权衡

| 因素 | LiteLLM（自托管） | OpenRouter（托管式） | Portkey（生产级） |
|------|-------------------|---------------------|-------------------|
| 代码 | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管 |
| 部署 | 部署一个代理 | 注册 | 任意 |
| 提供商 | 100+ | 300+ | 100+ |
| 计费 | 自有密钥 | OpenRouter 信用额度 | 自有密钥 |
| 可观测性 | OpenTelemetry | 仪表盘 | 完整 OTel + PII 脱敏 |
| 最佳适用 | 需要完全控制的团队 | 快速原型开发 | 有合规需求的生产环境 |

LiteLLM 在你有 SRE 团队且需要数据主权时胜出。OpenRouter 在你想要单一订阅且无基础设施负担时胜出。Portkey 在你需要开箱即用的护栏和合规时胜出。

### 成本追踪

每个请求携带 `provider`、`model`、`input_tokens`、`output_tokens`。乘以按模型每令牌价格（从网关维护的价格表中获取）。按用户 / 团队 / 项目聚合。

### MCP 加路由

网关可以同时路由 LLM 调用和 MCP 采样请求。当采样请求的 modelPreferences 偏好特定模型时，网关转换为正确的后端。这正是 Phase 13 · 17（MCP 网关）和本课的路由网关有时合并为一个服务的地方。

### 路由策略

- **静态优先级。** 列表中第一个；出错时回退。
- **负载均衡。** 轮询或加权。
- **成本感知。** 选择满足延迟/质量要求的最便宜模型。
- **延迟感知。** 选择最近 N 分钟内最快的模型。
- **任务感知。** 提示词分类器将编码路由到一个模型，将摘要路由到另一个模型。

## 动手实践

`code/main.py` 实现了一个约 150 行的路由网关：接受 OpenAI 形态的请求，转换为各提供商存根，运行优先级故障转移链，追踪每次请求成本，并对输入应用 PII 脱敏处理。用三种场景运行：正常请求、主提供商中断触发故障转移、PII 泄露被脱敏捕获。

关注点：

- `ROUTES` 字典：别名 -> 按优先级排序的具体提供商列表。
- 故障转移循环在 5xx 时重试。
- 成本追踪器将令牌使用量乘以每种模型单价。
- PII 脱敏模块在转发前清洗 SSN 格式的模式。

## 交付产物

本课产出 `outputs/skill-routing-config-designer.md`。给定工作负载配置（延迟、成本、合规），该技能选择 LiteLLM / OpenRouter / Portkey 并生成路由配置。

## 练习

1. 运行 `code/main.py`。触发中断场景；确认故障转移落到第二个提供商且成本归属正确。

2. 添加语义缓存：提示词的 SHA256 作为查找键；缓存命中即时返回。测量重复调用的成本节省。

3. 添加提示词分类器，将 "code ..." 提示词路由到偏好智能的别名，将 "summarize ..." 提示词路由到偏好速度的别名。

4. 设计逐团队预算：每个团队有月度消费上限；网关一旦达到上限就拒绝请求。选择执行粒度（逐请求或窗口式）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey 文档。说出每个工具提供而其他两个不提供的功能。

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Routing gateway | "LLM 代理" | 位于多个提供商之前的单一 API 表面层 |
| OpenAI-compatible | "使用 OpenAI schema" | 接受 `/v1/chat/completions` 形态，转换为任意后端 |
| Model alias | "our_smart_model" | 代码中的名称，网关将其映射到具体模型 |
| Fallback chain | "重试列表" | 失败时尝试的按序排列的提供商列表 |
| Semantic caching | "提示词向量嵌入缓存" | 键是提示词的向量嵌入；近似重复共享缓存命中 |
| Guardrails | "输入/输出过滤器" | 脱敏 PII，拒绝策略违规 |
| Per-key rate limit | "团队预算" | 限定到 API 密钥的配额 |
| Cost tracking | "每次请求消费" | 聚合令牌使用量 x 每种模型价格 |
| LiteLLM | "开放的代理" | 可自托管的开源路由网关 |
| OpenRouter | "托管 SaaS" | 基于信用额度计费的托管网关 |
| Portkey | "生产级选项" | 开源 + 托管，内置护栏 |

## 延伸阅读

- [LiteLLM — docs](https://docs.litellm.ai/) — 自托管路由网关
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS
- [Portkey — docs](https://portkey.ai/docs) — 带护栏的生产级路由
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 决策指南
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — 供应商调研