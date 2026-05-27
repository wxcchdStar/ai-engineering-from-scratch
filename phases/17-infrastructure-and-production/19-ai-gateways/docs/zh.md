# AI 网关 — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关（Gateway）位于你的应用和模型供应商之间。核心功能是供应商路由、回退、重试、限流、密钥引用、可观测性、guardrail。2026 年的市场分化：**LiteLLM** 是 MIT 开源，支持 100+ 供应商，兼容 OpenAI，但在约 2000 RPS（每秒请求数，Requests Per Second）时崩溃（已发布基准测试中 8 GB 内存，级联故障）；最适合 Python、< 500 RPS、开发/原型设计。**Portkey** 定位为控制平面（guardrail、PII（个人身份信息，Personally Identifiable Information）脱敏、越狱检测、审计追踪），2026 年 3 月以 Apache 2.0 开源，20-40 ms 延迟开销，$49/月生产层。**Kong AI Gateway** 基于 Kong Gateway 构建 — Kong 自己的基准测试在相同 12 CPU 上：比 Portkey 快 228%，比 LiteLLM 快 859%；$100/模型/月定价（Plus 层最多 5 个）；如果你已经在用 Kong，适合企业。**Bifrost**（Maxim AI）— 自动重试，可配置退避，OpenAI 429 时回退到 Anthropic。**Cloudflare / Vercel AI Gateways** — 托管、零运维、基本重试。数据驻留驱动自托管决策；Portkey 和 Kong 处于中间位置，提供开源 + 可选托管。

**类型：** 学习
**语言：** Python（标准库，网关路由模拟器）
**前置要求：** 第 17 阶段 · 01（托管 LLM 平台）、第 17 阶段 · 16（模型路由）
**时间：** 约 60 分钟

## 学习目标

- 列举六项核心网关功能（路由、回退、重试、限流、密钥、可观测性、guardrail）。
- 将四个 2026 年网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到规模上限和用例。
- 引用 Kong 基准测试（比 Portkey 快 228%，比 LiteLLM 快 859%）并解释为什么对 > 500 RPS 很重要。
- 在给定数据驻留和运维预算的情况下，选择自托管 vs 托管。

## 问题

你的产品调用 OpenAI、Anthropic 和一个自托管的 Llama。每个供应商有不同的 SDK、错误模型、限流和认证方案。你想要故障转移（如果 OpenAI 429，尝试 Anthropic）、单一凭证存储、统一可观测性和按租户限流。

在应用层重新发明这些会将每个服务耦合到每个供应商。网关层将其整合到一个进程中，使用一个 API（通常是 OpenAI 兼容的），扇出到供应商。

## 概念

### 六项核心功能

1. **供应商路由（Provider routing）** — OpenAI、Anthropic、Gemini、自托管等，在一个 API 后面。
2. **回退（Fallback）** — 在 429、5xx 或质量失败时，在其他地方重试。
3. **重试（Retries）** — 指数退避，有界尝试。
4. **限流（Rate limits）** — 按租户、按密钥、按模型。
5. **密钥引用（Secret references）** — 在运行时从 vault 拉取凭证（永远不在应用中）。
6. **可观测性（Observability）** — OTel + GenAI 属性（第 17 阶段 · 13）+ 成本归因。
7. **Guardrails** — PII 脱敏、越狱检测、允许主题过滤器。

### LiteLLM — MIT 开源，Python

- 100+ 供应商，兼容 OpenAI，路由器配置，回退，基本可观测性。
- 在 Kong 的基准测试中约 2000 RPS 时崩溃；8 GB 内存占用，持续负载下级联故障。
- 最适合：Python 应用、< 500 RPS、开发/测试网关、实验性路由。
- 成本：开源 $0；存在云免费层。

### Portkey — 控制平面定位

- 2026 年 3 月起 Apache 2.0 开源。Guardrails、PII 脱敏、越狱检测、审计追踪。
- 每请求 20-40 ms 延迟开销。
- $49/月生产层，含保留 + SLA（服务等级协议，Service Level Agreement）。
- 最适合：需要 guardrail + 可观测性捆绑的受监管行业。

### Kong AI Gateway — 规模之选

- 基于 Kong Gateway（成熟的 API 网关产品，lua+OpenResty）构建。
- Kong 自己的基准测试在 12-CPU 等效配置上：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 层最多 5 个。
- 最适合：已经在用 Kong；> 1000 RPS；愿意购买许可证。

### Bifrost（Maxim AI）

- 自动重试，可配置退避。
- OpenAI 429 时回退到 Anthropic 是经典配方。
- 较新的进入者；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管，零运维。基本重试和可观测性。
- 最适合：Cloudflare/Vercel 上的边缘服务 JavaScript 应用。
- 在 guardrail 和限流方面相比 Kong/Portkey 有限。

### 自托管 vs 托管

数据驻留是强制函数。医疗和金融默认自托管（LiteLLM 或 Portkey OSS 或 Kong）。消费者产品默认托管（Cloudflare AI Gateway）或中间层（Portkey 托管）。混合：受监管租户自托管，其他租户托管。

### 延迟预算

- LiteLLM：典型 5-15 ms 开销。
- Portkey：20-40 ms 开销。
- Kong：3-8 ms 开销。
- Cloudflare/Vercel：1-3 ms 开销（边缘优势）。

网关延迟直接加到 TTFT（首 token 时间，Time To First Token）上。对于 TTFT P99 < 100 ms SLA，选择 Kong 或 Cloudflare。对于 P99 < 500 ms，任何都可以。

### 限流语义很重要

简单的令牌桶（token-bucket）在中等规模下有效。多租户需要滑动窗口（sliding-window）+ 突发容忍 + 按租户分层。LiteLLM 提供令牌桶；Kong 提供滑动窗口；Portkey 提供分层限流。

### 网关 + 可观测性 + 路由组合

第 17 阶段 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产中是同一层。选择一个覆盖所有三者的工具，或仔细连接它们：大多数 2026 年部署将 Helicone（可观测性）或 Portkey（guardrail）与 Kong（规模）结合使用，以分担角色。

### 你应该记住的数字

- LiteLLM：约 2000 RPS 时崩溃，8 GB 内存。
- Portkey：20-40 ms 开销；2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 层最多 5 个。
- Cloudflare/Vercel：边缘 1-3 ms 开销。

## 动手实践

`code/main.py` 模拟在注入 429/5xx 的情况下跨 3 个供应商的网关路由和回退。报告延迟、重试率和回退命中率。

## 交付物

本课产出 `outputs/skill-gateway-picker.md`。给定规模、运维姿态、合规、延迟预算，选择网关。

## 练习

1. 运行 `code/main.py`。配置从 OpenAI→Anthropic→自托管的回退。在 5% 供应商错误率下的预期命中率是多少？
2. 你的 SLA 是 TTFT P99 < 200 ms，基线为 300 ms。哪些网关在预算内？
3. 医疗客户要求自托管 + PII 脱敏 + 审计。选择 Portkey OSS 或 Kong。
4. 比较 LiteLLM vs Kong：团队应该在什么 RPS 上限时迁移？
5. 为多租户 SaaS 设计限流策略：免费层、试用层、付费层。令牌桶还是滑动窗口？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 网关（Gateway） | "API 代理" | 位于应用和供应商之间的进程 |
| LiteLLM | "MIT 那个" | Python 开源，100+ 供应商，2K RPS 时崩溃 |
| Portkey | "guardrail 网关" | 控制平面 + 可观测性，Apache 2.0 |
| Kong AI Gateway | "规模那个" | 基于 Kong Gateway 构建，基准测试领先 |
| Bifrost | "Maxim 的网关" | 重试 + Anthropic 回退配方 |
| Cloudflare AI Gateway | "边缘托管" | 边缘部署的托管网关，零运维 |
| PII 脱敏（PII redaction） | "数据清洗" | 在发送到模型之前使用正则 + NER（命名实体识别，Named Entity Recognition）进行掩码 |
| 越狱检测（Jailbreak detection） | "提示词注入 guard" | 对用户输入进行分类器检测 |
| 审计追踪（Audit trail） | "受监管日志" | 每次 LLM 调用的不可变记录 |
| 令牌桶（Token-bucket） | "简单限流" | 基于补充的限流器 |
| 滑动窗口（Sliding-window） | "精确限流" | 基于时间窗口的限流器；更好的公平性 |
