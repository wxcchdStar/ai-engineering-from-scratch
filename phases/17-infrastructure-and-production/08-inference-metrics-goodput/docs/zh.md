# 推理指标与有效吞吐量 — TTFT、TPOT、TBT 与延迟百分位数

> LLM 推理指标与传统的 Web 服务指标有根本性的不同。**TTFT（首 token 时间，Time To First Token）** 衡量用户感知的响应速度 — 从请求发送到第一个 token 到达的时间。**TPOT（每输出 token 时间，Time Per Output Token）** 衡量生成速度 — 连续 token 之间的平均间隔（不包括第一个 token）。**TBT（Token 间时间，Time Between Tokens）** 是 TPOT 的另一种说法；一些工具（如 NVIDIA GenAI-Perf）将 ITL（Token 间延迟，Inter-Token Latency）定义为不包括 TTFT，而 LLMPerf 将其包括在内 — 两种工具对同一服务器报告不同的 TPOT。**有效吞吐量（Goodput）** 衡量成功完成的请求的吞吐量，与原始吞吐量（throughput）相对 — 原始吞吐量计算所有 token，包括那些因请求被拒绝或超时而浪费的 token。延迟百分位数（P50、P95、P99）至关重要，因为 LLM 延迟具有长尾分布 — P99 可能比 P50 差 5-10 倍。**请求成功率（Request success rate）** 跟踪 2xx 响应 vs 4xx/5xx。**GPU 利用率** 本身具有误导性 — 100% 的 GPU 利用率可能意味着高吞吐量，也可能意味着在重试风暴中重新计算被抢占的 KV 缓存（Key-Value 缓存）。**内存带宽利用率（Memory bandwidth utilization，MBU）** 是更好的指标 — 它衡量 HBM（高带宽内存，High Bandwidth Memory）带宽被有效使用的比例。2026 年的仪表盘标准是：TTFT P50/P95/P99、TPOT P50/P95/P99、有效吞吐量、请求成功率、GPU MBU 百分比。

**类型：** 学习
**语言：** Python（标准库，指标收集器模拟）
**前置要求：** 第 17 阶段 · 04（vLLM 服务内部原理）
**时间：** 约 60 分钟

## 学习目标

- 区分 TTFT（响应速度）、TPOT（生成速度）和 TBT（token 间延迟），并解释为什么 NVIDIA GenAI-Perf 和 LLMPerf 对同一服务器报告不同的 TPOT。
- 解释为什么有效吞吐量（goodput）比原始吞吐量（throughput）更重要，以及被抢占的 KV 缓存如何扭曲原始吞吐量。
- 说出为什么 GPU 利用率具有误导性，以及为什么 MBU 是更好的指标。
- 列举 2026 年仪表盘的五个标准指标。

## 问题

你的监控仪表盘显示 GPU 利用率为 95%，吞吐量为 500 token/秒。一切看起来都很好。然后用户抱怨"聊天机器人很慢"。你检查 TTFT P99：4.2 秒。你的 P50 是 200ms。长尾在杀死用户体验，而你的仪表盘完全没有显示。

LLM 指标需要捕捉流式响应的用户体验。一个 HTTP 200 响应，TTFT 为 4 秒，TPOT 为 50ms，感觉与 TTFT 为 200ms、TPOT 为 50ms 的响应完全不同 — 但两者都计为一次成功的请求。标准 Web 指标（RPS、延迟、错误率）不足以描述 LLM 服务质量。

## 概念

### 核心指标

**TTFT（首 token 时间）**：从请求到达到第一个 token 流式传输到客户端的时间。包括：网络延迟 + 排队 + 预填充时间。用户感知的响应速度。目标：P99 < 500ms 用于对话式 AI。

**TPOT（每输出 token 时间）**：连续输出 token 之间的平均时间（不包括第一个 token）。衡量生成速度。目标：< 50ms 用于流畅的阅读体验（约 20 token/秒）。

**TBT（Token 间时间）**：与 TPOT 同义。关键区别：NVIDIA GenAI-Perf 的 ITL（Token 间延迟）不包括 TTFT；LLMPerf 的 TPOT 包括 TTFT。两种工具对同一服务器报告不同的 TPOT — 在比较基准测试时要检查定义。

**有效吞吐量（Goodput）**：成功完成的请求中生成的 token 数 / 时间。排除：被拒绝的请求、超时、被抢占的 KV 缓存重新计算。原始吞吐量计算所有 token，包括浪费的 token。有效吞吐量衡量用户实际收到的价值。

### 为什么 GPU 利用率具有误导性

场景 A：GPU 以 95% 的利用率处理 50 个并发请求。有效吞吐量 = 450 token/秒。

场景 B：GPU 以 95% 的利用率重新计算被抢占的 KV 缓存（重试风暴）。有效吞吐量 = 50 token/秒。

两种场景的 GPU 利用率都是 95%。但场景 B 是灾难性的。GPU 利用率不能区分有用的工作和浪费的工作。

**MBU（内存带宽利用率）**：衡量 HBM 带宽被有效使用的比例。解码是内存带宽密集型的 — 如果 MBU 较低，说明 GPU 在等待内存，而不是在计算。高 MBU + 低有效吞吐量 = 重试风暴或碎片化。高 MBU + 高有效吞吐量 = 健康。

### 延迟百分位数

LLM 延迟具有长尾分布。一个 8K token 的提示词需要 200ms 的预填充；一个 128K token 的提示词需要 3 秒。P50 看起来不错；P99 暴露了长尾。

标准仪表盘：TTFT P50/P95/P99、TPOT P50/P95/P99。P99 是 SLO（服务等级目标，Service Level Objective）的指标。P50 是用户典型体验的指标。

### 请求成功率

2xx / 总请求数。按错误类型细分：429（限流）、5xx（服务器错误）、超时。429 率上升 → 扩容或增加限流阈值。5xx 率上升 → 事件响应。

### 2026 年仪表盘标准

1. **TTFT** P50/P95/P99（响应速度）。
2. **TPOT** P50/P95/P99（生成速度）。
3. **有效吞吐量**（token/秒，仅成功请求）。
4. **请求成功率**（按状态码细分）。
5. **GPU MBU 百分比**（实际内存带宽利用率）。

### 你应该记住的数字

- TTFT P99 目标：对话式 AI < 500ms。
- TPOT 目标：< 50ms（约 20 token/秒）。
- P99 可能比 P50 差 5-10 倍。
- GPU 利用率 ≠ 健康。MBU 是更好的指标。
- 有效吞吐量排除浪费的 token。

## 动手实践

`code/main.py` 模拟一个包含长尾延迟的 LLM 服务端点。收集 TTFT、TPOT、有效吞吐量和 MBU。

## 交付物

本课产出 `outputs/skill-metrics-dashboard.md`。给定 SLA 目标，设计指标仪表盘和告警规则。

## 练习

1. 运行 `code/main.py`。P50 TTFT 为 180ms，P99 为 3.8 秒。诊断长尾。
2. 计算有效吞吐量 vs 原始吞吐量：30% 的请求超时并重试。
3. GPU 利用率 92%，MBU 45%。诊断问题。
4. GenAI-Perf 报告 TPOT=6ms；LLMPerf 报告 TPOT=11ms。解释差异。
5. 为对话式 AI 产品设计 SLO：TTFT P99 < 500ms，TPOT P99 < 80ms，成功率 > 99.5%。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| TTFT | "首 token 延迟" | 从请求到第一个 token 的时间 |
| TPOT | "生成速度" | 每输出 token 的平均时间 |
| TBT | "token 间隔" | Token 间时间；与 TPOT 同义 |
| ITL | "NVIDIA 的 TBT" | Token 间延迟，不包括 TTFT |
| 有效吞吐量（Goodput） | "有用吞吐量" | 仅成功请求的吞吐量 |
| 原始吞吐量（Throughput） | "总吞吐量" | 包括浪费的 token |
| MBU | "内存带宽利用率" | HBM 带宽被有效使用的比例 |
| P99 | "尾部延迟" | 99% 的请求在此延迟以下 |
| SLO | "服务等级目标" | 服务可靠性的目标值 |
