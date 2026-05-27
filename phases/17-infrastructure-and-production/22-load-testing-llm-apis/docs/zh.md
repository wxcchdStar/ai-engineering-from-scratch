# LLM API 负载测试 — 为什么 k6 和 Locust 会撒谎

> 传统负载测试工具不是为流式响应、可变输出长度、token 级别指标或 GPU 饱和而设计的。两个陷阱困扰着大多数团队。**GIL 陷阱**：Locust 的 token 级别测量在 Python GIL（全局解释器锁，Global Interpreter Lock）下运行 tokenization，这在高并发下与请求生成竞争；tokenization 积压然后膨胀报告的 token 间延迟 — 你的客户端是瓶颈，而不是服务器。**提示词均匀性陷阱**：在循环中使用相同的提示词只测试 token 分布上的一个点；真实流量具有可变长度和多样化的前缀匹配。LLMPerf 使用 `--mean-input-tokens` + `--stddev-input-tokens` 来解决这个问题。2026 年的工具映射：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token 级别精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）** — 流式感知，通过 TestRun/PrivateLoadZone CRD（自定义资源定义，Custom Resource Definition）实现 Kubernetes 原生分布式测试，最适合 CI/CD 检查点；Vegeta 用于 Go 恒定速率饱和测试；Locust 2.43.3 仅在使用 LLM-Locust 扩展进行流式处理时可用。负载模式：稳态、斜坡、尖峰（自动扩缩容测试）、浸泡（内存泄漏）。

**类型：** 构建
**语言：** Python（标准库，真实提示词生成器 + 延迟收集器）
**前置要求：** 第 17 阶段 · 08（推理指标）、第 17 阶段 · 03（GPU 自动扩缩容）
**时间：** 约 75 分钟

## 学习目标

- 解释使通用负载测试工具对 LLM API 撒谎的两个反模式（GIL 陷阱、提示词均匀性陷阱）。
- 为给定目的选择工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI 检查点）、guidellm（大规模合成）、GenAI-Perf（NVIDIA 参考）。
- 设计四种负载模式（稳态、斜坡、尖峰、浸泡）并说出每种模式捕获的失败模式。
- 使用输入 token 的均值 + 标准差而非固定长度构建真实提示词分布。

## 问题

你用 k6 在 500 并发用户下测试了你的 LLM 端点。它撑住了。你上线了。在生产环境中，200 个实际用户就让服务崩溃了 — P99 TTFT（首 token 时间，Time To First Token）爆炸，GPU 满载。

发生了两件事。首先，k6 发送了 500 个相同的提示词 — 你的请求合并和前缀缓存使它看起来像是在处理 500 个并发解码，而实际上你只处理了一个。其次，k6 不像眼睛体验那样跟踪流式响应上的 token 间延迟；它看到一个 HTTP 连接，而不是 500 个以不同间隔到达的 token。

LLM 的负载测试是一门独立的学科。

## 概念

### GIL 陷阱（Locust）

Locust 使用 Python 并在客户端在 GIL 下运行 tokenization。在高并发下，tokenizer 在请求生成后面排队。报告的 token 间延迟包括客户端 tokenization 积压。你认为服务器很慢；其实是测试工具的问题。

修复：LLM-Locust 扩展将 tokenization 移到单独的进程中，或使用编译语言工具（k6、使用 tokenizers.rs 的 LLMPerf）。

### 提示词均匀性陷阱

所有已知的负载测试工具都让你配置一个提示词。在 10,000 次迭代的循环测试中，每次都发送完全相同的提示词。服务器每次都看到相同的前缀 — 前缀缓存命中率接近 100%，吞吐量看起来很棒。

修复：从提示词分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150` — 多样化的长度，多样化的内容。

### 四种负载模式

1. **稳态（Steady-state）** — 恒定 RPS（每秒请求数，Requests Per Second）持续 30-60 分钟。捕获：基线性能退化。
2. **斜坡（Ramp）** — 在 15 分钟内将 RPS 从 0 线性增加到目标值。捕获：容量断点、预热异常。
3. **尖峰（Spike）** — 突然 3-10 倍 RPS 持续 2 分钟然后恢复。捕获：自动扩缩容延迟、队列饱和、冷启动影响。
4. **浸泡（Soak）** — 稳态持续 4-8 小时。捕获：内存泄漏、连接池漂移、可观测性溢出。

### 2026 年工具映射

**LLMPerf**（Anyscale）— Python 但 Rust 支持的 tokenization。均值/标准差提示词。流式感知。性能运行的最佳默认选择。

**NVIDIA GenAI-Perf** — NVIDIA 的参考工具。使用 Triton 客户端；全面的指标覆盖。注意其 ITL（Token 间延迟，Inter-Token Latency）不包括 TTFT；LLMPerf 包括它。两种工具对同一服务器产生不同的 TPOT（每输出 token 时间，Time Per Output Token）。

**LLM-Locust**（TrueFoundry）— 修复了 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm** — 大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go，编译型，无 GIL）添加了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 进行 Kubernetes 原生分布式测试。
- 最适合 CI/CD 检查点和 SLA（服务等级协议，Service Level Agreement）测试。

**Vegeta** — Go，比 k6 更简单。恒定速率 HTTP 饱和测试。不感知 LLM，但适合网关/限流测试。

**Locust 2.43.3 原生** — 对 LLM 存在 GIL 陷阱。仅在使用 LLM-Locust 扩展时可用。

### CI 中的 SLA 检查点

在 PR 上运行 k6：

- 每次在基线 RPS 下运行 30-50 次迭代。
- 检查点：P50/P95 TTFT、5xx < 5%、TPOT 低于阈值。
- 违规时中断构建。

### 真实提示词分布

从真实流量样本（如果有的话）或已发布的分布（例如用于聊天的 ShareGPT 提示词、用于代码的 HumanEval）构建。将均值 + 标准差提供给 LLMPerf。不惜一切代价避免循环使用单一提示词。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：流式感知指标。
- 典型 LLMPerf 运行：100-1000 个请求，并发 X。
- 典型 CI 检查点：每个 PR 30-50 次迭代。
- 四种模式：稳态、斜坡、尖峰、浸泡。

## 动手实践

`code/main.py` 模拟具有真实提示词分布的负载测试，测量有效 TPOT，并演示均匀提示词陷阱。

## 交付物

本课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。比较均匀分布 vs 真实分布 — 差距在哪里？
2. 为 CI 检查点编写 k6 脚本：100 并发下 TTFT P95 < 800 ms，运行时间 5 分钟。
3. 你的浸泡测试显示内存每小时增长 50 MB。说出三个原因以及区分它们的仪器化方法。
4. 从 10 RPS 到 100 RPS 的尖峰测试。如果 Karpenter + vLLM production-stack 到位（第 17 阶段 · 03 + 18），预期恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 在同一服务器上报告 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| LLMPerf | "LLM 测试工具" | Anyscale 基准测试工具，流式感知 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考测试工具 |
| LLM-Locust | "Locust for LLMs" | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | "合成基准测试" | 大规模合成工具 |
| k6 Operator | "K8s k6" | 基于 CRD 的分布式 k6 |
| GIL 陷阱 | "Python 客户端开销" | Tokenization 积压膨胀报告的延迟 |
| 提示词均匀性陷阱 | "单一提示词谎言" | 循环使用相同提示词命中缓存，膨胀吞吐量 |
| 稳态（Steady-state） | "恒定负载" | 持续 N 分钟的平坦 RPS |
| 斜坡（Ramp） | "线性上升" | 在持续时间内从 0 到目标值 |
| 尖峰（Spike） | "突发测试" | 突然乘数然后恢复 |
| 浸泡（Soak） | "长测试" | 数小时用于泄漏检测 |
