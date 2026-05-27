# 实战项目 11 — LLM 可观测性与评估仪表盘

> Langfuse 走向开放核心。Arize Phoenix 发布了 2026 年 GenAI 语义约定（semconv）映射。Helicone 和 Braintrust 都加倍投入每用户成本归因。Traceloop 的 OpenLLMetry 成为事实上的 SDK 插桩标准。生产形态是：ClickHouse 用于追踪，Postgres 用于元数据，Next.js 用于 UI，以及一组小型评估作业（DeepEval、RAGAS、LLM 评委）在采样追踪上运行。构建一个自托管的，从至少四个 SDK 系列摄入数据，并演示在五分钟内捕获注入的回退。

**类型：** 实战项目（Capstone）
**语言：** TypeScript（UI）、Python / TypeScript（摄入 + 评估）、SQL（ClickHouse）
**前置课程：** 第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及阶段：** P11 · P13 · P17 · P18
**时间：** 25 小时

## 问题

2026 年，每个运行生产流量的 AI 团队都在模型旁边维护一个可观测性平面。成本归因。幻觉检测。漂移监控。越狱信号。SLO 仪表盘。PII 泄露告警。开源参考——Langfuse、Phoenix、OpenLLMetry——汇聚到 OpenTelemetry GenAI 语义约定作为摄入模式。你现在可以使用一个 SDK 插桩 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM，并发送兼容的 span。

你将构建一个自托管仪表盘，从至少四个 SDK 系列摄入数据，在采样追踪上运行一小组评估作业，检测漂移，并发出告警。衡量标准：给定一个故意注入的回退（一个开始产生 PII 的提示），仪表盘在五分钟内捕获它并触发告警。

## 概念

摄入是 OTLP HTTP。SDK 生成 GenAI 语义约定 span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Span 落入 ClickHouse 用于列式分析；元数据（用户、会话、应用）落入 Postgres。

评估作为批处理作业在采样追踪上运行。DeepEval 评分忠实度、毒性和答案相关性。RAGAS 在追踪携带检索上下文时评分检索指标。自定义 LLM 评委运行领域特定检查（PII 泄露、偏离策略的响应）。评估运行将评估 span 写回同一个 ClickHouse，链接到父追踪。

漂移检测监控嵌入空间分布随时间的变化（提示嵌入上的 PSI 或 KL 散度）以及评估分数趋势。告警馈送到 Prometheus Alertmanager，然后到 Slack/PagerDuty。UI 是 Next.js 15 配合 Recharts。

## 架构

```
生产应用：
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK 配合 GenAI 语义约定
       |
       v  OTLP HTTP
  收集器（摄入、采样、扇出）
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 归档
   （span）      （元数据）   （原始事件）
       |
       +---> 评估作业（DeepEval、RAGAS、LLM 评委）
       |     采样或全量追踪
       |     将评估 span 写回
       |
       +---> 漂移检测器（提示嵌入上的 PSI / KL）
       |
       +---> Prometheus 指标 -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 仪表盘（Recharts）
```

## 技术栈

- 摄入：OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输
- 收集器：OpenTelemetry Collector 配合尾部采样处理器（用于成本控制）
- 存储：ClickHouse 用于 span，Postgres 用于元数据，S3 用于原始事件归档
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix 评估器包、自定义 LLM 评委
- 漂移：每周在池化提示嵌入（sentence-transformers）上计算 PSI / KL
- 告警：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + 服务器操作
- 开箱即用支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 构建步骤

1. **收集器配置。** OpenTelemetry Collector 配合 OTLP HTTP 接收器，尾部采样器保留 100% 的错误追踪和 10% 的成功追踪，以及导出器到 ClickHouse 和 S3。

2. **ClickHouse 模式。** 表 `spans`，列镜像 GenAI 语义约定：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，加上用于长负载的 JSON 包。按 user_id 和 app_id 添加二级索引。

3. **SDK 覆盖率测试。** 使用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）编写一个小型客户端应用，配合 OpenLLMetry 自动插桩。验证每个都生成落入 ClickHouse 的规范 GenAI span。

4. **评估作业。** 一个定时作业读取最近 15 分钟的采样追踪，运行 DeepEval 忠实度、毒性和答案相关性。输出是链接到父追踪的评估 span。

5. **自定义 LLM 评委。** 一个 PII 泄露评委：给定一个响应，调用防护 LLM 评分 PII 泄露的可能性。高分响应落入分类队列。

6. **漂移检测。** 每周作业计算本周池化提示嵌入与过去 4 周基线的 PSI。如果 PSI 高于阈值，告警。

7. **仪表盘。** Next.js 15 页面：概览（spans/秒、成本/用户、p95 延迟）、追踪（搜索 + 瀑布流）、评估（忠实度趋势、毒性）、漂移（PSI 随时间变化）、告警。

8. **告警链。** Prometheus 导出器读取评估分数聚合和延迟百分位数；Alertmanager 路由到 Slack 用于警告，PagerDuty 用于严重违规。

9. **回退探测。** 注入一个错误：被评估的聊天机器人开始 1% 的时间泄露虚假 SSN。测量 MTTR：从错误部署到 Slack 告警。

## 使用方式

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[收集器]   已接受 1 个追踪，3 个 span
[ClickHouse] 已插入 3 个 span（app=chat，user=u_42）
[评估]     DeepEval 忠实度 0.82，毒性 0.03
[漂移]     每周 PSI 0.08（低于 0.2 阈值）
[UI]       在线：https://obs.example.com
```

## 交付标准

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM 应用，仪表盘摄入其追踪，运行评估，对漂移发出告警，并在 Next.js 中展示成本/用户细分。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 追踪模式覆盖率 | 生成规范 GenAI span 的 SDK 系列数量（目标：6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数 vs 手工标注集 |
| 20 | 仪表盘 UX | 注入回退的 MTTR（目标低于 5 分钟） |
| 20 | 成本/规模 | 在 1k spans/秒下持续摄入而无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链端到端演练 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义插桩。验证规范 span 以忠实的 `gen_ai.*` 属性落入 ClickHouse。
2. 在相同追踪上将 DeepEval 替换为 Phoenix 评估器。测量两个评估引擎之间的分数漂移。
3. 锐化漂移检测器：按 app-id 而非全局计算 PSI。显示每应用漂移轨迹。
4. 添加"用户影响"页面：每用户成本和每用户失败率，带迷你趋势图。
5. 构建一个尾部采样策略，保留 100% 毒性 > 0.5 的追踪，加上其余追踪的 10% 分层采样。测量引入的采样偏差。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| GenAI 语义约定 | "OTel LLM 属性" | 2025 年 OpenTelemetry 用于 LLM span 属性的规范（系统、模型、token） |
| 尾部采样 | "追踪后采样" | 收集器在追踪完成后决定保留或丢弃（可以查看错误） |
| PSI | "群体稳定性指数" | 比较两个分布的漂移指标；> 0.2 通常表示有意义的漂移 |
| LLM 评委 | "评估即模型" | 一个 LLM 根据评分标准（忠实度、毒性、PII）对另一个 LLM 的输出进行评分 |
| 尾部采样策略 | "保留规则" | 决定哪些追踪保留 vs 丢弃的规则；错误 + 采样率 |
| 评估 span | "链接的评估追踪" | 携带评估分数的子 span，链接到原始 LLM 调用 span |
| 每用户成本 | "单位经济学" | 在一个时间窗口内归因到 user_id 的美元成本；关键产品指标 |

## 扩展阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考开放核心可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 备选参考，具有强大的漂移支持
- [OpenLLMetry（Traceloop）](https://github.com/traceloop/openllmetry) — 自动插桩 SDK 系列
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 摄入模式
- [Helicone](https://www.helicone.ai) — 备选托管可观测性
- [Braintrust](https://www.braintrust.dev) — 备选评估优先平台
- [ClickHouse 文档](https://clickhouse.com/docs) — 列式 span 存储
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估器库
