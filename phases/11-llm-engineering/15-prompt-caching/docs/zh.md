# 提示缓存与上下文缓存

> 你的系统提示词有 4,000 个 token。你的 RAG 上下文有 20,000 个 token。每次请求你都要发送这两者，而且每次都要为它们付费。提示缓存让服务商在其侧保持该前缀处于热状态，并在复用时仅按正常费率的 10% 向你计费。使用得当，它可将推理成本降低 50–90%，将首 token 延迟降低 40–85%。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 11 · 01（提示工程）、Phase 11 · 05（上下文工程）、Phase 11 · 11（缓存与成本）
**预计时间：** ~60 分钟

## 问题

一个编程 Agent 在对话的每一轮都向 Claude 发送相同的 15,000 token 系统提示词。以 $3/M 输入 token 的价格计算，20 轮对话仅输入成本就达到 $0.90——这还不包括用户的任何实际消息。乘以每天 10,000 次对话，账单就变成了每天 $9,000，而这些文本从未改变过。

你无法在不影响质量的前提下缩减提示词。你也无法避免发送它——模型每一轮都需要它。唯一的出路是停止为服务商已经见过的前缀支付全价。

这个出路就是提示缓存。Anthropic 于 2024 年 8 月发布了该功能（2025 年又推出了 1 小时扩展 TTL 变体），OpenAI 在同年晚些时候将其自动化，Google 随 Gemini 1.5 一起推出了显式的上下文缓存，如今这三家都将其作为其前沿模型上的一等特性提供。

## 概念

![提示缓存：一次写入，廉价读取](../assets/prompt-caching.svg)

**机制。** 当某个请求的前缀与最近一次请求的前缀匹配时，服务商会直接从上次运行中提供 KV 缓存，而不是重新编码这些 token。你第一次支付少量的写入溢价，之后每次支付大幅的读取折扣。

**2026 年的三种服务商风格。**

| 服务商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL（生存时间） | 最小可缓存量 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 内容块上的显式 `cache_control` 标记 | 输入 9 折 | 25% 附加费 | 5 分钟（可延长至 1 小时） | 1,024 token（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入 5 折 | 无 | 最长 1 小时（尽力而为） | 1,024 token |
| Google（Gemini） | 显式 `CachedContent` API | 按存储计费；读取约正常费率的 25% | 每 token·小时的存储费 | 用户设定（默认 1 小时） | 4,096 token（Flash），32,768（Pro） |

**不变性原则。** 三家都只缓存前缀。如果请求之间有任何 token 不同，从第一个不同的 token 之后的所有内容都是缓存未命中。将*稳定的*部分放在顶部，将*可变的*部分放在底部。

### 缓存友好的布局

```
[系统提示词]          <-- 缓存此项
[工具定义]            <-- 缓存此项
[少样本示例]          <-- 缓存此项
[检索到的文档]        <-- 如果复用则缓存，否则不缓存
[对话历史]            <-- 缓存到上一轮为止
[当前用户消息]        <-- 永远不缓存（每次不同）
```

违反这个顺序——把用户消息放在系统提示词之上、在少样本示例之间穿插动态检索——缓存就永远不会命中。

### 盈亏平衡计算

Anthropic 的 25% 写入溢价意味着一个缓存块至少要被读取两次才能净省钱。1 次写入 + 1 次读取平均每次请求成本为 0.675 倍（节省 32%）；1 次写入 + 10 次读取平均为 0.205 倍（节省 80%）。经验法则：缓存任何你预期在 TTL（生存时间）内至少复用 3 次的内容。

## 构建它

### 步骤 1：使用显式标记的 Anthropic 提示缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将该块存储 5 分钟。在该时间窗口内复用即为命中；过期后复用则重新写入。

**响应 usage 字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 按 1.25 倍计费
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 按 0.1 倍计费
```

在 CI 中检查这两个字段——如果跨请求的 `cache_read_input_tokens` 一直为零，说明你的缓存键正在漂移。

### 步骤 2：一小时扩展 TTL（生存时间）

对于长时间运行的批处理作业，5 分钟的默认值会在作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL（生存时间）的写入溢价为 2 倍（相比基准线的 50% 而非 25%），但在任何复用前缀超过 5 次的批处理中都能快速回本。

### 步骤 3：OpenAI 自动缓存

OpenAI 无需你做任何配置。任何超过 1,024 token 且与最近请求匹配的前缀都会自动获得 5 折优惠。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 长且稳定
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 享受折扣的部分
```

同样的缓存友好布局规则也适用。有两件事会破坏 OpenAI 的缓存但不会影响 Anthropic 的：更改 `user` 字段（用作缓存键的组成部分）和重新排列工具。

### 步骤 4：Gemini 显式上下文缓存

Gemini 将缓存视为你可以创建并命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 在缓存存续期间按每 token·小时收取存储费，读取按正常输入费率的约 25% 计费。当你需要在多天内的许多会话中复用同一个巨型提示词时，这是正确的方案。

### 步骤 5：在生产环境中衡量命中率

参见 `code/main.py`，其中包含一个模拟的三服务商记账器，用于跟踪写入/读取/未命中计数并计算每 1,000 次请求的混合成本。按目标命中率设置部署门禁——大多数生产环境的 Anthropic 设置在预热后应达到 80% 以上的读取占比。

## 2026 年仍然常见的坑

- **顶部动态时间戳。** 系统提示顶部出现 `"当前时间：2026-04-22 15:30:02"`。每次请求都会缓存未命中。把时间戳移到缓存断点之后。
- **工具顺序变化。** 以稳定顺序序列化工具——两次部署之间字典重排会导致每次缓存都未命中。
- **近似重复的自由文本。** "You are helpful." vs "You are a helpful assistant."——一个字节的差异 = 完全未命中。
- **块太小。** Anthropic 强制要求最小 1,024 token（Haiku 为 2,048）。小于此值的块会静默不缓存。
- **无视成本仪表板。** 把"输入 token"拆分为缓存命中和未命中。否则流量下降看起来就像缓存生效了一样。

## 如何使用

2026 年的缓存技术栈：

| 场景 | 选择 |
|-----------|------|
| 带稳定 10k+ 系统提示的多轮 Agent | Anthropic `cache_control`，5 分钟 TTL |
| 批处理任务复用同一前缀超过 30 分钟 | Anthropic，设置 `ttl: "1h"` |
| GPT-5 上的 Serverless 端点，无自定义基础设施 | OpenAI 自动缓存（只需让前缀稳定且足够长） |
| 海量代码/文档语料库的多日复用 | Gemini 显式 `CachedContent` |
| 跨提供商回退 | 保持各提供商的缓存前缀布局一致，这样任何一家命中都能生效 |

结合语义缓存（Phase 11 · 11）用于用户消息层：提示缓存处理*逐 token 完全一致*的复用，语义缓存处理*含义一致*的复用。

## 交付物

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 取一个带 5,000 token 系统提示的 10 轮对话，对 Claude 分别在不使用 `cache_control` 和使用的情况下运行。报告每种情况的输入 token 费用。
2. **中等。** 编写一个测试工具，给定一个提示模板和一个请求日志，计算每个提供商（Anthropic 5 分钟、Anthropic 1 小时、OpenAI 自动、Gemini 显式）的预期命中率和节省金额。
3. **困难。** 构建一个布局优化器：给定一个提示和一组标记为 `stable=True/False` 的字段，重写提示以在最大缓存友好位置放置单个缓存断点，且不丢失信息。在真实的 Anthropic 端点上验证。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 提示缓存 | "让长提示变便宜" | 对匹配前缀复用提供商端的 KV 缓存；重复输入 token 享受 50-90% 折扣。 |
| `cache_control` | "Anthropic 的标记" | 内容块属性，声明"至此为止的所有内容均可缓存"；`{"type": "ephemeral"}`。 |
| 缓存写入 | "支付溢价" | 填充缓存的第一个请求；Anthropic 按约 1.25 倍输入费率计费，OpenAI 免费。 |
| 缓存读取 | "享受折扣" | 后续匹配前缀的请求；按 10%（Anthropic）、50%（OpenAI）、约 25%（Gemini）计费。 |
| TTL | "它能活多久" | 缓存保持热状态的秒数；Anthropic 默认 5 分钟（可延长至 1 小时），OpenAI 尽力而为最长 1 小时，Gemini 由用户设定。 |
| 扩展 TTL | "1 小时 Anthropic 缓存" | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价 2 倍，但对于批处理复用是值得的。 |
| 前缀匹配 | "为什么我的缓存未命中" | 只有当从开头到断点的每个 token 逐字节完全一致时，缓存才会命中。 |
| 上下文缓存（Gemini） | "显式的那种" | Google 的命名、按存储计费的缓存对象；最适合大型语料库的多日复用。 |

## 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1 小时 TTL、盈亏平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 及存储定价。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 原始发布文章，含延迟数据。
- Phase 11 · 05（上下文工程）——在哪里切分提示以便缓存生效。
- Phase 11 · 11（缓存与成本）——将提示缓存与用户消息的语义缓存搭配使用。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — 提示缓存向用户暴露的 KV 缓存内存模型；解释了为什么缓存前缀重读比重算便宜约 10 倍。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill 是提示缓存所缩短的阶段；本文解释了为什么缓存命中时 TTFT 大幅下降而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — 提示缓存与推测解码、Flash Attention、MQA/GQA 并列，是压低推理成本曲线的杠杆；阅读本文了解另外三个。
