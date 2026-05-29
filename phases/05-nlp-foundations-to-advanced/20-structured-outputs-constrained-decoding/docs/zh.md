# 结构化输出与约束解码（Structured Outputs & Constrained Decoding）

> 让大语言模型（LLM）返回 JSON。大多数时候它能返回 JSON。但在生产环境中，"大多数时候"就是问题所在。约束解码（Constrained Decoding）通过在采样前编辑 logits，将"大多数时候"变成"始终如此"。

**类型（Type）：** Build
**语言（Languages）：** Python
**前置要求（Prerequisites）：** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**时间（Time）：** ~60 minutes

## 问题所在（The Problem）

一个分类器（classifier）提示 LLM："返回 {positive, negative, neutral} 中的一个。"模型却返回了"The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."。你的解析器（parser）崩溃了。你的分类器 F1 分数为 0.0。

自由形式的生成（free-form generation）不是一份契约（contract），它只是一个建议。生产系统需要的是一份契约。

2026 年存在三个层次。

1. **提示工程（Prompting）。** 礼貌地请求。"只返回 JSON 对象。"在前沿模型上大约有 80% 的成功率，在较小模型上则更低。
2. **原生结构化输出 API（Native structured output APIs）。** OpenAI 的 `response_format`、Anthropic 的工具使用（tool use）、Gemini 的 JSON 模式（JSON mode）。在支持的 schema 上可靠，但被供应商锁定（vendor-locked）。
3. **约束解码（Constrained decoding）。** 在每个生成步骤修改 logits，使模型*无法*生成无效的 token。从构造上保证 100% 有效。适用于任何本地模型。

本课将建立对这三种方式的直觉，并说明何时该选择哪一种。

## 核心概念（The Concept）

![约束解码在每一步屏蔽无效 token](../assets/constrained-decoding.svg)

**约束解码的工作原理。** 在每个生成步骤中，LLM 在整个词汇表（约 10 万个 token）上产生一个 logit 向量。一个 *logit 处理器（logit processor）* 位于模型和采样器（sampler）之间。它根据当前在目标语法（grammar）中的位置——JSON Schema、正则表达式（regex）、上下文无关文法（context-free grammar）——计算哪些 token 是有效的，并将所有无效 token 的 logit 设为负无穷。对剩余 logit 进行 softmax 后，概率质量只落在有效的续写上。

2026 年的实现方案：

- **Outlines。** 将 JSON Schema 或正则表达式编译为有限状态机（finite-state machine，FSM）。每个 token 的合法下一个 token 查找复杂度为 O(1)。基于 FSM，因此递归 schema 需要展平（flattening）。
- **XGrammar / llguidance。** 上下文无关文法（CFG）引擎。处理递归 JSON Schema。解码开销几乎为零。OpenAI 在其 2025 年结构化输出实现中致谢了 llguidance。
- **vLLM 引导解码（guided decoding）。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，通过 Outlines、XGrammar 或 lm-format-enforcer 后端实现。
- **Instructor。** 基于 Pydantic 的包装器，可封装任何 LLM。在验证失败时重试。跨供应商，但不修改 logits——它依赖重试和结构化输出感知的提示。

### 反直觉的结果

约束解码通常比非约束生成*更快*。有两个原因。第一，它缩小了下一个 token 的搜索空间。第二，巧妙的实现会跳过强制 token 的生成过程（如 `{"name": "` 这样的脚手架——每个字节都是确定的）。

### 让你付出代价的陷阱

字段顺序很重要。把 `answer` 放在 `reasoning` 前面，模型会在思考之前就做出回答。JSON 是有效的，但答案是错的。没有任何验证能发现这个问题。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 的字段顺序是逻辑，不是格式。

## 动手构建（Build It）

### 步骤 1：从零实现基于正则表达式的约束生成

参见 `code/main.py` 中的独立 FSM 实现。核心思想用 30 行代码表达：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪我们到目前为止已满足语法的哪些部分。`valid_tokens(state, tokenizer)` 计算哪些词汇表 token 可以在不离开接受路径的情况下推进 FSM。

### 步骤 2：使用 Outlines 实现 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。永远如此。FSM 使无效输出变得不可达。

### 步骤 3：使用 Instructor 实现跨供应商的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

不同的机制。Instructor 不触及 logits。它将 schema 格式化到提示中，解析输出，并在验证失败时重试（默认 3 次）。适用于任何供应商。重试会增加延迟和成本。跨供应商的可移植性是其卖点。

### 步骤 4：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端约束解码。在支持的 schema 上，可靠性与 Outlines 相当。无需管理本地模型。但将你锁定在供应商上。

## 常见陷阱（Pitfalls）

- **递归 schema。** Outlines 将递归展平到固定深度。树形结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **超大枚举（Huge enums）。** 10,000 个选项的枚举编译缓慢或超时。改用检索器（retriever）：先预测 top-k 候选项，再约束到这些候选项。
- **语法过于严格。** 强制使用 `date: "YYYY-MM-DD"` 正则表达式，模型就无法为缺失的日期输出 `"unknown"`。模型会通过编造一个日期来补偿。允许 `null` 或一个哨兵值（sentinel）。
- **过早承诺（Premature commitment）。** 参见上文的字段顺序陷阱。始终将推理放在前面。
- **不带 schema 的供应商 JSON 模式。** 纯 JSON 模式只保证有效的 JSON，不保证对你的用例*有效*。始终提供完整的 schema。

## 使用指南（Use It）

2026 年的技术栈：

| 场景（Situation） | 选择（Pick） |
|-----------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生供应商结构化输出 |
| 任意供应商，Pydantic 工作流，可容忍重试 | Instructor |
| 本地模型，需要 100% 有效性，扁平 schema | Outlines (FSM) |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM 引导解码 |
| 可接受重试的批处理 | Instructor + 最便宜的模型 |

## 交付物（Ship It）

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习（Exercises）

1. **简单。** 在不使用约束解码的情况下，用一个小型开源权重模型（如 Llama-3.2-3B）对 `Review(sentiment, confidence, evidence_span)` 进行提示。测量 100 条评论中能解析为有效 JSON 的比例。
2. **中等。** 对同一语料库使用 Outlines JSON 模式。比较合规率（compliance rate）、延迟和语义准确性。
3. **困难。** 从零实现一个针对电话号码（`\d{3}-\d{3}-\d{4}`）的正则表达式约束解码器。在 1000 个样本上验证 0 个无效输出。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 实际含义（What it actually means） |
|------|-----------------|-----------------------|
| 约束解码（Constrained decoding） | 强制有效输出 | 在每个生成步骤屏蔽无效 token 的 logit。 |
| Logit 处理器（Logit processor） | 执行约束的那个东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机（Finite-state machine） | 编译后的语法表示；O(1) 合法下一个 token 查找。 |
| CFG | 上下文无关文法（Context-free grammar） | 处理递归的语法；比 FSM 慢但表达能力更强。 |
| Schema 字段顺序（Schema field order） | 这重要吗？ | 是的——第一个字段先提交；始终将推理放在答案前面。 |
| 引导解码（Guided decoding） | vLLM 对此的称呼 | 相同的概念，集成在推理服务器中。 |
| JSON 模式（JSON mode） | OpenAI 的早期版本 | 保证 JSON 语法；不保证 schema 匹配。 |

## 延伸阅读（Further Reading）

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速的基于 CFG 的约束解码。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 参考与注意事项。
- [Instructor library](https://python.useinstructor.com/) — 跨供应商的 Pydantic + 重试。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 对 6 个约束解码框架的基准测试。
