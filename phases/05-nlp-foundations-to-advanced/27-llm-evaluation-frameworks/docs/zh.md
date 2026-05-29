# LLM 评估（Evaluation）— RAGAS、DeepEval、G-Eval

> 精确匹配（Exact-match）和 F1 分数无法捕捉语义等价。人工审查无法规模化。LLM 作为评判者（LLM-as-judge）是生产环境的答案——通过充分的校准（calibration）来信任这个数字。

**类型（Type）：** Build
**语言（Languages）：** Python
**前置知识（Prerequisites）：** Phase 5 · 13（问答 Question Answering）、Phase 5 · 14（信息检索 Information Retrieval）
**时间（Time）：** ~75 分钟

## 问题

你的 RAG 系统回答："June 29th, 2007."
标准参考答案是："June 29, 2007."
精确匹配得分为 0。F1 得分约为 75%。而人类会给出 100 分。

现在乘以 10,000 个测试用例。再乘以检索器（retriever）、分块（chunking）、提示词（prompt）或模型的每一次变更。你需要一个能理解语义、大规模低成本运行、不会谎报回归（regression）、并能揭示正确失败模式（failure mode）的评估器。

2026 年有三个框架主导了这个问题。

- **RAGAS。** 检索增强生成评估（Retrieval-Augmented Generation ASsessment）。四个 RAG 指标（忠实度 faithfulness、答案相关性 answer-relevance、上下文精确度 context-precision、上下文召回率 context-recall），采用 NLI + LLM 评判者后端。有研究支撑，轻量级。
- **DeepEval。** LLM 的 Pytest。G-Eval、任务完成度（task-completion）、幻觉（hallucination）、偏见（bias）指标。CI/CD 原生。
- **G-Eval。** 一种方法（也是 DeepEval 的一个指标）：LLM 作为评判者，结合思维链（chain-of-thought）、自定义标准、0-1 评分。

三者都依赖 LLM 作为评判者。本课将建立对该方法及其信任层（trust layer）的直觉。

## 核心概念

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM 作为评判者。** 用一个 LLM 替代静态指标，根据评分标准（rubric）对输出进行评分。给定 `(query, context, answer)`，提示一个评判 LLM："对忠实度进行 0-1 评分。"返回分数。

为什么有效：LLM 以极低的成本近似人类判断。GPT-4o-mini 每个评分案例约 $0.003，使得 1000 样本的回归评估运行成本低于 $5。

为什么它会静默失败：

1. **评判者偏见（Judge bias）。** 评判者偏好更长的答案、来自自身模型家族的答案、以及匹配提示风格的答案。
2. **JSON 解析失败（JSON parsing failures）。** 错误的 JSON → NaN 分数 → 静默地从聚合结果中排除。RAGAS 用户深知这种痛苦。用 try/except + 显式失败模式进行防护。
3. **模型版本漂移（Drift over model versions）。** 升级评判者会改变所有指标。冻结评判者模型 + 版本。

**RAG 四大指标。**

| 指标（Metric） | 问题（Question） | 后端（Backend） |
|--------|----------|---------|
| 忠实度（Faithfulness） | 答案中的每个声明是否来自检索到的上下文？ | 基于 NLI 的蕴含（entailment） |
| 答案相关性（Answer relevance） | 答案是否回应了问题？ | 从答案生成假设性问题；与真实问题比较 |
| 上下文精确度（Context precision） | 检索到的分块中，有多少是相关的？ | LLM 评判者 |
| 上下文召回率（Context recall） | 检索是否返回了所有需要的内容？ | LLM 评判者对照标准答案 |

**G-Eval。** 定义一个自定义标准："答案是否引用了正确的来源？"框架自动扩展为思维链评估步骤，然后进行 0-1 评分。适用于 RAGAS 未覆盖的领域特定质量维度。

**校准（Calibration）。** 在获得与人工标注的相关性之前，永远不要信任原始的评判者分数。运行 100 个手工标注的示例。绘制评判者 vs 人类对比图。计算 Spearman rho。如果 rho < 0.7，你的评判者评分标准需要改进。

## 动手实践

### 步骤 1：使用 NLI 计算忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子声明（atomic claims）。对每个声明进行 NLI 检查，对照检索到的上下文。忠实度 = 被支持的声明占比。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与所问问题不同，相关性就会下降。

### 步骤 3：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是评分标准。显式步骤比隐式的"评分 0-1"提示更稳定。

### 步骤 4：CI 门禁

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件发布。在每次 PR 上运行。在出现回归时阻止合并。

### 步骤 5：从零构建玩具评估器

参见 `code/main.py`。仅使用标准库近似实现忠实度（答案声明与上下文的交叠）和相关性（答案 token 与问题 token 的交叠）。非生产级。展示基本形态。

## 常见陷阱

- **未校准（No calibration）。** 与人工标注相关性为 0.3 的评判者只是噪声。上线前必须进行校准运行。
- **自我评估（Self-evaluation）。** 使用同一个 LLM 生成和评判会使分数虚高 10-20%。评判者应使用不同的模型家族。
- **成对评判中的位置偏见（Positional bias in pairwise judging）。** 评判者偏好第一个呈现的选项。始终随机化顺序并双向运行。
- **原始聚合值掩盖失败（Raw aggregate hides failures）。** 平均分 0.85 往往掩盖了 5% 的灾难性失败。始终检查底部分位数（bottom quantile）。
- **黄金数据集腐化（Golden dataset rot）。** 未版本化的评估集随时间漂移，破坏了纵向对比。每次变更都要标记数据集。
- **LLM 成本（LLM cost）。** 大规模场景下，评判者调用主导了成本。使用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 使用指南

2026 年技术栈：

| 用例（Use case） | 框架（Framework） |
|---------|-----------|
| RAG 质量监控 | RAGAS（4 个指标） |
| CI/CD 回归门禁 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 在线实时流量监控 | RAGAS 无参考模式（reference-free mode） |
| 人工抽检（Human-in-the-loop spot checks） | LangSmith 或 Phoenix 配合标注界面 |
| 红队测试 / 安全评估 | Promptfoo + DeepEval |

典型技术栈：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新维度。三者都运行；它们的有益分歧能揭示问题。

## 交付成果

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 在 10 个已知存在幻觉的 RAG 示例上使用 RAGAS。验证忠实度指标能捕获每一个。
2. **中等。** 手工标注 50 个 QA 答案的正确性（0-1 分）。用 G-Eval 评分。测量评判者与人类之间的 Spearman rho。
3. **困难。** 用 DeepEval 构建一个 pytest CI 门禁。故意使检索器退化。验证门禁失败。通过对最低 10% 进行阈值检查来添加底部分位数告警。

## 关键术语

| 术语（Term） | 人们怎么说（What people say） | 实际含义（What it actually means） |
|------|-----------------|-----------------------|
| LLM 作为评判者（LLM-as-judge） | 用 LLM 评分 | 提示一个评判模型根据评分标准对输出进行 0-1 评分。 |
| RAGAS | RAG 指标库 | 开源评估框架，包含 4 个无参考 RAG 指标。 |
| 忠实度（Faithfulness） | 答案是否有依据？ | 答案声明中被检索上下文蕴含的比例。 |
| 上下文精确度（Context precision） | 检索到的分块是否相关？ | Top-K 分块中真正有用的比例。 |
| 上下文召回率（Context recall） | 检索是否找到了所有内容？ | 标准答案声明中被检索分块支持的比例。 |
| G-Eval | 自定义 LLM 评判者 | 评分标准 + 思维链评估步骤 + 0-1 评分。 |
| 校准（Calibration） | 信任但要验证 | 评判者分数与人类分数之间的 Spearman 相关性。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 开源生产级技术栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏见、校准与局限性。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 统一框架，集成 RAGAS、DeepEval、Phoenix。
