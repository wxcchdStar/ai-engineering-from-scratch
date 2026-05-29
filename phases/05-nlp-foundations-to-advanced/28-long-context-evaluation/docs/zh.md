# 长上下文评测（Long-Context Evaluation） — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称支持 1000 万 token 的上下文。但在 100 万 token 时，8 针 MRCR 的准确率降至 26.3%。宣称不等于可用。长上下文评测告诉你正在上线的模型的实际能力。

**类型（Type）：** Learn
**语言（Languages）：** Python
**前置条件（Prerequisites）：** Phase 5 · 13（问答系统 Question Answering），Phase 5 · 23（分块策略 Chunking Strategies）
**时间（Time）：** ~60 分钟

## 问题所在（The Problem）

你有一份 200 页的合同。模型声称支持 100 万 token 的上下文。你把合同粘贴进去并问："终止条款是什么？"模型回答了——但回答的是封面页的内容，因为终止条款位于 12 万 token 深处，已经超出了模型实际能关注到的范围。

这就是 2026 年的上下文容量差距（context-capacity gap）。规格表上写着 100 万或 1000 万，但现实是其中只有 60-70% 是可用的，而且"可用"还取决于任务类型。

- **检索（Retrieval，大海捞针 single needle in haystack）：** 在前沿模型上，在宣称的最大长度范围内几乎完美。
- **多跳/聚合（Multi-hop / aggregation）：** 在大多数模型上，超过约 128k 后性能急剧下降。
- **对分散事实的推理（Reasoning over dispersed facts）：** 最先失败的任务。

长上下文评测衡量这些维度。本课将介绍各个基准测试、它们各自实际衡量什么，以及如何为你的领域构建自定义的"大海捞针"测试。

## 核心概念（The Concept）

![NIAH 基线、RULER 多任务、LongBench 整体评测](../assets/long-context-eval.svg)

**大海捞针测试（Needle-in-a-Haystack，NIAH，2023）。** 在长上下文的受控深度位置放置一个事实（如"魔法词是 pineapple"），然后让模型检索它。对深度 × 长度进行扫描。这是最初的长上下文基准测试。前沿模型现在已能在此测试上达到饱和；它是一个必要但不充分的基线。

**RULER（Nvidia，2024）。** 涵盖 4 个类别的 13 种任务类型：检索（单键/多键/多值）、多跳追踪（Multi-hop tracing，变量追踪 variable tracking）、聚合（常见词频统计）、问答（QA）。可配置上下文长度（4k 到 128k+）。揭示了那些在 NIAH 上饱和但在多跳任务上失败的模型。在 2024 年的发布中，17 个声称支持 32k+ 上下文的模型中，只有一半在 32k 时保持了质量。

**LongBench v2（2024）。** 503 道多选题，8k-2M 词的上下文，六个任务类别：单文档问答（single-doc QA）、多文档问答（multi-doc QA）、长上下文学习（long in-context learning）、长对话（long dialogue）、代码仓库（code repo）、长结构化数据（long structured data）。用于真实世界长上下文行为的生产级基准测试。

**MRCR（多轮共指消解 Multi-Round Coreference Resolution）。** 大规模多轮共指消解。有 8 针、24 针、100 针变体。揭示了模型在注意力退化之前能同时处理多少个事实。

**NoLiMa。** "非词汇重叠针"（Non-lexical needle）。针和查询之间没有字面上的词汇重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 拼接多篇文档，从中任选一篇提问。测试选择性注意力（selective attention）。

**BABILong。** 将 bAbI 推理链嵌入无关的"干草堆"中。测试的是"干草堆中的推理"（reasoning-in-a-haystack），而不仅仅是检索。

### 实际应该报告什么

- **宣称的上下文窗口（Advertised context window）。** 规格表上的数字。
- **有效检索长度（Effective retrieval length）。** NIAH 在某个阈值（如 90%）下通过的长度。
- **有效推理长度（Effective reasoning length）。** 多跳或聚合任务在该阈值下通过的长度。
- **退化曲线（Degradation curve）。** 准确率随上下文长度的变化，按任务类型分别绘制。

你的规格表上需要两个数字：检索有效长度和推理有效长度。通常推理有效长度只有宣称窗口的 25-50%。

## 动手构建（Build It）

### 步骤 1：为你的领域构建自定义 NIAH

参见 `code/main.py`。骨架代码如下：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

对 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k} 进行扫描。绘制热力图。这就是你的目标模型的 NIAH 能力卡。

### 步骤 2：多针变体（Multi-needle variant）

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像"三个魔法词分别是什么？"这样的问题需要检索全部三个。单针成功并不能预测多针成功。

### 步骤 3：多跳变量追踪（Multi-hop variable tracing，RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

回答需要串联三次赋值。前沿模型在 128k 时，此任务的准确率通常降至 50-70%。

### 步骤 4：在你的技术栈上运行 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别报告准确率。聚合分数会掩盖任务级别上的巨大差异。

## 常见陷阱（Pitfalls）

- **仅做 NIAH 评测。** 在 100 万 token 上通过 NIAH 并不能说明多跳能力。务必运行 RULER 或自定义的多跳测试。
- **均匀深度采样。** 许多实现只测试 depth=0.5。请测试 depth=0, 0.25, 0.5, 0.75, 1.0——"迷失在中间"（lost in the middle）效应是真实存在的。
- **与填充文本的词汇重叠。** 如果针与填充文本共享关键词，检索就变得微不足道。使用 NoLiMa 风格的无重叠针。
- **忽略延迟。** 100 万 token 的提示需要 30-120 秒来预填充。在衡量准确率的同时也要衡量首 token 时间（time-to-first-token）。
- **供应商自报的数据。** OpenAI、Google、Anthropic 都会发布自己的分数。务必在你的用例上独立重新运行。

## 实际应用（Use It）

2026 年的技术栈：

| 场景（Situation） | 基准测试（Benchmark） |
|-----------|-----------|
| 快速健全性检查 | 自定义 NIAH，3 个深度 × 3 个长度 |
| 生产环境模型选型 | RULER（13 个任务），在你目标长度下运行 |
| 真实世界问答质量 | LongBench v2 单文档问答子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话/会话场景 | MRCR 8 针，在你目标长度下运行 |
| 模型升级回归测试 | 固定的内部 NIAH + RULER 测试套件，在每个新模型上运行 |

生产环境的经验法则：在你预期的长度上同时通过 NIAH + 至少一项推理任务之前，永远不要信任一个上下文窗口。

## 交付物（Ship It）

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习（Exercises）

1. **简单。** 构建一个 NIAH，3 个深度（0.25, 0.5, 0.75）× 3 个长度（1k, 4k, 16k）。在任意模型上运行。将通过率绘制为 3×3 热力图。
2. **中等。** 添加一个 3 针变体。在每个长度下测量全部 3 针的检索情况。与相同长度下的单针通过率进行比较。
3. **困难。** 构建一个变量追踪任务（X1 → X2 → X3，共 3 跳），嵌入在 64k 的填充文本中。在 3 个前沿模型上测量准确率。报告每个模型的有效推理长度。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | 大海捞针 | 在填充文本中埋入一个事实，让模型检索它。 |
| RULER | NIAH 的加强版 | 涵盖检索/多跳/聚合/问答的 13 种任务类型。 |
| 有效上下文（Effective context） | 真正的容量 | 准确率仍保持在阈值以上的长度。 |
| 迷失在中间（Lost in the middle） | 深度偏差 | 模型对长输入中间位置的内容关注不足。 |
| 多针（Multi-needle） | 同时处理多个事实 | 多个埋入点；测试的是注意力的"杂耍"能力，而不仅仅是检索。 |
| MRCR | 多轮共指消解 | 8、24 或 100 针共指消解；暴露注意力饱和点。 |
| NoLiMa | 非词汇重叠针 | 针和查询之间没有字面 token 重叠；需要推理。 |

## 延伸阅读（Further Reading）

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 最初的 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准测试。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界长上下文评测。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 干草堆中的推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏差论文。
