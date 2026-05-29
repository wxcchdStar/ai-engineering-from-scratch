# 对话状态追踪（Dialogue State Tracking）

> "I want a cheap restaurant in the north... actually make it moderate... and add Italian." 三轮对话，三次状态更新。对话状态追踪（DST）保持槽位-值字典同步，确保预订流程正确执行。

**类型（Type）：** Build
**语言（Languages）：** Python
**前置条件（Prerequisites）：** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**时间（Time）：** ~75 minutes

## 问题（The Problem）

在任务导向对话系统（task-oriented dialogue system）中，用户的目标被编码为一组槽位-值对（slot-value pairs）：`{cuisine: italian, area: north, price: moderate}`。每一轮用户发言都可能添加、修改或删除某个槽位。系统必须读取整个对话并正确输出当前状态。

只要搞错一个槽位，系统就会预订错误的餐厅、安排错误的航班或扣错银行卡。DST 是用户说了什么与后端执行什么之间的枢纽。

尽管有了大语言模型（LLM），DST 在 2026 年仍然重要，原因如下：

- 合规敏感领域（银行、医疗、机票预订）需要确定性的槽位值，而非自由形式的生成。
- 工具调用智能体（tool-use agents）在调用 API 之前仍然需要槽位解析（slot resolution）。
- 多轮纠正比看起来更难："actually no, make it Thursday."

现代流水线：经典 DST 概念 + LLM 提取器 + 结构化输出（structured-output）护栏。

## 核心概念（The Concept）

![DST: dialog history → slot-value state](../assets/dst.svg)

**任务结构。** 一个模式（schema）定义了领域（domains，如 restaurant、hotel、taxi）及其槽位（slots，如 cuisine、area、price、people）。每个槽位可以为空、填充来自封闭集合的值（price: {cheap, moderate, expensive}），或填充自由形式的值（name: "The Copper Kettle"）。

**两种 DST 表述方式。**

- **分类（Classification）。** 对每个 (slot, candidate_value) 对，预测 yes/no。适用于封闭词汇槽位。2020 年之前的标准做法。
- **生成（Generation）。** 给定对话，以自由文本形式生成槽位值。适用于开放词汇槽位。现代默认做法。

**评估指标。** 联合目标准确率（Joint Goal Accuracy，JGA）——每一轮对话中*所有*槽位都正确的比例。全有或全无。MultiWOZ 2.4 排行榜在 2026 年最高约 83%。

**架构。**

1. **基于规则（Rule-based，槽位正则 + 关键词）。** 窄领域下的强基线。可调试。
2. **TripPy / BERT-DST。** 基于复制的生成，使用 BERT 编码。LLM 出现之前的标准方案。
3. **LDST（LLaMA + LoRA）。** 指令微调的 LLM，配合领域-槽位提示（domain-slot prompting）。在 MultiWOZ 2.4 上达到 ChatGPT 级别的质量。
4. **无本体（Ontology-free，2024–26）。** 跳过模式；直接生成槽位名称和值。处理开放领域。
5. **提示 + 结构化输出（Prompt + structured output，2024–26）。** LLM 配合 Pydantic 模式 + 约束解码（constrained decoding）。5 行代码，生产就绪。

### 经典失败模式

- **跨轮共指消解（Co-reference across turns）。** "Let's stay with the first option." 需要解析出是哪个选项。
- **覆盖 vs 追加。** 用户说 "add Italian." 你是替换 cuisine 还是追加？
- **隐式确认（Implicit confirmations）。** "OK cool" —— 这是接受了提议的预订吗？
- **纠正（Correction）。** "Actually make it 7 pm." 必须更新时间而不清除其他槽位。
- **对前一条系统话语的共指。** "Yes, that one." 哪个 "that"？

## 动手构建（Build It）

### 步骤 1：基于规则的槽位提取器

参见 `code/main.py`。正则表达式 + 同义词词典在窄领域中可以覆盖 70% 的规范话语：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在规范词汇之外很脆弱。适用于确定性槽位确认。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 永远不要重置用户未触及的槽位。
- 显式否定（"never mind the cuisine"）必须清除。
- 用户纠正（"actually..."）必须覆盖，而非追加。

### 步骤 3：LLM 驱动的 DST 与结构化输出

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证输出一个有效的状态对象。无需正则、无模式不匹配、无幻觉槽位。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在多大比例的对话轮次中能正确获取所有槽位？对于 MultiWOZ 2.4，2026 年顶级系统：80-83%。你的领域内系统应在窄词汇上超过这个值，否则 LLM 基线会击败你。

### 步骤 5：处理纠正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到纠正时，覆盖最后更新的槽位而非追加。没有 LLM 帮助很难做对。现代模式：始终让 LLM 从历史中重新生成整个状态，而非增量更新——这自然能处理纠正。

## 常见陷阱（Pitfalls）

- **全历史重新生成成本。** 让 LLM 每轮重新生成状态的总 token 成本为 O(n²)。限制历史长度或对较早轮次进行摘要。
- **模式漂移（Schema drift）。** 事后添加新槽位会破坏旧的训练数据。为你的模式做版本管理。
- **大小写敏感性。** "Italian" vs "italian" vs "ITALIAN" —— 处处做规范化。
- **隐式继承。** 如果用户之前指定了 "for 4 people"，那么一个新的不同时间的请求不应清除人数。始终传递完整历史。
- **自由形式 vs 封闭集合。** 名称、时间和地址需要自由形式槽位；菜系和区域是封闭集合。在模式中混合使用两者。

## 使用指南（Use It）

2026 年技术栈：

| 场景（Situation） | 方案（Approach） |
|-----------|----------|
| 窄领域（一到两个意图） | 基于规则 + 正则 |
| 宽领域，有标注数据 | LDST（LLaMA + LoRA，基于 MultiWOZ 风格数据） |
| 宽领域，无标注，生产就绪 | LLM + Instructor + Pydantic 模式 |
| 语音/口语 | ASR + 规范化器 + LLM-DST |
| 多领域预订流程 | 模式引导的 LLM，配合每个领域的 Pydantic 模型 |
| 合规敏感 | 基于规则为主，LLM 为后备，配合确认流程 |

## 交付物（Ship It）

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习（Exercises）

1. **简单。** 在 `code/main.py` 中为 3 个槽位（cuisine、area、price）构建基于规则的状态追踪器。在 10 个手工编写的对话上测试。测量 JGA。
2. **中等。** 使用 Instructor + Pydantic + 一个小型 LLM 在相同数据集上测试。比较 JGA。检查最难的对话轮次。
3. **困难。** 同时实现两者并做路由：基于规则为主，当基于规则输出少于 2 个有置信度的槽位时回退到 LLM。测量组合 JGA 和每轮推理成本。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| DST | 对话状态追踪 | 在对话轮次之间维护槽位-值字典。 |
| 槽位（Slot） | 用户意图单元 | 后端所需的命名参数（cuisine、date）。 |
| 领域（Domain） | 任务区域 | Restaurant、hotel、taxi —— 槽位的集合。 |
| JGA | 联合目标准确率 | 所有槽位都正确的对话轮次比例。全有或全无。 |
| MultiWOZ | 基准数据集 | 多领域 WOZ 数据集；标准 DST 评估基准。 |
| 无本体 DST（Ontology-free DST） | 无模式 | 直接生成槽位名称和值，无固定列表。 |
| 纠正（Correction） | "Actually..." | 覆盖之前已填充槽位的对话轮次。 |

## 延伸阅读（Further Reading）

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 经典基准数据集。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA 指令微调用于 DST。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基于复制的 DST 主力方案。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督任务导向对话。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 经典 DST 结果。
