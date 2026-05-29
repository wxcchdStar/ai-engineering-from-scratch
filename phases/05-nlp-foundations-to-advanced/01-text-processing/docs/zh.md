# 文本处理（Text Processing）—— 分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是桥梁。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 2 · 14（朴素贝叶斯）
**时间：** 约 45 分钟

## 问题

模型无法读取 "The cats were running."。它读取整数。

每个 NLP 系统都以同样的三个问题开始。一个词从哪里开始。词的词根是什么。我们如何将 "run"、"running"、"ran" 在有帮助时视为同一个东西，在没有帮助时视为不同的东西。

分词（Tokenization）搞错了，模型就从垃圾中学习。如果你的分词器将 `don't` 视为一个 token，但将 `do n't` 视为两个 token，训练分布就会分裂。如果你的词干提取器（Stemmer）将 `organization` 和 `organ` 折叠到同一个词干，主题建模就完蛋了。如果你的词形还原器（Lemmatizer）需要词性上下文但你没有传入，动词就会被当作名词处理。

本课从零构建三个预处理原语，然后展示 NLTK 和 spaCy 如何做同样的工作，以便你看到其中的权衡。

## 概念

三个操作。每个都有其职责和失败模式。

**分词（Tokenization）** 将字符串拆分为 token。"Token" 故意模糊，因为正确的粒度取决于任务。经典 NLP 用词级。Transformer 用子词级。没有空格的语言用字符级。

**词干提取（Stemming）** 用规则砍掉后缀。快、激进、笨。`running -> run`。`organization -> organ`。第二个就是失败模式。

**词形还原（Lemmatization）** 使用语法知识将词还原为字典形式。慢、准确、需要查找表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则。当速度重要且你能容忍噪声时用词干提取（搜索索引、粗略分类）。当含义重要时用词形还原（问答、语义搜索、任何用户会看到的内容）。

## 构建它

### 步骤 1：正则表达式词分词器

最简单的有用分词器按非字母数字字符分割，同时将标点符号保留为独立 token。不完美，不最终，但一行就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排列。带可选内部撇号的单词（`don't`、`it's`）。纯数字。任何单个非空白非字母数字字符作为独立 token（标点符号）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式。`3pm` 拆分为 `['3', 'pm']`，因为我们在字母序列和数字序列之间交替。对大多数任务来说足够好了。URL、电子邮件、话题标签都会出问题。在生产环境中，在通用模式之前添加专用模式。

### 步骤 2：Porter 词干提取器（仅步骤 1a）

完整的 Porter 算法有五个阶段的规则。仅步骤 1a 就涵盖了最常见的英语后缀，并展示了模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上到下阅读规则。`ies -> i` 规则就是为什么 `ponies -> poni` 而不是 `pony`。真正的 Porter 有步骤 1b 可以修复它。规则相互竞争。先匹配的规则胜出。顺序比任何单条规则都重要。

### 步骤 3：基于查找表的词形还原器

正确的词形还原需要形态学。一个可操作的教学版本使用小型词形还原表和回退策略。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个案例是关键的教学时刻。`watched` 不在我们的表中，而我们的回退策略只处理 `ing`。真正的词形还原覆盖 `ed`、不规则动词、比较级形容词、带音变的复数（`children -> child`）。这就是为什么生产系统使用 WordNet、spaCy 的形态分析器或完整的形态分析器。

### 步骤 4：将它们串联起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是词性标注器（POS Tagger）。Phase 5 · 07（词性标注）会构建一个。目前，将所有内容默认为 `NOUN` 并承认这个限制。

## 使用它

NLTK 和 spaCy 提供了生产版本。各几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩写、Unicode、你的正则表达式遗漏的边缘情况。`PorterStemmer` 运行全部五个阶段。`WordNetLemmatizer` 需要将词性标签从 NLTK 的 Penn Treebank 方案转换为 WordNet 的缩写集。上面的转换接线是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 将整个流水线隐藏在 `nlp(text)` 后面。分词、词性标注和词形还原全部运行。大规模下比 NLTK 更快。开箱即用更准确。代价是你不能轻易替换单个组件。

### 何时选择哪个

| 场景 | 选择 |
|------|------|
| 教学、研究、替换组件 | NLTK |
| 生产环境、多语言、速度重要 | spaCy |
| Transformer 流水线（反正你会用模型的 tokenizer 分词） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人警告你的两个失败模式

大多数教程教完算法就停了。有两件事会咬到真实的预处理流水线，而且几乎从未被提及。

**可复现性漂移（Reproducibility Drift）。** NLTK 和 spaCy 在不同版本之间会改变分词和词形还原的行为。在 spaCy 2.x 中产生 `['do', "n't"]` 的，在 3.x 中可能产生 `["don't"]`。你的模型是在一个分布上训练的。推理现在运行在另一个分布上。准确率悄悄下降，没人知道为什么。在 `requirements.txt` 中锁定库版本。编写一个预处理回归测试，冻结 20 个样本句子的预期分词结果。每次升级时运行它。

**训练/推理不匹配（Training / Inference Mismatch）。** 训练时使用激进预处理（小写、停用词移除、词干提取），部署时处理原始用户输入，然后看着性能崩溃。这是最常见的生产 NLP 失败。如果你在训练期间预处理，你必须在推理期间运行完全相同的函数。将预处理作为模型包内的一个函数发布，而不是作为服务团队重写的 notebook 单元格。

## 交付它

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: 为 NLP 任务推荐分词、词干提取和词形还原方案。
phase: 5
lesson: 01
---

你为经典 NLP 预处理提供建议。给定任务描述，你输出：

1. 分词选择（regex、NLTK word_tokenize、spaCy 或 transformer tokenizer）。解释原因。
2. 是否进行词干提取、词形还原、两者都做或都不做。解释原因。
3. 具体的库调用。命名函数。如果涉及 NLTK，引用词性标签转换。
4. 用户应该测试的一个失败模式。

拒绝为用户可见的文本推荐词干提取。拒绝在没有词性标签的情况下推荐词形还原。标记非英语输入需要不同的流水线。
```

## 练习

1. **简单。** 扩展 `tokenize` 以将 URL 保留为单个 token。测试：`tokenize("Visit https://example.com today.")` 应产生一个 URL token。
2. **中等。** 实现 Porter 步骤 1b。如果一个词包含元音且以 `ed` 或 `ing` 结尾，则移除它。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个词形还原器，使用 WordNet 作为查找表，但当 WordNet 没有条目时回退到你的 Porter 词干提取器。在标注语料库上测量准确率，与纯 WordNet 和纯 Porter 对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Token | 一个词 | 模型消费的任何单元。可以是词、子词、字符或字节。 |
| Stem（词干） | 词的词根 | 基于规则的后缀剥离结果。不总是一个真实的词。 |
| Lemma（词形还原形式） | 字典形式 | 你会查字典的形式。需要语法上下文才能正确计算。 |
| POS tag（词性标签） | 词性 | 如 NOUN、VERB、ADJ 的类别。准确词形还原所必需。 |
| Morphology（形态学） | 词形变化规则 | 词如何根据时态、数、格改变形式。词形还原依赖它。 |

## 进一步阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，仍然是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实流水线如何接线。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边缘情况。
