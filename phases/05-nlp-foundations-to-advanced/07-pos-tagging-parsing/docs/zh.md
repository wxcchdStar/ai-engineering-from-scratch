# 词性标注与句法解析（POS Tagging and Syntactic Parsing）

> 语法（Grammar）曾一度不再流行。后来，每个大语言模型（LLM）管线都需要验证结构化抽取结果，于是它又回来了。

**类型（Type）：** Build
**语言（Languages）：** Python
**前置条件（Prerequisites）：** 阶段 5 · 01（文本处理（Text Processing）），阶段 2 · 14（朴素贝叶斯（Naive Bayes））
**时间（Time）：** ~45 分钟

## 问题（The Problem）

第 01 课承诺过，词形还原（lemmatization）需要词性（part-of-speech）信息。如果不知道 `running` 是动词，词形还原器就无法将其还原为 `run`。如果不知道 `better` 是形容词，它也无法还原为 `good`。

那个承诺背后隐藏了一整个子领域。词性标注（part-of-speech tagging）负责分配语法类别。句法解析（syntactic parsing）负责恢复句子的树状结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典自然语言处理（NLP）花了二十年时间不断精进这两项技术。之后，深度学习（deep learning）将它们压缩为预训练变换器（pretrained transformer）之上的一个序列标注（token-classification）任务，学术界也就转向了。

但应用界没有。今天，每一个结构化抽取（structured-extraction）管线在底层仍然使用词性标注和依存树（dependency trees）。由大语言模型生成的 JSON 需要根据语法约束进行验证。问答系统利用依存解析来分解查询。机器翻译质量评估器通过对比解析树来检查对齐。

值得了解。本课将介绍标注集（tagsets）、基线方法，以及你应该停止从头实现并直接调用 spaCy 的那个转折点。

## 概念（The Concept）

**词性标注**为每个词元（token）分配一个语法类别。英文默认使用**宾州树库（Penn Treebank，PTB）**标注集。它包含 36 个标签，其中不少区分在外行看来过于精细：`NN` 单数名词，`NNS` 复数名词，`NNP` 单数专有名词，`VBD` 动词过去式，`VBZ` 动词第三人称单数现在式，等等。**通用依存关系（Universal Dependencies，UD）**标注集则更粗粒度（17 个标签），且与语言无关；它已成为跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法解析**产出一棵树。主要有两种风格：

- **成分解析（Constituency parsing）。** 名词短语、动词短语、介词短语层层嵌套。输出是一棵由非终结符类别（NP、VP、PP）构成的树，词语为其叶子节点。
- **依存解析（Dependency parsing）。** 每个词有一个它所依赖的中心词（head word），并带有一个语法关系标签。输出是一棵树，每条边都是一个（中心词、依存词、关系）三元组。

依存解析在 2010 年代胜出，因为它能跨语言干净地泛化，尤其适用于自由语序的语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 动手构建（Build It）

### 步骤 1：最高频标签基线（most-frequent-tag baseline）

最简单但可用的词性标注器。对每个词，预测它在训练集中出现频率最高的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这个基线能达到约 85% 的准确率。不算好，但任何严肃模型都不应低于这个下限。

### 步骤 2：二元语法隐马尔可夫模型标注器（bigram HMM tagger）

对序列的联合概率进行建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：转移概率（transition probabilities，给定前一个标签时当前标签的概率），发射概率（emission probabilities，给定标签时词语的概率）。两者都基于计数并使用拉普拉斯平滑（Laplace smoothing）来估计。使用维特比算法（Viterbi，在标签网格上进行动态规划）进行解码。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM 在 Brown 语料库上能达到约 93% 的准确率。从 85% 跃升至 93% 主要归功于转移概率——模型学会了 `DET NOUN` 很常见而 `NOUN DET` 很罕见。

### 步骤 3：为什么现代标注器能超越上述方法

转移概率与发射概率都是局部的。它们无法捕捉到 `saw` 在 "I bought a saw" 中是名词而在 "I saw the movie" 中是动词。一个使用任意特征（后缀、词形、前后词、词本身）的条件随机场（CRF，Conditional Random Field）可以达到约 97% 的准确率。BiLSTM-CRF 或变换器（transformer）则能达到约 98% 以上。

该任务的上限由标注者分歧（annotator disagreement）决定。在宾州树库上，标注者之间的一致性约为 97%。超过 98% 的模型很可能在过拟合测试集。

### 步骤 4：依存解析概览

完整的从头实现依存解析超出了本课范围；规范的教材级论述见 Jurafsky 和 Martin 的著作。需要了解两个经典流派：

- **基于转移（Transition-based）的解析器**（arc-eager、arc-standard）类似于移进-归约解析器（shift-reduce parser）：它们读取词元，将其移入栈中，并执行创建弧的归约动作。贪婪解码速度很快。经典实现是 MaltParser。现代神经版本：Chen 和 Manning 的基于转移的解析器。
- **基于图（Graph-based）的解析器**（Eisner 算法、Dozat-Manning 双仿射）对每一条可能的中心词-依存词边进行打分，并选出最大生成树。更慢但更准确。

对于大多数应用工作，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读 `dep` 列，句子的语法结构就自然呈现。

## 实际使用（Use It）

每一个生产级 NLP 库都将词性标注和依存解析器作为标准管线的一部分提供。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确，与分词（tokenization）+ 命名实体识别（NER）+ 词形还原集成在一起。`token.tag_`（Penn 标注集），`token.pos_`（UD 标注集），`token.dep_`（依存关系）。
- **Stanford NLP (stanza)**。Stanford 对 CoreNLP 的继任者。在 60 多种语言上达到当前最优水平。
- **trankit**。基于变换器，在 UD 上准确率较高。
- **NLTK**。`pos_tag`。可用但较慢、较旧。适合教学使用。

### 2026 年这仍然重要的原因

- **词形还原。** 第 01 课需要词性才能正确进行词形还原。永远如此。
- **从 LLM 输出中做结构化抽取。** 验证生成的句子是否符合语法约束（例如，主谓一致、必需的修饰语）。
- **基于方面的情感分析（Aspect-based sentiment）。** 依存解析可以告诉你哪个形容词修饰哪个名词。
- **查询理解（Query understanding）。** "movies directed by Wes Anderson starring Bill Murray" 通过解析可以分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系与语言无关，从而支持对新语言进行零样本结构化分析。
- **低算力管线。** 如果你没法部署一个变换器，词性标注 + 依存解析 + 地名词典（gazetteer）能带给你出人意料的成果。

## 产出（Ship It）

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习（Exercises）

1. **简单。** 在一个小的标注语料库（如 NLTK 的 Brown 子集）上使用最高频标签基线，评测其在留出句上的准确率。验证约 85% 的结果。
2. **中等。** 训练上述 Bigram HMM，并报告每个标签的精确率和召回率。HMM 最容易混淆哪些标签？
3. **困难。** 使用 spaCy 的依存解析从 1000 句样本中抽取主-谓-宾三元组。在 50 个手工标注的三元组上进行评估。记录抽取失败的场景（通常涉及被动语态、并列结构和省略主语）。

## 关键术语（Key Terms）

| 术语（Term） | 人们常说的话（What people say） | 实际含义（What it actually means） |
|------|-----------------|-----------------------|
| POS tag | 词的类别 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准标注集 | 英文专用。对动词时态和名词单复数做了细粒度区分。 |
| Universal Dependencies | 多语言标注集 | 比 PTB 更粗粒度；与语言无关；跨语言工作的默认选择。 |
| Dependency parse | 句法树 | 每个词有一个中心词，每条边带有一个语法关系标签。 |
| Viterbi | 动态规划 | 在给定发射概率和转移概率的前提下，找出概率最高的标签序列。 |

## 延伸阅读（Further Reading）

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 关于词性标注和句法解析的规范教材级论述。
- [Universal Dependencies project](https://universaldependencies.org/) — 每个多语言解析器都在使用的跨语言标注集与树库合集。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 上每个属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经解析器带入主流的论文。