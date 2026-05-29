# 子词分词（Subword Tokenization）—— BPE、WordPiece、Unigram、SentencePiece

> 基于词的分词器（tokenizer）遇到未见过的词会失败。基于字符的分词器会大幅增加序列长度。子词分词器在两者之间取得了平衡。每一个现代大语言模型（LLM）都建立在其中一种之上。

**类型：** 学习
**语言：** Python
**前置要求：** 第 5 阶段 · 01（文本处理），第 5 阶段 · 04（GloVe / FastText / 子词）
**时间：** 约 60 分钟

## 问题所在

你的词表（vocabulary）有 50,000 个词。用户输入了 "untokenizable"。你的分词器返回 `[UNK]`。模型现在对这个词没有任何信号。更糟糕的是：你的语料库（corpus）中第 90 百分位的文档有 40 个罕见词，这意味着每个文档丢失了 40 位信息。

子词分词解决了这个问题。常见词保持为单个 token。罕见词分解为有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据覆盖一切，因为任何字符串最终都是一串字节序列。

2026 年的每一个前沿大语言模型都建立在三种算法（BPE、Unigram、WordPiece）之一之上，并封装在三个库（tiktoken、SentencePiece、HF Tokenizers）之一之中。如果不选择其中一个，你无法交付一个语言模型。

## 核心概念

![BPE vs Unigram vs WordPiece，逐字符对比](../assets/subword-tokenization.svg)

**BPE（Byte-Pair Encoding，字节对编码）。** 从字符级词表开始。统计每一对相邻字符。将最频繁出现的字符对合并（merge）为一个新 token。重复此过程，直到达到目标词表大小。主流算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级BPE（Byte-level BPE）。** 相同的算法，但基于原始字节（256 个基础 token）而非 Unicode 字符。保证零 `[UNK]` token —— 任何字节序列都可以编码。GPT-2 使用 50,257 个 token（256 个字节 + 50,000 次合并 + 1 个特殊 token）。

**Unigram。** 从一个巨大的词表开始。为每个 token 分配一个 unigram 概率。迭代地剪枝那些移除后对语料库对数似然（log-likelihood）增加最少的 token。推理时具有概率性：可以采样多种分词结果（通过子词正则化（subword regularization）用于数据增强）。被 T5、mBART、ALBERT、XLNet、Gemma 使用。

**WordPiece。** 合并那些最大化训练语料库似然的字符对，而非基于原始频率。被 BERT、DistilBERT、ELECTRA 使用。

**SentencePiece 与 tiktoken。** SentencePiece 是一个*训练*词表的库（BPE 或 Unigram），直接在原始 Unicode 文本上训练，将空白字符（whitespace）编码为 `▁`。tiktoken 是 OpenAI 的快速*编码器*，针对预构建的词表进行编码；它不负责训练。

经验法则：

- **训练新词表：** SentencePiece（多语言（multilingual），无需预分词（pre-tokenization））或 HF Tokenizers。
- **针对 GPT 词表进行快速推理（inference）：** tiktoken（cl100k_base、o200k_base）。
- **两者兼顾：** HF Tokenizers —— 一个库，训练 + 服务。

## 动手构建

### 步骤 1：从零实现 BPE

参见 `code/main.py`。核心循环：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

该算法编码了三个关键事实。`</w>` 标记词尾，使得 "low"（后缀）和 "lower"（前缀）保持区分。频率加权使高频字符对在早期胜出。合并列表是有序的 —— 推理时按训练顺序应用合并。

### 步骤 2：使用学习到的合并规则进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现的时间复杂度为 O(n·|merges|)。生产级实现（tiktoken、HF Tokenizers）使用合并排名查找（merge-rank lookup）配合优先队列（priority queue），运行时间接近近线性时间（near-linear time）。

### 步骤 3：SentencePiece 实战

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 或 "unigram"
    character_coverage=0.9995, # CJK（中日韩文字）可设更低（如英语 0.9995，日语 0.995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格被编码为 `▁`，`character_coverage` 控制罕见字符被保留还是映射为 `<unk>` 的激进程度。

### 步骤 4：tiktoken 用于 OpenAI 兼容词表

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅支持编码。速度快（Rust 后端）。与 GPT-4/5 分词结果精确匹配，用于字节计数、成本估算、上下文窗口（context window）预算。

## 2026 年仍然存在的陷阱

- **分词器漂移（Tokenizer drift）。** 在词表 A 上训练，却部署到词表 B 上。Token ID 不同；模型输出垃圾结果。在 CI（持续集成）中检查 `tokenizer.json` 的哈希值。
- **空白字符歧义。** BPE 中 "hello" 和 " hello" 会产生不同的 token。始终显式指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语言训练不足。** 以英语为主的语料库产生的词表会将非拉丁文字拆分为 5-10 倍以上的 token。在 GPT-3.5 上，相同的提示词在日语/阿拉伯语中的成本是英语的 5-10 倍。o200k_base 部分修复了这个问题。
- **Emoji 拆分。** 一个 emoji 可能占用 5 个 token。在预算上下文时务必检查 emoji 处理。

## 使用指南

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 从零训练单语言（monolingual）模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供 OpenAI 兼容 API 服务 | tiktoken（GPT-4+ 使用 `o200k_base`） |
| 领域特定（domain-specific）词表（代码、数学、蛋白质） | 在领域语料库上训练自定义 BPE，与基础词表合并 |
| 边缘推理（edge inference），小模型 | Unigram（较小的词表效果更好） |

词表大小是一个扩展决策，而非常量。粗略启发式：<1B 参数用 32k，1-10B 用 50-100k，多语言/前沿模型用 200k+。

## 交付成果

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **简单。** 在 `code/main.py` 的小型语料库上训练一个 500 次合并的 BPE。对三个留出词进行编码。有多少个恰好产生 1 个 token，有多少个产生 >1 个 token？
2. **中等。** 在 100 个英语维基百科句子上比较 `cl100k_base`、`o200k_base` 和你训练的 vocab=32k 的 SentencePiece BPE 的 token 数量。报告每种方案的压缩比（compression ratio）。
3. **困难。** 用 BPE、Unigram 和 WordPiece 分别训练相同的语料库。在使用每种分词器的小型情感分类器（sentiment classifier）上测量下游准确率。选择的不同是否会使 F1 分数变化超过 1 个百分点？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BPE | 字节对编码 | 贪心地合并最频繁的字符对，直到达到目标词表大小。 |
| 字节级BPE | 永远不会有未知 token | 基于原始 256 字节的 BPE；GPT-2 / Llama 使用此方案。 |
| Unigram | 概率分词器 | 使用对数似然从大型候选集中剪枝；被 T5、Gemma 使用。 |
| SentencePiece | 那个处理空格的 | 在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | 那个快的 | OpenAI 的 Rust 后端 BPE 编码器，用于预构建词表。不负责训练。 |
| 合并列表 | 那些神奇的数字 | `(a, b) → ab` 合并的有序列表；推理时按顺序应用。 |
| 字符覆盖率 | 多罕见才算太罕见？ | 分词器必须覆盖的训练语料库中字符的比例；通常约为 0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) —— Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) —— SentencePiece 库论文。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) —— 简明参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) —— 使用指南与编码列表。
