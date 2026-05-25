# ColPali 与视觉原生文档 RAG

> 传统 RAG 将 PDF 解析为文本、切分为块、嵌入块、存储向量。每一步都损失信号：OCR 丢弃图表数据，分块打断表格行，文本嵌入忽略图形。ColPali（Faysse et al., 2024年7月）提出了一个更简单的问题：为什么要提取文本？直接通过 PaliGemma 嵌入页面图像，使用 ColBERT 风格的后期交互进行检索，保留文档承载的所有版面、图形、字体和格式信号。公开基准：在视觉丰富的文档上端到端准确率比文本 RAG 高20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了这一模式。本课解读视觉原生 RAG 论点并构建一个简易 ColPali 式索引器。

**类型：** 构建（Build）
**语言：** Python（标准库，多向量索引器 + MaxSim 评分器）
**前置课程：** Phase 11（LLM 工程 — RAG 基础）, Phase 12 · 05（LLaVA）
**时长：** ~180 分钟

## 学习目标

- 解释双编码器检索（每文档一个向量）与后期交互检索（每文档多个向量）的区别。
- 描述 ColBERT 的 MaxSim 操作以及 ColPali 如何将其从文本 token 推广到图像 patch。
- 构建简易 ColPali 式索引器：页面 → patch 嵌入 → 对查询词嵌入做 MaxSim → top-k 页面。
- 比较 ColPali + Qwen2.5-VL 生成器 vs 文本 RAG + GPT-4 在发票/财报用例上的表现。

## 问题

对 PDF 做文本 RAG 会丢掉文档的大部分信息。财报的 Q3 营收增长通常在图表中；医疗报告的发现在标注图像中；法律合同的签名块是版面事实而非文本事实。

文本 RAG 管线：

1. PDF → 通过 OCR / pdftotext 提取文本。
2. 文本 → 300-500 token 块。
3. 块 → 双编码器嵌入（一个向量）。
4. 用户查询 → 嵌入 → 余弦相似度 → top-k 块。
5. 块 + 查询 → LLM。

五个有损步骤。图表未被捕获。表格跨块断裂。多栏版面被扁平化。图形标注消失。

ColPali 的修复：跳过 OCR，直接嵌入页面图像。使用 ColBERT 风格的后期交互进行检索，使模型在查询时可以关注细粒度 patch。

## 核心概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia, arXiv:2004.12832）是一种文本检索方法。它不是每文档一个向量，而是每 token 一个向量。在查询时：

- 查询 token 获得各自的嵌入（N_q 个向量）。
- 文档 token 获得嵌入（N_d 个向量，通常预先缓存）。
- 分数 = 对每个查询 token 取其与所有文档 token 的最大余弦相似度之和：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个查询 token "挑选"其最佳匹配的文档 token。最终分数为求和。

优点：召回率强，处理词级语义。缺点：每文档 N_d 个向量，存储昂贵。

### ColPali

ColPali（Faysse et al., arXiv:2407.01449）将 ColBERT 模式应用于图像。

- 每页通过 PaliGemma（ViT + 语言）编码为 patch 嵌入：每页 N_p 个向量。
- 每个用户查询（文本）编码为查询 token 嵌入：N_q 个向量。
- 分数 = Σ_i max_j cos(q_i, p_j)，即对查询文本 token 和页面图像 patch 做 MaxSim。
- 按总分检索 top-k 页面。

文档摄入时：用 PaliGemma 嵌入每一页，存储所有 patch 嵌入。查询时：嵌入查询 token，对所有已存储的页面嵌入计算 MaxSim，返回 top-k 页面。

优点：在视觉丰富文档上端到端比文本 RAG 高20-40%。每个 patch 向量捕获局部版面和内容。

缺点：每页 N_p patch × 4字节浮点 × D维向量 = 存储增长快。通过 PQ / OPQ 量化缓解。

### ColQwen2 与 ColSmol

ColQwen2（illuin-tech, 2024-2025）将 PaliGemma 替换为 Qwen2-VL。更好的基础编码器，更好的检索效果。

ColSmol 是用于本地/边缘部署的小规模变体。约1B参数的 ColSmol 检索器可在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu et al., arXiv:2410.10594）是另一种变体：不对 patch 做 MaxSim，而是用 VLM 将每页池化为单个向量，然后双编码器检索。索引更快 + 存储更小，召回率较弱。

质量-成本权衡：ColPali 追求质量，VisRAG 追求规模。

### M3DocRAG

M3DocRAG（Cho et al., arXiv:2411.04952）将多模态检索扩展到多页多文档推理。跨文档检索页面，为 VLM 组合多页上下文。

### ViDoRe — 基准测试

ColPali 的配套基准。视觉文档检索评估 (Visual Document Retrieval Evaluation)。任务包括财报、科学论文、行政文件、医疗记录、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上约80% nDCG@5；同样文档上文本 RAG 约50-60%。

### 端到端 RAG 管线

视觉原生 RAG：

1. 摄入：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch 嵌入。
2. 查询：用户文本 → 查询 token 嵌入 → 对所有索引页面做 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

全程无 OCR。图形、图表、字体、版面全部流入答案。

### 存储计算

一份50页的财报，每页729个 patch，128维嵌入：

- ColPali：50 * 729 * 128 * 4 字节 = 约18 MB 原始，PQ 后约4 MB。
- 文本 RAG：50 块 * 768维 * 4 字节 = 约150 kB。

ColPali 每文档约30倍存储。大规模下 OPQ / PQ 将其降至约5-10倍，通常可接受。

### 文本 RAG 仍然胜出的场景

- 无版面信号的纯文本文档（维基文章、聊天记录）。文本 RAG 更简单且存储更便宜。
- 百万页级档案库，存储主导成本。
- 严格监管要求在检索之外提供可提取的 OCR 文本。

2026年其他一切场景——财报、科学论文、法律合同、医疗记录、UX 文档——视觉原生 RAG 胜出。

## 动手实践

`code/main.py`：

- 简易 patch 编码器：将一个"页面"（小型特征向量网格）映射为 patch 嵌入数组。
- MaxSim 评分器：计算查询 token 嵌入集与页面 patch 集之间的 ColBERT 风格分数。
- 索引5个简易页面，运行3个查询，返回带分数的 top-k。

## 交付产出

本课产出 `outputs/skill-vision-rag-designer.md`。给定一个文档 RAG 项目，选择 ColPali / ColQwen2 / VisRAG / 文本 RAG 并确定存储规模。

## 练习

1. 一份200页年报，每页729个 patch，128维嵌入，4字节浮点。计算原始存储和 PQ 压缩（8x）后的存储。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个求和捕获了什么是简单平均相似度所不能的？

3. ColPali 将页面索引为 patch 集。如果我们改为在词级别索引（如 ColBERT 所做的）会有什么变化？权衡是什么？

4. 为100万页语料库设计端到端管线，每查询延迟预算500ms。选择 ColQwen2 / VisRAG 并论证。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述多页注意力模式以及它与单页 ColPali 检索的区别。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 后期交互 (Late interaction) | "ColBERT 风格" | 使用逐 token 或逐 patch 嵌入 + MaxSim 的检索，而非单个文档向量 |
| MaxSim | "逐 patch 取最大" | 对每个查询 token，取其与文档 token 最高相似度；跨查询求和 |
| 双编码器 (Bi-encoder) | "单向量" | 每文档一个向量；更快但丢失粒度 |
| 多向量 (Multi-vector) | "每文档多向量" | 每文档/页存储 N_p 个向量；存储成本增加但召回率提升 |
| Patch 嵌入 (Patch embedding) | "页面特征" | VLM 编码器产生的每图像 patch 一个向量，每页预先缓存 |
| ViDoRe | "视觉文档基准" | ColPali 的视觉文档检索基准测试套件 |
| PQ 量化 (PQ quantization) | "乘积量化" | 保持向量相似度同时缩减存储约8倍的压缩方法 |

## 延伸阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
