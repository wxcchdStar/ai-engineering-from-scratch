# 实战项目 04 — 多模态文档问答（视觉优先的 PDF、表格、图表）

> 2026 年的文档问答前沿已从 OCR 后文本处理转向视觉优先的后期交互（Late Interaction）。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，使用多向量后期交互进行嵌入，并让查询直接关注图像块（patch）。在财务 10-K 报告、科学论文和手写笔记上，这种模式大幅优于 OCR 优先方案。在 10,000 页上端到端构建该管道，并发布与 OCR 后文本方案的并排对比报告。

**类型：** 实战项目（Capstone）
**语言：** Python（管道）、TypeScript（查看器 UI）
**前置课程：** 第 4 阶段（计算机视觉）、第 5 阶段（NLP）、第 7 阶段（Transformers）、第 11 阶段（LLM 工程）、第 12 阶段（多模态）、第 17 阶段（基础设施）
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

企业积压了大量被 OCR 管道损坏的 PDF：包含旋转表格的扫描版 10-K 报告、公式密集的科学论文、只有作为图像才有意义的图表、手写批注。将它们视为文本优先意味着丢失一半的信号。2026 年的答案是：对原始页面图像进行后期交互多向量检索。ColPali（Illuin Tech）首创了这一方法；ColQwen2.5-v0.2 和 ColQwen3-omni 进一步提升了准确性。在 ViDoRe v3 上，视觉优先检索的得分显著高于 OCR 后文本方案——而且在图表、表格和手写内容上差距更大。

其代价是存储和延迟。一个 ColQwen 嵌入约为每页 2048 个图像块向量，而非单个 1024 维向量。原始存储量急剧膨胀。DocPruner（2026 年）实现了 50% 的剪枝而无明显精度损失。你需要为 10,000 页建立索引，测量 ViDoRe v3 nDCG@5，在 2 秒内提供答案，并直接与 OCR 后文本基线进行对比。

## 概念

后期交互意味着每个查询 token 与每个图像块 token 进行评分，每个查询 token 的最大分数被求和。你可以在不需要单一池化向量的情况下获得细粒度匹配。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储每个图像块的嵌入，并在检索时运行 MaxSim。

回答器是一个视觉语言模型（VLM），它接收查询加上前 k 个检索到的页面作为图像，并写出带有证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于公式和科学符号，OCR 回退方案（Nougat、dots.ocr）作为可选的文本通道被拼接进来。

评估是一个二维矩阵。一个轴：内容类型（纯文本段落、密集表格、柱状/折线图、手写笔记、公式）。另一个轴：检索方法（视觉优先后期交互 vs OCR 后文本 vs 混合）。每个单元格包含 nDCG@5 和答案准确率。报告即为交付物。

## 架构

```
PDF -> 页面渲染器（PyMuPDF，180 DPI）
           |
           v
  ColQwen2.5-v0.2 嵌入（每页多向量，约 2048 个图像块）
           |
           +------> DocPruner 50% 压缩
           |
           v
   多向量索引（Vespa 或 Qdrant 多向量）
           |
查询 ----+----> 检索前 k 页（MaxSim）
           |
           v
  VLM 回答器：Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    输入：查询 + 前 k 页图像 + 可选 OCR 文本
           |
           v
  带引用页码 + 证据区域的答案
           |
           v
  Streamlit / Next.js 查看器：源页面上高亮显示的边界框
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，纵向归一化
- 后期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队）
- 索引：Vespa 配合多向量字段，或 Qdrant 多向量，或 AstraDB 配合 MaxSim
- 剪枝：DocPruner 2026 策略（保留高方差图像块，50% 压缩率，精度损失 < 0.5%）
- OCR 回退（公式/密集表格）：dots.ocr 或 Nougat
- VLM 回答器：Qwen3-VL-30B 自托管或 Gemini 2.5 Pro 托管；InternVL3 作为备选
- 评估：ViDoRe v3 基准，M3DocVQA 用于多页推理
- 查看器 UI：Next.js 15 配合 canvas 叠加层用于证据区域

## 构建步骤

1. **数据摄入。** 遍历包含 10,000 个 PDF 页面的语料库，涵盖 10-K 报告、科学论文和扫描文档。将每页渲染为 1536x2048 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每个页面图像运行 ColQwen2.5-v0.2。输出形状约为 2048 个图像块嵌入，维度为 128。应用 DocPruner 保留信号最强的一半。写入 Vespa 多向量字段或 Qdrant 多向量。

3. **查询。** 对于每个传入查询，使用查询塔（token 级嵌入）进行嵌入。对索引运行 MaxSim：对于每个查询 token，取与页面图像块嵌入的最大点积，求和。返回前 k 页。

4. **合成。** 使用查询和前 5 页图像调用 Qwen3-VL-30B。提示词："仅使用提供的页面回答。每个声明需引用（doc_id, page）并指明区域（图、表、段落）。"

5. **证据区域。** 后处理答案以提取引用区域。如果 VLM 输出边界框（Qwen3-VL 可以），则在查看器中将其渲染为叠加层。

6. **OCR 回退。** 对于被识别为公式密集的页面（基于图像方差的启发式方法），运行 Nougat 或 dots.ocr，并将 OCR 文本作为图像之外的额外通道传递。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页 QA 准确率）。同时在同一语料库上使用相同的合成器运行 OCR 后文本管道。生成内容类型 × 方法的矩阵。

8. **UI。** 首先使用 Streamlit 原型；然后使用 Next.js 15 生产查看器，配合逐页证据区域叠加。

## 使用方式

```
$ doc-qa ask "2024 年 EMEA 地区的运营利润率变化是多少？"
[检索]   前 5 页，耗时 320 毫秒（ColQwen2.5，MaxSim，Vespa）
[合成]   qwen3-vl-30b，耗时 1.4 秒，引用（form-10k-2024, p. 88）+（..., p. 92）
答案：
  EMEA 运营利润率从 18.2% 降至 16.8%，下降 140 个基点。
  引用：10-K-2024.pdf p.88（表 4，分部运营利润率）
        10-K-2024.pdf p.92（MD&A，运营绩效）
[查看器] 打开，p.88 表 4 上叠加高亮边界框
```

## 交付标准

`outputs/skill-doc-qa.md` 描述交付物：一个针对特定语料库调优的视觉优先多模态文档问答系统，并在 ViDoRe v3 上与 OCR 后文本基线进行评估对比。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 与 OCR 文本基线和已发布排行榜对比的基准数据 |
| 20 | 证据区域定位 | 实际包含答案片段的引用区域比例 |
| 20 | 存储与延迟工程 | DocPruner 压缩率、索引 p95、答案 p95 |
| 20 | 多页推理 | 在手工标注的 100 个多页问题集上的准确率 |
| 15 | 源文件检查 UX | 查看器清晰度、叠加保真度、并排对比工具 |
| **100** | | |

## 练习

1. 在同一语料库上测量 ColQwen2.5-v0.2 与 ColQwen3-omni 的对比。哪些页面一个模型正确而另一个错误？向索引添加"内容类别"标签以按类型路由。
2. 激进剪枝嵌入（75%、90%）。找到压缩悬崖：ViDoRe nDCG@5 降至 OCR 基线以下的临界点。
3. 构建混合方案：并行运行 OCR 后文本和 ColQwen，使用 RRF 融合，使用交叉编码器重排序。混合方案是否优于任一单独方案？在哪些方面帮助最大？
4. 将 Qwen3-VL-30B 替换为更小的 VLM（Qwen2.5-VL-7B）。测量准确率-每美元曲线。
5. 添加手写笔记支持。渲染手写语料库，使用 ColQwen 嵌入，测量检索效果。与手写 OCR 管道进行对比。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| 后期交互 | "ColPali 风格检索" | 查询 token 独立与页面图像块评分；MaxSim 聚合 |
| 多向量 | "每图像块嵌入" | 每个文档有多个向量，而非一个池化向量 |
| MaxSim | "后期交互评分" | 对于每个查询 token，取与文档向量的最大相似度；求和 |
| DocPruner | "图像块压缩" | 2026 年剪枝技术，保留 50% 的图像块，精度损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026 年衡量视觉文档检索的标准基准 |
| 证据区域 | "引用边界框" | 源页面上定位答案片段的边界框 |
| OCR 回退 | "公式通道" | 在视觉方案之外，为公式或表格密集的页面使用的文本管道 |

## 扩展阅读

- [ColPali（Illuin Tech）仓库](https://github.com/illuin-tech/colpali) — 参考后期交互文档检索
- [ColPali 论文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [Hugging Face 上的 ColQwen 系列](https://huggingface.co/vidore) — 生产就绪的检查点
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 参考服务栈
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 备选索引
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 备选托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 回退方案
