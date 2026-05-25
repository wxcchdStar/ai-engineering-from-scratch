# 文档与图表理解

> 文档不是照片。一个 PDF、科学论文、发票或手写表单有版面、表格、图表、脚注、标题和语义结构——普通的图像理解无法捕获这些。VLM 之前的技术栈是一条管线：Tesseract OCR + LayoutLMv3 + 表格提取启发式规则。VLM 浪潮用免 OCR 模型取代了它——Donut（2022）、Nougat（2023）、DocLLM（2023）——直接输出结构化标记。到2026年，前沿做法就是"将页面图像以2576px原生分辨率喂给 Claude Opus 4.7"，结构化标记输出自动获得。本课解读文档 AI 的三时代演进。

**类型：** 构建（Build）
**语言：** Python（标准库，版面感知文档解析器骨架）
**前置课程：** Phase 12 · 05（LLaVA）, Phase 5（NLP）
**时长：** ~180 分钟

## 学习目标

- 解释文档 AI 的三个时代：OCR 管线、免 OCR、VLM 原生。
- 描述 LayoutLMv3 的三路输入：文本、版面（边界框）、图像 patch，以及统一掩码。
- 比较 Donut（免 OCR，图像 → 标记）、Nougat（科学论文 → LaTeX）、DocLLM（版面感知生成式）、PaliGemma 2（VLM 原生）。
- 为新任务（发票、科学论文、手写表单、中文收据）选择文档模型。

## 问题

"理解这个 PDF"看似简单实则很难。信息存在于：

- 文本内容（90%的信号）。
- 版面（标题、脚注、侧边栏、双栏格式）。
- 表格（行、列、合并单元格）。
- 图形和图表。
- 手写标注。
- 字体和排版（标题 vs 正文）。

原始 OCR 只提取文本而丢失其余信息。关心发票的系统需要知道"合计：$1,245"来自右下角，而非脚注。

## 核心概念

### 时代一 — OCR 管线（2021年前）

经典技术栈：

1. PDF → 每页图像。
2. Tesseract（或商业 OCR）提取带逐词边界框的文本。
3. 版面分析器识别区块（标题、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + 正则表达式提取字段。

对清晰印刷文本有效。在手写、倾斜扫描、复杂表格、非英文脚本上失败。每种失败模式需要自定义异常路径。

### TrOCR（2021）

TrOCR（Li et al., arXiv:2109.10282）用 Transformer 编码器-解码器替代了 Tesseract 经典的 CNN-CTC，在合成 + 真实文本图像上训练。在手写和多语言文本上明显胜出。仍然是管线式（检测器然后 TrOCR 然后版面），但 OCR 步骤大幅改进。

### 时代二 — 免 OCR（2022-2023）

首批免 OCR 模型说：完全跳过检测，直接从图像像素映射到结构化输出。

Donut（Kim et al., arXiv:2111.15664）：
- 编码器-解码器 Transformer，编码器为 Swin-B。
- 输出是表单理解的 JSON、摘要的 markdown 或任何任务特定 schema。
- 无 OCR、无版面、无检测。

Nougat（Blecher et al., arXiv:2308.13418）：
- 专门针对科学论文训练。
- 输出为 LaTeX / markdown。
- 处理公式、多栏版面、图形。
- 每个 arXiv 解析器都在调用的模型。

这些是专家模型，不是通用模型。Donut 处理科学论文会失败；Nougat 处理发票会失败。

### LayoutLMv3（2022）

另一条路线。LayoutLMv3（Huang et al., arXiv:2204.08387）保留 OCR 但增加版面理解：

- 三路输入：OCR 文本 token、每 token 的2D边界框、图像 patch。
- 跨所有三种模态的掩码训练目标（掩码文本、掩码 patch、掩码版面）。
- 下游任务：分类、实体提取、表格问答。

LayoutLMv3 是基于 OCR 的文档理解的巅峰。在表单和发票上表现强劲。需要上游 OCR。在标准化文档基准上是 VLM 之前的最佳准确率。

### DocLLM（2023）

DocLLM（Wang et al., arXiv:2401.00908）是 LayoutLM 的生成式兄弟。基于版面 token 条件生成自由格式答案。对文档问答更好；仍依赖 OCR 输入。

### 时代三 — VLM 原生（2024+）

2024年 VLM 已经足够好，可以完全取代管线。将完整页面图像以高分辨率喂给 VLM，提出问题，获得答案。

- LLaVA-NeXT 336-tile AnyRes 适用于小型文档。
- Qwen2.5-VL 动态分辨率原生处理 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档。
- PaliGemma 2（2025年4月）专门针对文档 + 手写训练。

VLM 原生与 OCR 管线之间的差距迅速缩小。到2026年，VLM 原生在以下场景胜出：

- 场景文本（手写 + 印刷，混合脚本）。
- 带合并单元格的复杂表格。
- 嵌入文本中的数学公式。
- 带文本标注的图形。

OCR 管线仍在以下场景胜出：

- 大规模纯扫描工作负载，每页延迟很重要。
- 管线可靠性（确定性故障 vs VLM 幻觉）。
- 需要可审计 OCR 输出的监管环境。

### Claude 4.7 / GPT-5 前沿

以2576像素原生输入，前沿 VLM 的文档理解接近人类准确率。2026年初的基准数据：

- DocVQA：Claude 4.7 ~95.1, PaliGemma 2 ~88.4, Nougat ~77.3, 管线式 LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2, GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

闭源模型的差距主要在于分辨率和基础 LLM 规模。7B 开源模型落后几个百分点但正在追赶。

### 数学公式与 LaTeX 输出

科学论文需要精确的 LaTeX 公式输出。Nougat 专门为此训练。以 LaTeX 为目标训练的 VLM（Qwen2.5-VL-Math、Nougat 衍生物）能产出可用的 LaTeX。没有显式 LaTeX 训练的 VLM 产出可读但不精确的转录。

2026年科学论文管线：对 PDF 运行 Nougat，再对棘手页面用 VLM。

### 手写

仍然是最难的子任务。混合印刷 + 手写（医生笔记、填写的表单）是 OCR 管线在成本上仍然胜过 VLM 的地方。纯手写 VLM 正在改进（Claude 4.7, PaliGemma 2）。

### 2026年方案

新文档 AI 项目：

- 大规模纯印刷发票：LayoutLMv3 + 规则，成本高效。
- 混合文档（科学 + 手写 + 表单）：VLM 原生（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 摄入：Nougat 处理数学，VLM 处理图形。
- 监管合规：OCR 管线 + VLM 验证器交叉检查。

## 动手实践

`code/main.py`：

- 简易版面感知分词器：给定（文本，边界框）对，产生 LayoutLMv3 风格输入。
- Donut 风格任务 schema 生成器：表单的 JSON 模板。
- 跨 OCR 管线、Donut、Nougat 和 VLM 原生的每页 token 预算对比。

## 交付产出

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定一个文档 AI 项目（领域、规模、质量、监管要求），选择 OCR 管线、免 OCR 专家模型或 VLM 原生。

## 练习

1. 你的项目每天处理1000万张发票。哪种技术栈在不损失准确率的前提下最小化每页成本？

2. 为什么 LayoutLMv3 在表单问答上优于纯 CLIP-VLM 但在场景文本上不如？边界框流放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM 原生输出在 LaTeX 保真度上胜过 Nougat 的测试用例，以及一个 Nougat 胜出的用例。

4. 阅读 PaliGemma 2 论文（Google, 2024）。相比 PaliGemma 1，提升文档准确率的关键训练数据添加是什么？

5. 设计一个监管安全的混合方案：OCR 管线为主，VLM 为辅交叉检查。如何解决分歧？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| OCR 管线 (OCR pipeline) | "Tesseract 风格" | 分阶段栈：检测 -> OCR -> 版面 -> 规则；确定性，脆弱 |
| 免 OCR (OCR-free) | "Donut 风格" | 跳过显式 OCR 的图像到输出 Transformer；单模型 |
| 版面感知 (Layout-aware) | "LayoutLM" | 输入包含每 token 的边界框坐标；跨模态统一掩码 |
| VLM 原生 (VLM-native) | "前沿 VLM" | 将页面图像以高分辨率直接喂给 Claude/GPT/Qwen VLM；无管线 |
| DocVQA | "文档基准" | 文档 VQA 标准；最常引用的分数 |
| 标记输出 (Markup output) | "LaTeX / MD" | 结构化输出格式而非自由文本；支持下游自动化 |

## 延伸阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
