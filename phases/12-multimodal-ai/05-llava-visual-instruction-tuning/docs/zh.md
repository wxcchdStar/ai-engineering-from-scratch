# LLaVA 与视觉指令微调

> LLaVA（2023 年 4 月）是全世界被复制最多的多模态架构。它用 2 层 MLP 替换了 BLIP-2 的 Q-Former，用朴素的 token 拼接替换了 Flamingo 的门控交叉注意力，并在 GPT-4 基于纯文本描述生成的 158k 条视觉指令对话上进行了训练。任何在 2023 年到 2026 年间构建过 VLM 的实践者，几乎都构建了 LLaVA 的某种变体。LLaVA-1.5 添加了 AnyRes（多分辨率切分）。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 在一个配方中统一了单图像、多图像和视频。本课将解读这一配方，实现投影器，并解释为什么"更简单的一方赢了"。

**类型：** 构建
**语言：** Python（标准库，投影器 + 指令模板构建器）
**前置课程：** 第 12 阶段 · 02（CLIP），第 11 阶段（LLM 工程 — 指令微调）
**时间：** 约 180 分钟

## 学习目标

- 构建一个 2 层 MLP 投影器，将 ViT patch 嵌入（维度 1024）映射到 LLM 的嵌入维度（维度 4096）。
- 走通 LLaVA 的两阶段配方：(1) 在 558k 图文对上做投影器对齐，(2) 在 158k 条 GPT-4 生成的对话上进行视觉指令微调。
- 构建 LLaVA 格式的 prompt，包含图像 token 占位符、系统 prompt 以及用户/助手对话轮次。
- 解释为什么社区从 Q-Former 转向 MLP，尽管 Q-Former 在 token 预算上更优。

## 问题所在

BLIP-2 的 Q-Former（第 12.03 课）将一张图像压缩为 32 个 token。干净、高效，在基准测试上表现不错。但它有两个问题。

首先，Q-Former 是可训练的，但其损失函数并非最终任务。阶段 1 训练 ITC+ITM+ITG。阶段 2 训练 LM 损失。查询向量学到的是某种中间表示，LLM 随后必须对其进行解码。信息在瓶颈中丢失。

其次，Q-Former 有 1.88 亿参数，而且在 LLaVA 2023 年的规模下，你必须将其与目标 LLM 协同设计。换了 LLM，就得重新训练 Q-Former。换了视觉编码器，也得重新训练。每一种组合都是一个独立的研发项目。

LLaVA 的答案简单到令人尴尬：取 ViT 的 576 个 patch token，每个通过一个 2 层 MLP（`1024 → 4096 → 4096`），然后把全部 576 个 token 直接丢进 LLM 的输入序列。没有瓶颈。没有基于奇怪目标的阶段 1 预训练。直接用 LM 损失训练 MLP。

数据从哪来？LLaVA 的第二个洞察：用 GPT-4（纯文本）生成指令数据。把一张图像的 COCO 描述和边界框数据喂给 GPT-4，让它生成对话、详细描述和复杂推理问题。158k 条指令-回答对话轮次，零成本。无需人工标注。

结果：一个 VLM 在 8 块 A100 上跑一天就能训练出来，在 MMMU 上击败了 Flamingo，并且发布了社区可以扩展的开放 checkpoint。到 2023 年底，它已经催生了 50 多个分支。

## 核心概念

### 架构

LLaVA-1.5 在 13B 规模下：
- 视觉编码器：CLIP ViT-L/14 @ 336（阶段 1 冻结，阶段 2 可选解冻）。
- 投影器：2 层 MLP，使用 GELU 激活函数，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来是 Llama-3.1-8B）。

对图像 + 文本 prompt 的前向传播：

```
img -> ViT -> 576 个 patch，维度 1024
patches -> MLP -> 576 个 token，维度 4096
prompt: system + "<image>" 占位符 + 用户问题
将 <image> token 替换为 576 个投影后的 token
将完整序列输入 LLM
解码回复
```

图像占据 LLM 上下文中的 576 个 token。在 2048 上下文下，文本还剩 1472 个 token。在 32k 上下文下，这点开销几乎可以忽略不计。

### 阶段 1：投影器对齐

冻结 ViT。冻结 LLM。只训练 2 层 MLP。数据集：558k 图文对（LAION-CC-SBU）。损失：在投影后的图像 token 条件下，对描述进行语言建模。

在 batch 128 下一个 epoch，几小时就完成了。投影器学会将 ViT 空间映射到 LLM 空间。无需任务特定的监督。

### 阶段 2：视觉指令微调

解冻投影器（保持可训练）。解冻 LLM（通常是全量解冻，有时使用 LoRA）。在 158k 条视觉指令对话上训练。

指令数据是关键技巧。Liu 等人通过以下方式生成了这些数据：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工描述 + 边界框列表）。
3. 用三种 prompt 模板发送给 GPT-4：
   - 对话："生成一段用户和助手之间关于这张图像的来回对话。"
   - 详细描述："给出这张图像的丰富、详细的描述。"
   - 复杂推理："提出一个需要关于这张图像进行推理的问题，然后回答它。"
4. 将 GPT-4 的输出解析为（指令，回复）对。

这一切都没有直接接触图像——只用了文本描述。GPT-4 对可信的图像内容进行幻觉生成。有些噪声，但确实有效：158k 条对话轮次足以解锁对话能力。

### 为什么社区复制了这个方案

- 没有阶段 1 特定的损失需要调参。全程使用 LM 损失。
- 投影器数小时就能训练完成，而不是数天。
- LLM 可以替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重新训练投影器。
- 视觉指令数据管线使用 GPT-4，针对新领域重新生成成本很低。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）增加了：
- 在指令微调中混入学术任务数据（VQA、OKVQA、RefCOCO）。
- 更好的系统 prompt。
- 2048 → 32k 上下文。

LLaVA-NeXT（2024 年 1 月）增加了：
- AnyRes（多分辨率切分）：将高分辨率图像切分为 2×2 或 1×3 的 336×336 裁剪网格，外加一张全局低分辨率缩略图。每块裁剪产生 576 个 token；每张图像总计约 2880 个视觉 token。OCR 和图表任务表现大幅提升。
- 更好的指令数据混合，加入了 ShareGPT4V（高质量的 GPT-4V 描述）。
- 更强的基座 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第 12.08 课将深入讲解 OneVision。简要版本：相同的投影器，但采用了一种课程训练策略，在共享视觉 token 预算下，用一个模型覆盖单图像、多图像和视频。

### 与 Q-Former 的对比

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每张图像的视觉 token 数 | 32 | 576（基础）或 2880（AnyRes（多分辨率切分）） |
| 可训练参数量 | 188M + LM | 40M + LM |
| 阶段 1 损失 | ITC+ITM+ITG | 仅 LM |
| LLM 即插即用 | 需要重新训练 | 更换 LLM 仅需极少重新训练 |
| 多图像 | 笨拙 | 自然（拼接） |
| 视频 | 笨拙 | 自然（逐帧拼接） |
| Token 预算 | 小 | 大 |

MLP 在简洁性和 token 灵活性上胜出。Q-Former 在 token 预算上胜出。到 2023 年底，token 预算已不再是决定性约束（LLM 上下文扩展到 32k-128k+），简洁性占据了主导地位。

### Prompt 格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是一个占位 token。在分词之前，它被替换为 576 个视觉 token（或 AnyRes（多分辨率切分）下的 2880 个）。分词器看到的序列比训练时稍长，但 LLM 能够处理这种新颖的输入，因为阶段 1 已经教会了它这样做。

### 参数量经济性

LLaVA-1.5-7B 分解：
- CLIP ViT-L/14 @ 336：303M（阶段 1 冻结，阶段 2 通常解冻）。
- 投影器（2 层线性层）：约 22M 可训练参数。
- Llama-7B：7B。
- 总计：7.3B 参数。阶段 2 期间可训练：完整的 7B + 22M 投影器。

阶段 2 的训练成本：约 20 小时在 8×A100 上。这是关键数字——一天、一个节点、可复现。这就是 LLaVA 得以广泛传播的原因。

## 使用

`code/main.py` 实现了：

1. 用纯 Python 编写的 2 层 MLP 投影器（玩具级维度 16 → 32 → 32）。
2. 提示构建管线：系统提示 + `<image>` 被替换为 N 个投影后的视觉 token + 用户轮次 + 助手生成占位符。
3. 一个可视化工具，展示 576 个 token 的视觉块在 LLM 上下文中占多少比例（分别对比 2k / 32k / 128k 上下文窗口）。

## 交付

本课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 系列检查点，它运行一套 10 条提示的"体感评估"套件（3 条描述、3 条视觉问答、2 条推理、2 条拒绝），并输出一份人类可读的评分卡。这不是一个基准测试，而是一个冒烟测试，用于确认投影器和 LLM 之间是否连接良好。

## 练习

1. 计算维度为 `1024 → 4096 → 4096` 的 2 层 MLP 投影器的可训练参数数量。加上 GELU 和偏置后，它在 LLaVA-13B 中占多大比例？

2. 为一个"拒绝"场景构造一条 LLaVA 提示——图像中包含一个私人个体。写出预期的助手回复。为什么 LLaVA 应该以零样本方式拒绝这个请求，需要哪些训练数据来强化这种拒绝行为？

3. 阅读 LLaVA-NeXT 博客中的 AnyRes 部分。计算一张 1344×672 图片在 AnyRes 下的视觉 token 数量。与基础 336×336 下的 576 个 token 进行对比。

4. LLaVA 第一阶段投影器使用描述文本上的 LM 损失进行训练。如果跳过第一阶段直接进入第二阶段（视觉指令微调）会发生什么？请引用 Prismatic VLMs 消融研究（arXiv:2402.07865）来回答。

5. LLaVA-Instruct-150k 使用 GPT-4 配合 COCO 描述来生成指令。对于一个新领域（医学 X 光片、卫星图像），描述生成领域指令的四步数据管线。每一步可能出现什么问题？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| Projector（投影器） | "MLP 桥接层" | 2 层带 GELU 的 MLP，将 ViT 维度映射到 LLM 维度 |
| Image token（图像 token） | "`<image>` 占位符" | 提示中的标记，在推理前被替换为 N 个投影后的视觉 token |
| Visual instruction tuning（视觉指令微调） | "LLaVA 第二阶段" | 在 GPT-4 生成的（图像、指令、回复）三元组上进行训练 |
| Stage 1 alignment（第一阶段对齐） | "投影器预训练" | 冻结 ViT 和 LLM，仅用描述文本上的 LM 损失训练投影器 |
| AnyRes | "多裁剪平铺" | 将高分辨率图像分割成瓦片网格，并将每个瓦片的视觉 token 拼接起来 |
| LLaVA-Instruct | "GPT-4 生成的" | 从 COCO 描述 + GPT-4 合成的 158k 条指令-回复对 |
| Vision encoder freeze（视觉编码器冻结） | "骨干网络锁定" | CLIP 权重在第一阶段不更新，有时第二阶段也不更新 |
| ShareGPT4V | "更好的描述" | GPT-4V 生成的 100 万条密集描述，用于更高质量的对齐 |
| VQA（视觉问答） | "Visual question answering" | 对图像中的自由形式问题进行回答的任务 |
| Prismatic VLMs | "设计空间论文" | Karamcheti 2024 年的消融研究，系统性地测试投影器和数据选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集描述数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融研究。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一的单图、多图、视频模型。
