# 开源 VLM 配方：真正重要的因素

> 2024-2026 年的开源 VLM 文献是一片消融实验表格的森林。Apple 的 MM1 测试了图像编码器、连接器和数据混合的 13 种组合。Allen AI 的 Molmo 证明了详细人工描述优于 GPT-4V 蒸馏。Cambrian-1 进行了 20 多种编码器对比。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在受控基准上比较了 27 种训练方案。在所有这些噪音中，有一小部分结果在论文之间保持一致：图像编码器比连接器架构更重要，数据混合比两者都更重要，而详细人工描述优于蒸馏的合成数据。本课替你阅读这些表格，让你不必亲自去啃。

**类型：** 学习 + 实验
**语言：** Python（标准库，消融实验表格解析器 + 配方选择器）
**前置要求：** 第 12 阶段 · 05（LLaVA 基线）
**时间：** 约 180 分钟

## 学习目标

- 说出 VLM 的五轴设计空间：图像编码器、连接器、LLM、数据混合、分辨率调度。
- 阅读 MM1 / Idefics2 / Cambrian-1 的消融实验表格，预测哪个旋钮会改变给定基准的分数。
- 在给定算力预算和任务组合的情况下，为新的 VLM 选择一个配方（编码器、连接器、数据、分辨率）。
- 解释为什么在相同 token 数量下，详细人工描述优于 GPT-4V 蒸馏。

## 问题

市面上有数百个开源 VLM。「好」和「最先进」之间的大部分差距并不在于架构，而在于数据、分辨率调度和编码器选择。知道在你的模型表现不佳时应该首先转动哪个旋钮，可以让你避免 500 万 GPU 小时的错误。

2023 年的浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）基于描述对预训练 + LLaVA-Instruct-150k 运行。这是不错的基线，但最高只达到 MMMU 35% 左右。

2024 年的浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）进行了详尽的消融实验。结果令人惊讶且极具实用价值。

## 概念

### 五轴设计空间

Idefics2（Laurençon 等人，2024）命名了这些轴：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch 大小、分辨率和预训练目标上各不相同。
2. 连接器。MLP（2-4 层）、Q-Former（32 个查询 + 交叉注意力）、Perceiver Resampler（64 个查询）、C-Abstractor（卷积 + 双线性池化）。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 的大小是主要的参数成本。
4. 训练数据。描述对（CC3M、LAION）、交错数据（OBELICS、MMC4）、指令数据（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率调度。固定 224/336/448、AnyRes、原生动态。训练期间逐步提升或保持不变。

每个生产级 VLM 在每个轴上都会做出选择。MMMU 分数的大部分方差由轴 1、4 和 5 解释——而不是你选择了哪个连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节表明：从 CLIP ViT-L/14 换成 SigLIP SO400m/14 增加了 3 分以上的 MMMU。将连接器从 MLP 换成 Perceiver Resampler 增加的不到 1 分。Idefics2 复现了这一点：SigLIP > CLIP，在相同 token 数量下 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的「Cambrian Vision Encoders Match-Up」（Tong 等人，2024）在视觉中心基准（CV-Bench）上运行了 20 多种编码器。排行榜顶部是 DINOv2 和 SigLIP 的混合体；CLIP 处于中游；ImageBind 和 ViT-MAE 排名较低。从 CLIP ViT-L 到 DINOv2 ViT-g/14 的差距在 CV-Bench 上约为 5-7 分。

2026 年开源 VLM 的默认编码器是 SigLIP 2 SO400m/14 用于语义 + 密集特征，有时与 DINOv2 ViT-g/14 特征拼接（Cambrian 的「Spatial Vision Aggregator」就是这样做的）。

### 轴 2：连接器设计无关紧要

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出了相同的结论：在固定的视觉 token 数量下，连接器架构几乎无关紧要。在相同的 token 预算下，对均值池化 patch 使用 2 层 MLP 的性能与 32 查询 Q-Former 相差不到 1 分。

真正重要的是 token 数量。更多的视觉 token = 更多的 LLM 计算 = 更好的性能，但到一定程度后收益递减。每张图像 64 个 token 对于 OCR 来说太少。576-1024 个 token 是大多数开源 VLM 的最佳点。2048 个以上仅对文档和图表有帮助。

Q-Former 与 MLP 是一个成本问题，而非质量问题：无论图像分辨率如何，Q-Former 将 token 限制在 32-64 个；MLP 输出所有 patch token。对于高分辨率输入，Q-Former 节省 LLM 上下文；对于低分辨率，差异只是噪音。

### 轴 3：LLM 大小决定上限

将 LLM 从 7B 翻倍到 13B，在每篇 VLM 论文中都能可靠地在 MMMU 上增加 2-4 分。到 70B 时，大多数基准趋于饱和。VLM 的多模态推理上限就是 LLM 的文本推理上限——视觉编码器只能为其提供信息，而不能代替它进行推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro 和 ScreenSpot-Pro 上表现碾压：语言大脑太大了。一个 7B 的 VLM 无法通过巧妙的连接器设计来替代 70B 的 VLM。

### 轴 4：数据——详细人工描述优于蒸馏

Molmo + PixMo（Deitke 等人，2024）是 2024 年每个人都应该阅读的成果。Allen AI 让人类标注员在 1-3 分钟的密集语音转文字过程中描述图像，产生了 71.2 万张密集描述的图像。训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11 个基准测试中的 11 个上击败了 Llama-3.2-90B-Vision。差距不在于架构——而在于描述质量。详细人工描述每张图像包含的信息量是简短网页描述的 5-10 倍，并且在 GPT-4V 蒸馏会产生幻觉的地方保持事实准确性。

ShareGPT4V（Chen 等人，2023）和 Cauldron（Idefics2）采用了相同的策略，使用人工 + GPT-4V 混合描述。趋势很明显：对于 2026 年的前沿来说，描述密度 > 描述数量 > 蒸馏的便利性。

### 轴 5：分辨率及其调度

Idefics2 的消融实验：384 -> 448 增加 1-2 分。448 -> 980 配合图像分割（AnyRes）在 OCR 基准上再增加 3-5 分。固定分辨率训练在中等准确率时趋于平台期；分辨率逐步提升（从 224 开始，到 448 或原生分辨率结束）训练更快且最终结果更高。

Cambrian-1 进行了分辨率与 token 的权衡实验：在固定算力下，你可以在较低分辨率下获得更多 token，或在较高分辨率下获得更少 token。更高分辨率在 OCR 上胜出；低分辨率-更多 token 在通用场景理解上胜出。

2026 年的生产配方：阶段 1 以固定 384 训练，阶段 2 使用动态分辨率，对于 OCR 密集型任务最高可达 1280。

### Prismatic 受控比较

Prismatic VLMs（Karamcheti 等人，2024）是控制了所有轴的论文。相同的 13B LLM，相同的指令数据，相同的评估——每次只变化一个轴。结果：

- 每张图像的视觉 token 数量解释了约 60% 的方差。
- 编码器选择解释了约 20%。
- 连接器架构解释了约 5%。
- 其他所有因素（数据混合、调度器、学习率）占剩余的约 15%。

这是一个粗略的分解，但它是文献中关于「我应该首先消融什么」的最清晰的答案。

### 2026 年的方案选择器

基于已有证据，2026 年新项目的默认开源 VLM 配方：

- 编码器：SigLIP 2 SO400m/14 以原生分辨率配合 NaFlex，如果需要分割/定位则与 DINOv2 ViT-g/14 拼接以获取密集特征。
- 连接器：对 patch token 使用 2 层 MLP。除非你受到 token 限制，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，7B 用于控制成本，70B 用于追求质量，根据目标延迟选择。
- 数据：PixMo + ShareGPT4V + Cauldron，辅以任务特定的指令数据。
- 分辨率：动态（每边长边最小 256，最大 1280 像素）。
- 调度：阶段 1 对齐（仅训练投影器），阶段 2 全量微调，阶段 3 任务特定微调。

以上每一个默认选择都可以追溯到本课末尾引用的论文中经过测量的消融实验。

## 使用方式

`code/main.py` 是一个消融表解析器和配方选择器。它编码了 MM1 和 Idefics2 的消融表（精简版），支持以下查询：

- "给定预算 X 和任务 Y，哪个配方最优？"
- "如果我在 7B Llama 上将 SigLIP 替换为 CLIP，预期的 MMMU 变化是多少？"
- "要达到 80% 置信度的答案，我应该优先消融哪个维度？"

输出是一个带排名的配方列表，包含预期基准得分变化以及"优先消融"的建议。

## 产出物

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、算力预算和延迟目标，它会输出一份完整的配方（编码器、连接器、大语言模型、数据配比、分辨率调度方案），并引用支持每项选择的消融实验。让工程师在启动新的 VLM 项目时，不必每次都重新发明 Idefics2 的消融表。

## 练习

1. 阅读 MM1 第 3.2 节。在固定 2B 大语言模型、预算 50M 张图片的条件下，哪个编码器胜出？如果将大语言模型扩大到 13B，答案会反转吗？为什么？

2. Cambrian-1 发现，将 DINOv2 与 SigLIP 拼接使用在视觉中心型基准上优于单独使用其中任何一个，但在 MMMU 上没有额外提升。预测哪些基准会受益，哪些会持平。

3. 你的目标是基于 2B 大语言模型构建一个移动端 UI 智能体。选择编码器、连接器、分辨率和数据配比。用具体的消融表为每个选择提供理由。

4. Molmo 发布了 4B 和 72B 两个模型。4B 模型可与闭源 7B VLM 媲美；72B 模型在 11/11 项基准上超越 Llama-3.2-90B-Vision。这对"大语言模型规模平台期假说"意味着什么？

5. 设计一个消融表，在 7B VLM 上将数据配比质量与编码器质量的影响分离开来。最少需要多少次训练运行？提出四种维度设置方案。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Ablation（消融） | "只动一个旋钮" | 进行多轮训练，每轮只在设计空间的一个维度上有所不同，其他条件完全保持不变 |
| Connector（连接器） | "桥接模块" / "投影器" | 将视觉编码器输出映射到大语言模型 token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| Detailed human caption（详细人工描述） | "密集描述" | 由人工撰写的多句描述（通常 80-300 个 token），比网页 alt 文本更丰富 |
| Distillation（蒸馏） | "GPT-4V 描述" | 由更强的闭源 VLM 生成的训练数据；使用方便，但容易继承幻觉 |
| AnyRes / dynamic res（任意分辨率 / 动态分辨率） | "高分辨率路径" | 通过分块或 M-RoPE 方式，将大于编码器原生分辨率的图像输入模型的策略 |
| Resolution ramp（分辨率渐进） | "课程式调度" | 从低分辨率开始、逐步提高的训练计划，以加速对齐学习 |
| Vision-centric bench（视觉中心型基准） | "CV-Bench / BLINK" | 侧重细粒度视觉感知而非语言推理的评测 |
| PixMo | "Molmo 的数据集" | Allen AI 的 712K 密集描述图像数据集；将人类语音转写为密集描述 |

## 扩展阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
