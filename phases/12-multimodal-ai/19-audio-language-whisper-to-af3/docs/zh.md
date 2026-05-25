# 音频-语言模型：从 Whisper 到 Audio Flamingo 3 的演进

> Whisper（Radford et al., 2022年12月）奠定了语音识别的基础——68万小时弱监督多语言语音数据、一个简单的编码器-解码器 Transformer、一个让后续所有 ASR 发布都引用的基准。但识别不等于推理。问"这段录音里有什么乐器"或"说话人表达了什么情绪"或"第3分钟发生了什么"需要的是音频理解，而非转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3, 2025年7月）逐步构建了这个技术栈：保留 Whisper 级编码器，接上 Q-former，在音频-文本指令数据上训练，加入链式思维推理。本课解读这一演进弧线。

**类型：** 构建（Build）
**语言：** Python（标准库，log-Mel 频谱图 + 音频 Q-former 骨架）
**前置课程：** Phase 6（语音与音频）, Phase 12 · 03（Q-Former）
**时长：** ~180 分钟

## 学习目标

- 从波形计算 log-Mel 频谱图：加窗、FFT、滤波器组、对数变换。
- 比较编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合方案。各自的优势场景。
- 构建音频 Q-former：N 个可学习查询对频谱图 patch 做交叉注意力。
- 解释级联式（Whisper 转录后接 LLM）与端到端音频 LLM 训练：为什么端到端在推理任务上扩展性更好。

## 问题

语音识别已被 Whisper 解决。"音频 OCR"已是基础能力。但"基础能力"止步于转录。如果模型无法对其听到的内容进行推理——时间、说话人、情绪、音乐结构、环境音——仅凭转录无法驱动产品功能。

三条显而易见的路线：

1. 级联：Whisper 转录，LLM 对转录文本推理。适用于纯语音场景。对音乐、环境音频、多说话人重叠、情绪分析无效。

2. 端到端音频 LLM：音频编码器将音频 token 直接喂入 LLM，跳过转录。保留声学信息（情绪、说话人、环境）。需要新的训练数据。

3. 混合式：音频编码器 + 文本解码器，既能转录也能推理。Qwen-Audio 和 Audio Flamingo 选择此路线。

## 核心概念

### Log-Mel 频谱图：输入特征

每个音频编码器都从相同的特征开始：log-Mel 频谱图。

1. 重采样到 16 kHz。
2. 短时傅里叶变换，25ms 窗口，10ms 步进。
3. 取 FFT 结果的幅度。
4. 应用 Mel 滤波器组（通常80个滤波器，在 0-8000 Hz 间对数分布）将频率映射到感知尺度。
5. 对数压缩（log(1 + x)）处理动态范围。

结果：形状为 (T, 80) 的二维数组，T 为时间帧数。30秒片段在100 Hz 帧率下为 (3000, 80)。

### Whisper 的编码器

Whisper 的编码器是一个12层 ViT 风格 Transformer，将 log-Mel 频谱图作为时间帧序列处理。输出：每个时间帧一个隐藏状态向量。

对于 ASR，Whisper 的解码器是一个交叉注意力 Transformer，基于编码器输出生成文本 token。标准编码器-解码器架构。

对于 ALM（音频大语言模型），你需要将编码器输出作为另一个 LLM 的输入。模式：Whisper 编码器冻结，Q-former 可训练，LLM 冻结或微调。

### BEATs 和音频专用编码器

Whisper 主要在以语音为主的数据上训练。在音乐和环境音频上表现较弱。

BEATs（Chen et al., 2022）是在 AudioSet 上训练的自监督 Transformer。在相同参数量下，对音乐和环境音的捕获能力优于 Whisper。

AF-Whisper（Audio Flamingo 3 的混合方案）：将 Whisper + BEATs 特征拼接作为音频输入。Whisper 携带语言信号，BEATs 携带声学信号。

### 音频 Q-former

与 BLIP-2 的视觉 Q-former 模式相同。固定数量的可学习查询（通常32或64个）对音频编码器的输出帧做交叉注意力。查询变为 LLM 消费的音频 token。

对齐训练阶段：仅 Q-former，在音频-文本对（AudioCaps, Clotho）上使用对比 + 描述损失。指令阶段：端到端，解冻 LLM，在指令数据上训练。

### 演进弧线 — SALMONN、Qwen-Audio、AF3

SALMONN（Tang et al., 2023）：Whisper + BEATs + Q-former + LLaMA。第一个具有严肃推理能力的开源音频 LLM。MMAU 基准约0.55综合分。

Qwen-Audio（Chu et al., 2023）：类似架构，在更丰富的数据集上训练，针对多轮对话调优。MMAU 约0.60。

LTU — Listen, Think, Understand（Gong et al., 2023）：显式推理数据，专注于对音频片段的链式思维推理。规模较小但更专注。

Audio Flamingo 3（Goel et al., 2025年7月）：当前开源 SOTA。8B LLM 主干（Qwen2 7B），Whisper-large 编码器拼接 BEATs，64查询 Q-former，在100万+音频-文本指令对上训练。MMAU 0.72，在部分子任务上匹配闭源前沿。

AF3 还引入了按需链式思维 (on-demand chain-of-thought) 用于音频：模型可以选择性地在最终答案前输出思考 token（"让我先识别乐器：..."）。启用思考时，复杂推理任务准确率提升3-5个百分点。

### 级联式 vs 端到端

级联管线：

1. Whisper 将音频转录为文本。
2. LLM 对文本进行推理。

对"总结这期播客"完美适用。对以下场景失败：
- "这首歌的情绪是什么？"——情绪在声音中，不在文字里。
- "说话的是 Alice 还是 Bob？"——需要说话人识别。
- "爆炸发生在第几秒？"——时序定位在文本中丢失。
- "这是真实音频还是生成的？"——深度伪造检测需要声学特征。

端到端保留声学信号。Qwen-Audio 和 AF3 原生处理音乐、环境和情绪。

### 2026年生产方案

构建新的音频理解产品：

- 级联式适用于：转录是目标、无音乐、无情绪推理。
- AF3 / Qwen-Audio 系列适用于：音乐、情绪、多说话人或复杂音频推理。

级联式更便宜更简单。端到端能力更强。

### MMAU — 音频推理基准

MMAU（Massive Multimodal Audio Understanding，大规模多模态音频理解）是2024-2025年的音频推理基准：

- 10,000个音频-文本问答对，覆盖语音、音乐、环境音。
- 涵盖分类、时序推理、因果推理、开放问答。
- 测试级联管线系统性遗漏的能力。

开源 SOTA（AF3）为0.72；闭源前沿约0.78（Gemini 2.5 Pro, Claude Opus 4.7）。差距小于 VideoMME 的开源-闭源差距，表明音频 LLM 正在成熟。

## 动手实践

`code/main.py`：

- 用标准库实现 log-Mel 频谱图计算：加窗、朴素 DFT、Mel 滤波器组。
- 音频 Q-former 骨架：给定编码器输出帧，计算 Q、K、V、注意力，输出 N 个 token。
- 级联式 vs 端到端在简易任务上的对比。

## 交付产出

本课产出 `outputs/skill-audio-llm-pipeline-picker.md`。给定一个音频任务（转录、音乐标注、情绪推理、多说话人分离、环境分类），选择级联式、端到端 AF3 或混合方案。

## 练习

1. 计算30秒片段在16kHz、25ms窗口、10ms步进、80个 Mel 频段下的 log-Mel 频谱图维度。48kHz 时如何变化？

2. 为什么 Whisper 在音乐上表现不佳？BEATs 捕获了 Whisper 未捕获的哪些音频特征？

3. 音频 Q-former 用64个查询 vs 32个：在什么任务复杂度下64个有回报？32个为什么场景节省计算？

4. 阅读 AF3 第4节关于按需思考的内容。提出三个链式思维帮助最大的音频任务。

5. 使用 AF3 的输出实现一个最小化的说话人分离管线。如何标示说话人切换？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Log-Mel 频谱图 (Log-Mel spectrogram) | "Mel 特征" | 经 Mel 滤波器组后的对数幅度值二维数组 (时间, 频率) |
| 音频 Q-former (Audio Q-former) | "音频感知器" | 从音频编码器输出到固定长度查询的交叉注意力瓶颈，喂给 LLM |
| 级联式 (Cascaded) | "ASR 后接 LLM" | Whisper 转录后文本 LLM 推理的管线；丢失声学信息 |
| 端到端 (End-to-end) | "音频 LLM" | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | "AudioSet 音频编码器" | 在 AudioSet 上训练的自监督 Transformer；擅长音乐 + 环境音 |
| MMAU | "音频推理基准" | 10k 问答对覆盖语音、音乐、环境；2024年评估标准 |
| 按需思考 (On-demand thinking) | "音频 CoT" | 模型可选择在最终答案前输出推理 token，提升准确率3-5个百分点 |

## 延伸阅读

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
