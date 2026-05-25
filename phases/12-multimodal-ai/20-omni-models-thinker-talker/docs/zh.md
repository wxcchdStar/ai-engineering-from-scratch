# 全能模型：Qwen2.5-Omni 与 Thinker-Talker 拆分

> GPT-4o 在2024年5月的产品演示之所以颠覆性，并非因为底层模型，而是因为产品形态——一个语音界面，你说话、模型看到摄像头拍到的内容、并在250ms内回话。开源生态在2024和2025年剩余时间里都在追赶这一产品形态。Qwen2.5-Omni（2025年3月）是参考级开源设计：一个 Thinker（大型文本生成 Transformer）加一个 Talker（并行语音生成 Transformer），通过流式语音 token 连接。Mini-Omni 简化了它，Moshi 匹配了其延迟，GLM-4-Voice 将其扩展到中文。本课解读 Thinker-Talker 架构以及使流式实时对话得以运作的延迟预算。

**类型：** 构建（Build）
**语言：** Python（标准库，流式管线延迟模拟器 + VAD 循环）
**前置课程：** Phase 12 · 19（音频 LLM）, Phase 12 · 16（任意到任意）
**时长：** ~180 分钟

## 学习目标

- 将推理管线拆分为 Thinker（文本推理）和 Talker（语音合成），解释为什么并行流式能工作。
- 逐组件计算对话交互的首音频字节时间 (TTFAB) 预算。
- 描述 TMRoPE 在 Thinker 内部如何跨视觉、音频和文本进行时间对齐的位置编码。
- 列举三种实时对话模式：半双工、轮换发言、全双工。

## 问题

一个实时语音助手需要快速完成大量工作：

1. 听到用户。实时语音分词，语音活动检测 (VAD) 判断用户何时说完。
2. 可选地看。摄像头输入以 2-4 FPS 流入 Thinker，与音频并行。
3. 思考。基于对话历史生成回复。
4. 说话。合成音频 token，解码为波形，流式传输到用户扬声器。

每一步都增加延迟。对话感要求总往返时间 < 500ms——低于此值用户不再感知卡顿。GPT-4o 声称约250ms。Moshi 约160ms。Qwen2.5-Omni 约350-500ms。

每个组件都需要流式处理。没有什么可以"全部批处理再解码"。

## 核心概念

### Thinker 与 Talker

Qwen2.5-Omni 的分解：

- Thinker：一个 7B-80B 文本生成 Transformer。消费交错的文本 + 图像 + 音频 token。输出代表"要说什么"的文本 token。
- Talker：一个较小的语音生成 Transformer（200M-1B）。消费 Thinker 的文本输出 token 加上近期语音上下文 token。输出离散语音 token（残差 VQ 索引）。
- 语音解码器：流式波形解码器（SNAC, MoVQGAN 系列），将语音 token 实时转为音频采样。

这种分离很重要。Thinker 必须大才能有好的推理能力。Talker 可以小，因为它的任务是局部性的——将文本转为语音 token。更大的 Talker 不会更有表现力；只会更慢。

并行运行：

1. Thinker 输出文本 token t_i。
2. Talker 消费 t_i（通过流式传输）并输出语音 token s_i, s_{i+1}, ..., s_{i+k}。
3. 语音解码器收到语音 token 就立即输出音频采样。
4. 当 Thinker 到达文本 token t_{i+3} 时，Talker 已经为 t_0..t_{i+2} 流式输出了音频。

### TMRoPE — 时间对齐的多模态位置编码

Thinker 需要整合图像帧（以约 4 FPS 到达）、音频帧（以每秒50帧到达）和对话历史中的文本。简单的序列顺序（先所有图像，再所有音频，再文本）会丢失时间对齐。

TMRoPE 为每个 token 分配绝对时间戳。视觉 token 在 t=2.3s。音频 token 在 t=2.32s。用户文本 token "停" 在 t=2.35s。RoPE 按时间戳旋转注意力；模型将它们视为时间上并发。

这是"他边挥手边说你好"得以工作的基础设施——模型在同一个概念时刻看到视频帧和音频。

### 流式语音合成

语音 token 必须流式传输。Mini-Omni（Xie & Wu, 2024）引入了"语言模型可以边思考边流式说话"：Thinker 输出 token 和 Talker 输出 token 在同一序列中交错。Talker 在 Thinker 提交下一个文本 token 后立即发射。没有批处理边界。

Moshi（Défossez et al., 2024年10月）是最快的开源实现。单张 A100 上160ms TTFAB。架构：单个 7B Transformer 在交替位置输出文本和语音 token，带有"内心独白"来分离思考流和说话流。这实际上是经过精心训练将 Thinker + Talker 融合到一个模型中。

### VAD 与轮换发言

语音活动检测运行在输入侧。两种模式：

- 半双工：用户说，模型听。模型说，用户听。通过 VAD 静音检测（约200ms）明确交接。
- 全双工：双方可同时说话。模型可以做回应性信号（"嗯"）或打断。难度更大。Moshi 支持此模式。

Qwen2.5-Omni 默认支持半双工，通过静音阈值实现轮换发言。全双工需要应用层处理。

### Qwen3-Omni（2025年11月）

后继版本。Qwen3-80B Thinker，更大的 Talker，改进的 TMRoPE-v2。延迟接近 GPT-4o 的250ms。开放权重。OmniBench 基准与 Gemini 2.0 Live 竞争力相当。

### 生产延迟预算

典型流式交互：

- 麦克风 -> 音频 token：40-80ms。
- 预填充（提示 + 历史）：7B 时100-200ms，70B 时大幅增加。
- 首个 Thinker 文本 token：40ms。
- Talker 处理首个文本 token：20ms。
- 首批语音 token 提交：40ms。
- 残差 VQ 解码：30ms。
- 语音波形解码：50-80ms。

总 TTFAB：7B 时320-510ms，70B 时600-900ms。前沿质量通常意味着 70B+；因此存在前沿延迟差距。

### Token 速率计算

16kHz 语音以50 Hz 基础语音 token 计算，每秒输出需要50个语音 token。Talker 必须输出 ≥50 tok/s 才能跟上。以 H100 上典型 LLM 吞吐量30-80 tok/s 计算，一个小型（200-300M）Talker 足够快；7B Talker 会跟不上。

这就是为什么需要专用小型 Talker 模型，而非"直接用主模型"。

## 动手实践

`code/main.py`：

- 用模拟 token 输出速率模拟 Thinker-Talker 管线。
- 针对可配置的模型大小和麦克风采样率计算 TTFAB。
- 演示带 VAD 静音阈值的半双工轮换发言。

## 交付产出

本课产出 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集（视觉输入、双语、全双工），选择 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni，并确定 Thinker/Talker 的规模。

## 练习

1. 你的目标 TTFAB 是300ms。在 7B Thinker 和 300M Talker 上，逐一列出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述当用户在 t=1s 开始说话、摄像头在 t=1.2s 捕捉到一个手势时，模型看到的是什么。

3. 全双工支持要求模型在听的同时输出音频。提出一种教授此能力的训练数据格式。

4. 阅读 Moshi 论文第4节。描述"内心独白"分离以及为什么它避免了 Thinker-Talker 拆分。

5. 计算吞吐量预算：Talker 需要多快才能以每秒50个基础层 token 跟上16kHz 语音？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Thinker | "推理大脑" | 产生要说什么的大型文本生成 Transformer |
| Talker | "语音生成嘴巴" | 从 Thinker 文本生成离散语音 token 的小型 Transformer |
| TTFAB | "延迟预算" | 首音频字节时间：从用户说话结束到第一个音频采样输出 |
| TMRoPE | "时间对齐 RoPE" | 跨视觉、音频、文本使用绝对时间戳的位置编码 |
| 半双工 (Half-duplex) | "轮换发言" | 用户和模型交替；VAD 静音检测用户结束 |
| 全双工 (Full-duplex) | "同时" | 模型可以同时说和听；可做回应性信号 |
| 内心独白 (Inner monologue) | "Moshi 分离" | 单模型设计中思考流和说话流交错 |

## 延伸阅读

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
