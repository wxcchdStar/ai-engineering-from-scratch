# 视频-语言模型：时间 Token 与时序定位

> 视频不是一堆照片的堆叠。一个5秒的片段具有因果顺序、动作动词和事件时间信息——这是图像模型无法表征的。Video-LLaMA（Zhang et al., 2023年6月）推出了首个开源视频大语言模型，支持音视觉联合定位。VideoChat 和 Video-LLaVA 扩展了这一范式。到2025年，Qwen2.5-VL 的 TMRoPE 缩小了与前沿闭源模型的差距。每个系统以不同方式解决时间 Token (temporal tokens) 问题——Q-former 逐片段、拼接池化逐帧、TMRoPE 逐 Token。本课解读这些设计模式，构建均匀/动态帧采样器，并在时序定位任务上评估。

**类型：** 构建（Build）
**语言：** Python（标准库，帧采样器 + 时序定位评估器）
**前置课程：** Phase 12 · 08（LLaVA-OneVision）
**时长：** ~180 分钟

## 学习目标

- 解释为什么时间位置编码 (temporal positional encoding) 能独立于视觉编码器影响视频 VLM 的性能。
- 比较均匀采样、动态 FPS 和事件驱动帧采样在 token/秒与定位精度之间的权衡。
- 描述 Q-former-逐片段（Video-LLaMA）、池化-逐帧（Video-LLaVA）、M-RoPE-逐 Token（Qwen2.5-VL）三种设计。
- 列举四个视频基准测试：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题

一段1分钟、30 FPS 的视频有1800帧。以 ViT-B（224分辨率）每帧196个视觉 token 计算，总共352k token——超过了2024年所有 LLM 的上下文长度。

三种压缩策略：

1. 帧下采样（根据内容取 1-8 FPS）。
2. 对每帧的 patch token 做激进池化（3x3 或 4x4 双线性池化）。
3. 通过 Q-former 压缩：取16帧片段输出64个 token。

每种策略有不同的取舍。下采样丢失时间细节。池化丢失空间细节。Q-former 两者都少量损失但节省 token。

时间位置编码是另一个维度：模型如何知道第5帧在第6帧之前？选项包括简单的一维时间 RoPE（Video-LLaMA）、可学习的时间嵌入（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，完整3D）。

## 核心概念

### Video-LLaMA：Q-former 逐片段 + 音频分支

Video-LLaMA（2023）是首个开源视频大语言模型。架构：

- 16帧片段，2 FPS（即8秒）。
- 逐帧 ViT 特征 -> Video Q-former 对所有16帧做交叉注意力 -> 32个可学习查询 -> LLM。
- 并行音频分支：波形 -> ImageBind 音频编码器 -> Audio Q-former -> 32个查询 -> LLM。

优势：音视觉联合推理。劣势：固定片段长度，无法做任意时间点定位。

### VideoChat 与 Video-LLaVA

VideoChat 保留了 Video-LLaMA 的思路，但去掉了音频分支并做了简化。Video-LLaVA（Lin et al., 2023）在图像和视频帧上训练单一视觉编码器（"先对齐再投影"），获得统一表示。两者都是冻结 CLIP 编码器 + MLP + LLM。

两者都无法处理长视频。都是8-16帧系统。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入了 TMRoPE — 时间-模态旋转位置编码 (Temporal-Modality Rotary Position Embedding)。每个 patch token 携带 (t, h, w) 位置，其中 t 是实际时间戳（不是帧索引）。

与简单时间嵌入的关键差异：

- 绝对时间，非索引。模型看到的是"在4.2秒"而非"在第15帧"。
- 逐 Token 旋转，非逐片段。每个视觉 token 按其时间戳独立旋转。
- 兼容动态 FPS。如果某处以 2 FPS 采样、某处以 4 FPS 采样，TMRoPE 原生处理不均匀间隔。

TMRoPE 使"猫是在第几秒跳起来的？"这类查询成为可能。模型可以输出"在4.2秒"。Video-LLaMA 只能说"在片段早期"。

### 帧采样策略

均匀采样：在整段时长内均匀取 N 帧。简单，但会丢失运动峰值。

动态 FPS：根据运动强度自适应采样。光流或帧差法选取高运动片段做更密集采样。Qwen2.5-VL 基于此训练。

事件驱动：运行轻量检测器，在动作发生处采更多帧。VideoAgent 使用此方法。

关键帧 + 上下文：在镜头边界处采样 + 附近几帧。用于电影类内容。

### 逐帧池化

以 1 FPS 每帧576个 token 计算，5分钟片段为172,800个 token。Qwen2.5-VL-72B 的128k上下文可以处理，但代价高昂。

3x3 双线性池化将每帧降至64个 token -> 5分钟为19,200个 token。对大多数任务是最佳平衡点。

更激进的池化（6x6 -> 每帧16个 token）适用于空间细节不太重要的智能体工作流。

### 四个视频基准测试

- VideoMME：综合视频理解，短/中/长视频。
- TempCompass：细粒度时序推理，"之前"/"之后"类问题。
- EgoSchema：长时间第一人称视频。
- Video-MMMU：多模态多学科视频问答。

完整的视频 VLM 评估需覆盖全部四个。它们考察不同维度——TempCompass 专注时序排序，EgoSchema 考察3分钟以上的推理，VideoMME 跨越各种时长。

### 定位输出格式

时序定位的输出格式：

- 自由文本："猫大约在4秒时跳起来。"易于解析但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练此格式。
- 基于 Token：特殊的 `<time>4.1</time>` token 与答案交错。Qwen2.5-VL 的内部格式。

基于 Token 的格式对下游使用最准确。Qwen2.5-VL 的 JSON 输出格式可直接解析。

### 2026年最佳实践

2026年视频 VLM 最佳实践：

- 编码器：SigLIP 2 配合 M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- 帧采样：动态 FPS（1-4 取决于运动量）带最大帧数上限。
- 逐帧池化：3x3 双线性。
- 输出：带时间 + 事件字段的结构化 JSON。
- 基准测试：VideoMME + TempCompass 用于通用场景；EgoSchema 用于长时间推理。

## 动手实践

`code/main.py` 包含：

- 均匀和动态 FPS 帧采样器。
- 简易时序定位评估器：给定"真实"事件发生时间 T 和模型输出，在容差范围内评分准确率。
- Video-LLaMA（16帧，Q-former）、Video-LLaVA（8帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）的对比。

## 交付产出

本课产出 `outputs/skill-video-vlm-frame-planner.md`。给定一个视频任务（监控、动作识别、时序定位、摘要），它选择帧采样器、池化因子、输出格式，以及预期精度等级。

## 练习

1. 对于一段3分钟的烹饪演示，选择均匀采样还是动态 FPS？用 token 数量论证。

2. TMRoPE 相比简单的时间嵌入查找表，具体增加了什么能力？

3. 编写一个用于时序定位的 JSON schema，使 VLM 可以学习生成。包含错误情况。

4. 阅读 Video-LLaVA 第3节"先对齐再投影"。为什么这比分别训练图像和视频编码器更好？

5. 根据 VideoMME 排行榜，截至2026年最佳开源模型与最佳闭源模型的差距是多少？这个差距中多少归因于时间编码 vs 基础 LLM 规模？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 时序定位 (Temporal grounding) | "时间定位回答" | VLM 输出事件发生的具体时间戳范围 |
| TMRoPE | "时间-多模态 RoPE" | 带绝对时间戳的3D旋转位置编码，Qwen2.5-VL 使用 |
| 动态 FPS (Dynamic FPS) | "运动感知采样" | 在高运动片段采更多帧，静态片段采更少 |
| 帧池化 (Frame pooling) | "逐帧空间压缩" | 在送入 LLM 前用双线性插值减少每帧 patch 数 |
| Video Q-former | "片段压缩器" | 交叉注意力瓶颈，将 N 帧映射到 K 个可学习查询 |
| VideoMME | "视频基准测试" | 综合性短/中/长视频基准，2500+样本 |

## 延伸阅读

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
