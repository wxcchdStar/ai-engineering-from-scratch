# 实战项目 12 — 视频理解管道（场景、问答、搜索）

> Twelve Labs 产品化了 Marengo + Pegasus。VideoDB 发布了视频 CRUD API。AI2 的 Molmo 2 发布了开放 VLM 检查点。Gemini 长上下文原生处理数小时的视频。TimeLens-100K 定义了规模化时间定位（Temporal Grounding）。2026 年的管道已经确定：场景分割、每场景字幕 + 嵌入、转录对齐、多向量索引，以及一个以（开始、结束）时间戳加帧预览来回答的查询。本实战项目要求摄入 100 小时视频，达到公开基准，并测量计数和动作类问题上的幻觉。

**类型：** 实战项目（Capstone）
**语言：** Python（管道）、TypeScript（UI）
**前置课程：** 第 4 阶段（计算机视觉）、第 6 阶段（语音）、第 7 阶段（Transformers）、第 11 阶段（LLM 工程）、第 12 阶段（多模态）、第 17 阶段（基础设施）
**涉及阶段：** P4 · P6 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

长视频问答是 2026 年规模下带宽最密集的多模态问题。Gemini 2.5 Pro 可以原生读取 2 小时的视频，但将 100 小时的视频摄入到可查询的语料库中仍然需要场景级索引。生产形态结合了场景分割（TransNetV2 或 PySceneDetect）、使用 VLM（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）的每场景字幕生成、转录对齐（Whisper-v3-turbo 带单词时间戳），以及一个将字幕、帧嵌入和转录并排存储的多向量索引。查询管道以（开始、结束）时间戳加帧预览来回答。

基准是公开的（ActivityNet-QA、NeXT-GQA）加上你自己的 100 个查询自定义集。计数和动作类型问题上的幻觉是已知的困难失败类别；本实战项目明确测量它。

## 概念

三个管道在摄入时并行运行。**场景分割**将视频切割为场景。**VLM 字幕生成**为每个场景生成字幕和关键帧的帧嵌入。**ASR 对齐**生成单词级时间戳。三个流按（scene_id, 时间范围）连接。每个场景在多向量索引（Qdrant）中获得三种向量类型：字幕嵌入、关键帧嵌入、转录嵌入。

在查询时，自然语言问题针对所有三个向量触发；结果使用 RRF 合并；时间定位适配器（TimeLens 风格）在顶部场景内细化（开始、结束）窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询 + 顶部场景 + 裁剪帧，并以引用时间戳和帧预览来回答。

幻觉测量很重要。计数（"有多少人进入房间？"）和动作类型（"厨师是在搅拌之前倒的吗？"）问题是出了名的不可靠。将准确率与描述性问题分开报告。

## 架构

```
视频文件 / URL
      |
      v
PySceneDetect / TransNetV2（场景分割）
      |
      +--- 每场景关键帧 --- VLM 字幕 + 帧嵌入
      |                     （Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2）
      |
      +--- 音频通道 --- Whisper-v3-turbo ASR + 单词时间戳
      |
      v
多向量 Qdrant：{caption_emb, keyframe_emb, transcript_emb}
      |
查询：
  针对所有三个向量的密集查询 -> RRF 合并 -> 前 k 个场景
      |
      v
TimeLens / VideoITG 时间定位（在场景内细化开始/结束）
      |
      v
VLM 合成：查询 + 顶部场景 + 帧预览
      |
      v
答案 +（开始、结束）时间戳 + 帧缩略图 + 引用
```

## 技术栈

- 场景分割：TransNetV2（2024-26 年最先进）或 PySceneDetect
- ASR：Whisper-v3-turbo 通过 faster-whisper，带单词时间戳
- VLM 字幕生成器 + 回答器：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时间定位：TimeLens-100K 训练的适配器或 VideoITG
- 索引：Qdrant 配合多向量支持（字幕/帧/转录）
- UI：Next.js 15 配合 HTML5 视频播放器和场景缩略图
- 评估：ActivityNet-QA、NeXT-GQA、自定义 100 个问题手工标注集
- 幻觉基准：计数和动作类型子集，手工标注

## 构建步骤

1. **摄入遍历器。** 接受 YouTube URL 或本地 MP4。如有需要，降采样到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect 生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k 个场景。

3. **ASR 阶段。** 对音频运行 Whisper-v3-turbo；导出单词级时间戳；分割为每场景转录切片。

4. **VLM 字幕生成。** 每场景，使用关键帧和简短字幕模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。生成字幕 + 帧嵌入。

5. **多向量索引。** Qdrant 集合，包含三个命名向量。负载：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题触发三个密集查询；使用倒数排名融合合并；前 k=5 个场景。

7. **时间定位。** 在顶部场景上运行 TimeLens 风格适配器，以在场景内细化（开始、结束）窗口。

8. **VLM 合成。** 使用查询 + 前 3 个场景剪辑（作为图像或短视频片段）+ 转录调用 Gemini 2.5 Pro。要求 `(video_id, start_ms, end_ms)` 引用。

9. **评估。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 个查询自定义集。报告总体准确率 + 每类细分（计数、动作、描述）。

## 使用方式

```
$ video-qa ask --url=https://youtube.com/watch?v=X "第一分钟有多少辆车通过路口？"
[场景]    检测到 23 个场景
[ASR]     转录完成，4 分 12 秒
[索引]    已写入 69 个向量（23 个场景 x 3）
[查询]    顶部场景：场景 3 [01:32-01:54]，置信度 0.84
[定位]    细化窗口：[00:12-00:58]
[合成]    gemini 2.5 pro，1.4 秒
答案：    在 00:12 到 00:58 之间，5 辆车通过路口。
引用：[场景 3：00:12-00:58]
      [帧预览：00:14、00:27、00:44、00:51、00:57]
```

## 交付标准

`outputs/skill-video-qa.md` 是交付物。给定一个 YouTube URL 或上传的视频，管道为场景建立索引，并以带时间戳的引用回答问题。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 时间定位 IoU | 在保留定位集上的交并比 |
| 20 | QA 准确率 | NeXT-GQA 和自定义 100 个查询 |
| 20 | 摄入吞吐量 | 每美元花费的视频小时数 |
| 20 | UI 和引用 UX | 时间戳链接、缩略图条、跳转到帧 |
| 15 | 幻觉率 | 分别统计计数和动作类型准确率 |
| **100** | | |

## 练习

1. 在字幕生成阶段将 Gemini 2.5 Pro 替换为 Qwen3-VL-Max。在人工评分的 50 个场景样本上报告字幕质量差异。
2. 将每场景帧嵌入减少为一个池化向量而非多向量。测量检索回退。
3. 构建"严格计数"模式：合成器提取每个计数实例及其时间戳，用户点击验证。测量用户验证是否减少幻觉。
4. 对摄入成本进行基准测试：三种 VLM 选择下的每美元视频小时数。选择最佳平衡点。
5. 添加说话人分类转录：对音频运行 pyannote 说话人分类，并嵌入每说话人转录。演示"Alice 对 X 说了什么？"查询。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| 场景分割 | "镜头检测" | 在镜头边界处将视频切割为场景 |
| 多向量索引 | "字幕 + 帧 + 转录" | Qdrant 集合，每个表示有命名向量 |
| 时间定位 | "它到底什么时候发生的" | 为查询答案细化（开始、结束）窗口 |
| 帧嵌入 | "视觉表示" | 关键帧的向量嵌入；用于场景视觉相似度 |
| RRF 融合 | "倒数排名融合" | 跨多个排名列表的合并策略；经典的混合检索技巧 |
| 计数幻觉 | "误数" | VLM 在"有多少 X"问题上的已知失败模式 |
| ActivityNet-QA | "视频 QA 基准" | 长视频 QA 准确率基准 |

## 扩展阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开放 VLM 检查点
- [TimeLens（CVPR 2026）](https://github.com/TencentARC/TimeLens) — 规模化时间定位
- [Gemini Video 长上下文](https://deepmind.google/technologies/gemini) — 托管参考
- [VideoDB](https://videodb.io) — 视频 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代方案
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估基准
