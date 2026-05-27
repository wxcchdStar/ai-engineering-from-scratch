# 实战项目 03 — 实时语音助手

> 2026 年，OpenAI 的实时 API、Gemini Live、Cartesia Sonic-2、ElevenLabs 和 LiveKit Agents 1.0 将语音延迟推到了 200 毫秒以下。其形态是：LiveKit 用于 WebRTC 传输，Whisper-v3-turbo 用于 ASR（自动语音识别），Claude Sonnet 4.7 或 GPT-5.4 用于推理，Cartesia Sonic-2 用于 TTS（文本转语音），支持打断（barge-in），以及一个函数调用层用于工具使用。本实战项目是构建一个语音助手，在 10 个真实任务上进行基准测试，并发布一份与 OpenAI 实时 API 的并排对比报告。

**类型：** 实战项目（Capstone）
**语言：** Python（后端）、TypeScript（Web 客户端）
**前置课程：** 第 6 阶段（语音）、第 11 阶段（LLM 工程）、第 12 阶段（多模态）、第 17 阶段（基础设施）
**涉及阶段：** P6 · P11 · P12 · P17
**时间：** 25 小时

## 问题

2026 年，语音 AI 跨越了消费级产品的门槛。OpenAI 的实时 API 为 ChatGPT 语音模式提供支持。Gemini Live 为 Google Assistant 提供支持。Cartesia Sonic-2 在 80 毫秒内生成语音。ElevenLabs 覆盖了 32 种语言。LiveKit Agents 1.0 将整个管道打包成一个自托管的 WebRTC 工作流。其形态已趋于稳定：WebRTC 传输、流式 ASR、LLM 推理、流式 TTS、打断处理以及用于工具调用的函数调用层。

困难之处不在于单个模型，而在于全栈延迟（麦克风到扬声器必须在 500 毫秒以内才能让人感觉自然）、打断处理（用户在助手还在说话时就开始说话）以及中断恢复（网络切换、背景噪音）。本实战项目要求你构建整个管道，在 10 个真实任务上进行基准测试，并发布一份与 OpenAI 实时 API 的并排对比报告。

## 概念

该管道有五个阶段。**音频输入**：LiveKit 通过 WebRTC 捕获音频，应用回声消除和降噪。**ASR（自动语音识别）**：Whisper-v3-turbo 将音频流式转录为文本，带单词级时间戳。**推理**：Claude Sonnet 4.7 或 GPT-5.4 接收转录文本，推理并生成响应文本。**函数调用**：LLM 可以调用工具（搜索网页、查询数据库、控制智能家居设备）。**TTS（文本转语音）**：Cartesia Sonic-2 将响应文本流式转换为语音，目标首音节延迟低于 200 毫秒。

打断处理是关键的工程挑战。当用户在助手说话时开始说话时，系统必须：停止 TTS 流、丢弃部分转录文本、处理新输入。LiveKit Agents 1.0 原生处理这一问题，通过 `AgentSession` 管理打断和回合切换。

## 架构

```
用户音频（WebRTC）
      |
      v
LiveKit 服务器（回声消除、降噪）
      |
      v
Whisper-v3-turbo ASR（流式，单词级时间戳）
      |
      v
Claude Sonnet 4.7 / GPT-5.4（推理 + 函数调用）
      |
      +-- 工具：搜索、数据库、设备控制
      |
      v
Cartesia Sonic-2 TTS（流式，首音节 < 200 毫秒）
      |
      v
用户音频输出（WebRTC）
      |
      v
打断处理（LiveKit AgentSession 回合管理）
```

## 技术栈

- 传输：LiveKit WebRTC（自托管或 LiveKit Cloud）
- ASR：Whisper-v3-turbo 通过 faster-whisper，带单词级时间戳
- 推理：Claude Sonnet 4.7 或 GPT-5.4，配合提示缓存
- TTS：Cartesia Sonic-2（流式，多语言）
- 函数调用：通过 MCP 工具服务器的 FastMCP
- 打断：LiveKit Agents 1.0 AgentSession 回合管理
- 监控：Langfuse 配合每回合追踪
- 客户端：Next.js 15 Web 应用，配合 LiveKit 客户端 SDK

## 构建步骤

1. **LiveKit 设置。** 部署 LiveKit 服务器（自托管 Docker 或 LiveKit Cloud）。配置 WebRTC 并启用回声消除。

2. **ASR 工作器。** 将 Whisper-v3-turbo 挂载到 LiveKit 音频轨道上。流式输出带单词级时间戳的转录文本。处理静音检测以进行回合结束判定。

3. **推理工作器。** 接收转录文本，运行 Claude Sonnet 4.7 配合系统提示。支持函数调用以使用工具。将响应文本流式传输到 TTS。

4. **TTS 工作器。** Cartesia Sonic-2 接收文本流并生成语音。目标首音节延迟低于 200 毫秒。支持多语言语音。

5. **打断处理。** LiveKit AgentSession 管理回合状态。当用户在助手说话时开始说话时：停止 TTS、丢弃转录文本、处理新输入。

6. **函数调用。** 通过 FastMCP 暴露工具：网络搜索、数据库查询、日历访问。LLM 可以实时调用这些工具。

7. **中断恢复。** 处理网络切换（WiFi 到蜂窝网络）。在连接中断时保持会话状态。在重新连接时恢复。

8. **延迟基准测试。** 测量：麦克风到转录文本延迟、转录文本到 LLM 首 token 延迟、LLM 到 TTS 首音节延迟、端到端麦克风到扬声器延迟。

9. **任务基准测试。** 10 个真实任务：设置提醒、回答事实性问题、控制智能家居设备、预订餐厅、发送消息等。测量任务完成率和用户满意度。

## 使用方式

```
用户："旧金山今天天气怎么样？"
[ASR]     "what's the weather in San Francisco today?" 耗时 180 毫秒
[LLM]     函数调用：weather_api.get("San Francisco") 耗时 320 毫秒
[LLM]     "It's 68°F and partly cloudy in San Francisco today."
[TTS]     首音节 140 毫秒，完整语音 1.2 秒
[总延迟]  麦克风到扬声器 640 毫秒
```

## 交付标准

`outputs/skill-voice-assistant.md` 描述交付物。一个实时语音助手，附有延迟基准测试和与 OpenAI 实时 API 的并排对比报告。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 麦克风到扬声器 p50/p95，目标 < 500 毫秒 |
| 20 | 打断处理 | 打断恢复延迟，回合切换准确性 |
| 20 | 任务完成率 | 10 个真实任务，完成率 |
| 20 | 函数调用 | 工具调用延迟和准确性 |
| 15 | 中断恢复 | 网络切换恢复时间 |
| **100** | | |

## 练习

1. 将 TTS 引擎从 Cartesia Sonic-2 替换为 ElevenLabs。测量首音节延迟和自然度的差异。
2. 添加说话人分类：当多人同时说话时，为每个说话人标注转录文本。测量分类准确性。
3. 实现"思考中"音频提示：当 LLM 推理时间超过 500 毫秒时，播放微妙的音频提示，让用户知道系统仍在处理。
4. 在嘈杂环境（咖啡馆、汽车）中对 ASR 进行基准测试。测量词错误率（WER）的下降程度。
5. 添加语言切换：用户用西班牙语提问，助手用西班牙语回答。测量切换延迟。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| WebRTC | "实时通信" | 用于音频/视频流的浏览器 API，带回声消除 |
| ASR | "语音转文本" | 自动语音识别——将音频转录为文本 |
| TTS | "文本转语音" | 将文本合成为语音音频 |
| 打断 | "中断处理" | 用户在助手说话时开始说话；系统必须停止并切换 |
| 首音节延迟 | "首个音频时间" | 从文本到第一个可听声音的时间；目标 < 200 毫秒 |
| 回合管理 | "谁在说话" | 跟踪对话回合状态：用户回合 vs 助手回合 |
| 函数调用 | "工具使用" | LLM 实时调用外部 API 的能力 |

## 扩展阅读

- [OpenAI 实时 API](https://platform.openai.com/docs/guides/realtime) — 参考生产形态
- [Gemini Live](https://deepmind.google/technologies/gemini) — 备选参考
- [Cartesia Sonic-2](https://cartesia.ai) — 参考 TTS 引擎
- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考语音智能体框架
- [Whisper-v3-turbo](https://github.com/openai/whisper) — 参考 ASR 模型
- [ElevenLabs](https://elevenlabs.io) — 备选 TTS 引擎
- [FastMCP](https://github.com/jlowin/fastmcp) — 工具服务器框架
