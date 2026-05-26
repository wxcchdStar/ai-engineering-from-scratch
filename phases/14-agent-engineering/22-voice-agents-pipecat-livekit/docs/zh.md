# 语音 Agent：Pipecat 与 LiveKit (Voice Agents: Pipecat and LiveKit)

> 语音 Agent 是 2026 年的一等生产类别。Pipecat 给你一个基于帧的 Python 流水线（VAD → STT → LLM → TTS → 传输）。LiveKit Agents 通过 WebRTC 将 AI 模型桥接到用户。生产延迟目标在高端技术栈上达到端到端 450–600ms。

**类型：** 学习 (Learn)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 01 (Agent 循环), Phase 14 · 12 (工作流模式)
**时间：** 约 60 分钟

## 学习目标

- 描述 Pipecat 基于帧的流水线：DOWNSTREAM（源→汇）和 UPSTREAM（控制）。
- 列举规范的语音流水线阶段以及 Pipecat 支持的传输方式。
- 解释 LiveKit Agents 的两种语音 Agent 类（MultimodalAgent、VoicePipelineAgent）及其各自适合的场景。
- 总结 2026 年生产延迟预期及其如何驱动架构选择。

## 问题

语音 Agent 不是文本循环加上 TTS。延迟预算极其紧张（约 600ms），部分音频是默认的，轮次检测是一个模型，传输范围从电话 SIP 到 WebRTC。要么你构建基于帧的流水线（Pipecat），要么你依赖平台（LiveKit）。

## 概念

### Pipecat (pipecat-ai/pipecat)

- Python 基于帧的流水线框架。
- `Frame` → `FrameProcessor` 链。
- 两个流方向：
  - **DOWNSTREAM** — 源 → 汇（音频输入，TTS 输出）。
  - **UPSTREAM** — 反馈和控制（取消、指标、打断）。
- `PipelineTask` 通过事件管理生命周期（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）和用于指标/追踪/RTVI 的观察者。

典型流水线：

```
VAD (Silero) → STT → LLM（上下文交替用户/助手） → TTS → 传输
```

传输：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 添加结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents (livekit/agents)

- 通过 WebRTC 将 AI 模型桥接到用户。
- 关键概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两种语音 Agent 类：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或等效方式的直接音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级控制。
- 通过 transformer 模型的语义轮次检测。
- 原生 MCP 集成。
- 通过 SIP 的电话。
- 50+ 模型无需 API 密钥，通过 LiveKit Inference；200+ 更多通过插件。

### 商业平台

Vapi（在优化的高端技术栈上约 450–600ms）和 Retell（在 180 次测试通话中端到端约 600ms）构建在这些之上。当你想要托管语音技术栈而不需要 WebRTC 团队时选择平台。

### 这个模式哪里会出错

- **没有打断处理。** 用户打断；Agent 继续说话。需要在 Pipecat 中使用 UPSTREAM 取消帧，LiveKit 中等效。
- **忽略 STT 置信度。** 低置信度转录被当作真理喂给 LLM。按置信度门控或请求确认。
- **TTS 句中截断。** 当流水线在话语中途取消时，TTS 需要知道或切断音频。
- **忽略延迟预算。** 每个组件增加 50–200ms。发布前求和你的链。

### 2026 年典型延迟

- VAD: 20–60ms
- STT 部分: 100–250ms
- LLM 首个 token: 150–400ms
- TTS 首个音频: 100–200ms
- 传输 RTT: 30–80ms

端到端 450–600ms 是高端。800–1200ms 是常见的。超过 1500ms 感觉像坏了。

## 构建

`code/main.py` 是一个基于帧的玩具流水线，包含：

- `Frame` 类型（audio、transcript、text、tts_audio、control）。
- `Processor` 接口，带 `process(frame)`。
- 五阶段流水线（VAD → STT → LLM → TTS → 传输）作为脚本化处理器。
- 一个 UPSTREAM 取消帧来演示打断。

运行：

```
python3 code/main.py
```

追踪显示正常流程和一个在话语中途停止 TTS 的打断取消。

## 使用

- **Pipecat** 用于完全控制——自定义处理器、Python 优先、可插拔提供商。
- **LiveKit Agents** 用于 WebRTC 优先的部署和电话。
- **Vapi / Retell** 用于托管语音 Agent，无需 WebRTC 团队。
- **OpenAI Realtime / Gemini Live** 用于直接音频输入/输出（MultimodalAgent）。

## 交付

`outputs/skill-voice-pipeline.md` 搭建一个 Pipecat 形态的语音流水线，包含 VAD + STT + LLM + TTS + 传输以及打断处理。

## 练习

1. 为你的玩具流水线添加指标观察者：每秒统计每个阶段的帧数。延迟在哪里累积？
2. 实现置信度门控 STT：低于阈值，请求"你能重复一下吗？"
3. 添加语义轮次检测：简单规则——如果转录以"？"结尾，轮次结束。
4. 阅读 Pipecat 的传输文档。将标准库传输替换为 SmallWebRTCTransport 配置（桩）。
5. 在相同查询上测量 OpenAI Realtime vs STT+LLM+TTS 级联。文本级控制带来了什么延迟成本？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Frame | "事件" | 流水线中类型化的数据单元（音频、转录、文本、控制） |
| Processor | "流水线阶段" | 带 process(frame) 的处理器 |
| DOWNSTREAM | "前向流" | 源到汇：音频输入，语音输出 |
| UPSTREAM | "反馈流" | 控制：取消、指标、打断 |
| VAD | "语音活动检测" | 检测用户何时在说话 |
| 语义轮次检测 (Semantic turn detection) | "智能轮次结束" | 基于模型的用户说完决策 |
| MultimodalAgent | "直接音频 Agent" | 音频输入，音频输出；中间无文本 |
| VoicePipelineAgent | "级联 Agent" | STT + LLM + TTS；文本级控制 |

## 扩展阅读

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的流水线、处理器、传输
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + 语音原语
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，延迟基准测试
