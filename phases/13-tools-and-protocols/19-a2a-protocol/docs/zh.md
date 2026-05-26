# A2A — Agent-to-Agent 协议

> MCP 是 agent-to-tool。A2A (Agent2Agent) 是 agent-to-agent——一种开放协议，让基于不同框架构建的不透明 agent 相互协作。由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux Foundation，2026 年 4 月达到 v1.0，拥有 150+ 支持者，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP 并增加了 AP2 支付扩展。本课走读 Agent Card、Task 生命周期和两种传输绑定。

**类型：** 构建
**语言：** Python（标准库，Agent Card + Task 框架）
**前置课程：** Phase 13 · 06（MCP 基础），Phase 13 · 08（MCP 客户端）
**时长：** 约 75 分钟

## 学习目标

- 区分 agent-to-tool (MCP) 与 agent-to-agent (A2A) 的用例。
- 在 `/.well-known/agent.json` 发布带有技能和端点元数据的 Agent Card。
- 走通 Task 生命周期 (submitted → working → input-required → completed / failed / canceled / rejected)。
- 使用包含 Part (text, file, data) 的 Message 和作为输出的 Artifact。

## 问题背景

一个客服 agent 需要将报告撰写委派给一个专业的写作 agent。A2A 之前的选项：

- 自定义 REST API。可用，但每对 agent 都是一次性方案。
- 共享代码库。要求两个 agent 运行同一框架。
- MCP。不适合：MCP 用于调用工具，而非让两个 agent 在各自保持不透明内部推理的情况下协作。

A2A 填补了这一空白。它将交互建模为一个 agent 向另一个 agent 发送 Task，具有生命周期、消息和产出物。被调用 agent 的内部状态保持不透明——调用者只看到任务状态转换和最终输出。

A2A 是"让跨框架的 agent 相互对话"的协议。它不替代 MCP；二者互补。

## 核心概念

### Agent Card

每个符合 A2A 的 agent 在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现是基于 URL 的：拉取卡片，获取 A2A 端点 URL，枚举技能。

### 签名 Agent Card (AP2)

AP2 扩展（2025 年 9 月）为 Agent Card 增加了密码学签名。发布者用 JWT 签署自己的卡片；消费者验证。防止身份伪造。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working（通过消息循环）
```

客户端通过 `tasks/send` 发起。被调用 agent 在状态间转换；客户端通过 SSE 订阅状态更新或轮询。

### Message 与 Part

一条消息携带一个或多个 Part：

- `text` — 纯文本内容。
- `file` — 带 mimeType 的 base64 二进制数据。
- `data` — 类型化 JSON 负载（面向被调用 agent 的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifact

输出是 Artifact，而非原始字符串。Artifact 是命名的、类型化的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifact 可以以分块流式传输。调用者累积组装。

### 两种传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，POST 用于请求，可选 SSE 用于流式。默认绑定。
2. **gRPC。** 用于 gRPC 为原生协议的企业环境。

两种绑定承载相同的逻辑消息形态。

### 不透明性保持

一项关键设计原则：被调用 agent 的内部状态是不透明的。调用者看到任务状态和产出物。被调用 agent 的思维链、其工具调用、其子 agent 委派——全部不可见。这与 MCP 不同，在 MCP 中工具调用是透明的。

理由：A2A 使竞争者可以协作而不暴露内部实现。A2A 可以是"调用此客服 agent"，而调用者无需知道该 agent 如何实现服务。

### 时间线

- **2025-04-09。** Google 发布 A2A。
- **2025-06-23。** 捐赠给 Linux Foundation。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展 (Agent Payments) 发布。
- **2026-04。** v1.0 发布，150+ 支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|------|-----|-----|
| 用例 | Agent-to-tool | Agent-to-agent |
| 不透明性 | 透明的工具调用 | 不透明的内部推理 |
| 典型调用者 | Agent 运行时 | 另一个 agent |
| 状态 | 工具调用结果 | 带生命周期的 Task |
| 授权 | OAuth 2.1 (Phase 13 · 16) | JWT 签名的 Agent Card (AP2) |
| 传输 | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想调用一个具体工具时使用 MCP。当你想将整个任务委派给另一个 agent 时使用 A2A。许多生产系统二者并用：agent 将其工具层使用 MCP，将其协作层使用 A2A。

## 动手实践

`code/main.py` 实现了一个最小 A2A 框架：一个 research agent 发布其卡片，一个 writer agent 接收包含 PDF 和文本指令的 part 的 `tasks/send`，经历 working → input_required → working → completed 的状态转换，并返回文本 artifact。全部使用标准库；使用内存传输以聚焦消息形态。

关注点：

- Agent Card JSON 形态。
- Task id 分配和状态转换。
- 混合类型 part 的消息。
- 任务中途的 input-required 分支。
- 完成时返回 artifact。

## 交付产物

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个应能被其他 agent 调用的新 agent，该技能产出 Agent Card JSON、技能 schema 和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的 Task 生命周期，包括被调用 agent 请求澄清的 input-required 暂停。

2. 添加签名的 Agent Card。对卡片的标准 JSON 使用 HMAC 签名。编写验证器并确认其在卡片被篡改时失败。

3. 实现任务流式传输：writer agent 通过 SSE 发送三个增量 artifact 分块，调用者累积它们。

4. 设计一个包装 MCP 服务器的 A2A agent。将每个 MCP 工具映射到 A2A 技能。注意权衡——失去了哪些不透明性？

5. 阅读 A2A v1.0 公告，找出截至 2026 年 4 月尚未被任何框架实现的一项功能。（提示：与多跳任务委派有关。）

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| A2A | "Agent-to-Agent 协议" | 面向不透明 agent 协作的开放协议 |
| Agent Card | "`.well-known/agent.json`" | 描述 agent 技能和端点的发布元数据 |
| Skill | "可调用单元" | agent 支持的命名操作（类似 MCP tool） |
| Task | "委派单元" | 具有生命周期和最终产出物的任务项 |
| Message | "任务输入" | 携带 Part (text, file, data) |
| Part | "类型化数据块" | 消息中的 `text` / `file` / `data` 元素 |
| Artifact | "任务输出" | 完成时返回的命名、类型化输出 |
| AP2 | "Agent Payments Protocol" | 签名 Agent Card 扩展，用于信任和支付 |
| Opacity | "黑箱协作" | 被调用 agent 的内部细节对调用者隐藏 |
| Input-required | "任务暂停" | agent 需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A 规范正本
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月治理转移
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图和合作伙伴动力
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明和向后兼容性指导