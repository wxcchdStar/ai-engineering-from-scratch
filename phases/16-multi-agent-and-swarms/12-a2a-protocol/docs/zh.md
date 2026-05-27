# A2A——智能体到智能体协议（Agent-to-Agent Protocol）

> Google 于 2025 年 4 月宣布 A2A；到 2026 年 4 月，规范位于 https://a2a-protocol.org/latest/specification/，150+ 组织支持它。A2A 是 MCP（第 13 课）的水平补充：MCP 是垂直的（智能体 ↔ 工具），A2A 是对等的（智能体 ↔ 智能体）。它定义了 Agent Card（智能体卡片，用于发现）、带有工件（artifacts，文本、结构化数据、视频）的任务（tasks）、不透明的任务生命周期（opaque task lifecycles）和认证（auth）。生产系统越来越多地将 MCP 与 A2A 配对。Google Cloud 在 2025-2026 年间将 A2A 支持推广到 Vertex AI Agent Builder。

**类型：** 学习 + 构建
**语言：** Python（标准库，`http.server`，`json`）
**前置条件：** 第 16 阶段 · 04（原语模型）
**时间：** ~75 分钟

## 问题

你的智能体需要调用另一个系统上的另一个智能体。怎么做？你可以暴露一个 HTTP 端点，定义一个定制的 JSON 模式，并希望对方能理解。每对智能体都变成一个自定义集成。

A2A 是该调用的通用线路协议。标准发现、标准任务模型、标准传输、标准工件。就像 HTTP+REST，但智能体是一等公民。

## 概念

### 四个元素

**Agent Card（智能体卡片）。** 位于 `/.well-known/agent.json` 的 JSON 文档，描述智能体：名称、技能、端点、支持的模态、认证要求。通过读取卡片进行发现。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task（任务）。** 工作单元。一个异步、有状态的对象，具有生命周期：`submitted → working → completed / failed / canceled`。客户端发送任务，轮询或订阅更新。

**Artifact（工件）。** 任务产生的结果类型。文本、结构化 JSON、图像、视频、音频。工件是有类型的，因此不同的模态是一等公民。

**不透明生命周期（Opaque lifecycle）。** A2A 不规定远程智能体*如何*解决任务。客户端看到状态转换和工件；实现可以自由使用任何框架。

### MCP/A2A 分工

-   **MCP**（第 13 课）：智能体 ↔ 工具。智能体通过 JSON-RPC 向工具服务器读写。默认无状态。
-   **A2A**：智能体 ↔ 智能体。对等协议；双方都是具有自己推理能力的智能体。

生产级多智能体系统同时使用两者。A2A 对等方在其一侧调用 MCP 工具。这种分工保持了两个关注点的清晰。

### 发现流程

```
客户端                     智能体服务器
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或使用流式传输：SSE 订阅 `/tasks/{id}/events` 以获取推送更新。

### 认证

A2A 支持三种常见模式：

-   **Bearer token**——OAuth2 或不透明令牌。
-   **mTLS**——双向 TLS；组织相互证明身份。
-   **签名请求**——对负载进行 HMAC 签名。

认证在 Agent Card 中声明；客户端发现并遵守。

### 到 2026 年 4 月已有 150+ 组织

企业采用推动了 A2A 的规模。头条新闻：A2A 成为企业智能体系统跨越信任边界的方式。Google Cloud 发布了 Vertex AI Agent Builder A2A 支持；Microsoft Agent Framework 支持它；大多数主要框架（LangGraph、CrewAI、AutoGen）都发布了 A2A 适配器。

### A2A 在哪里胜出

-   **跨组织调用。** 公司 A 的智能体调用公司 B 的智能体。没有 A2A，每对都是一个定制合同。
-   **异构框架。** LangGraph 智能体调用 CrewAI 智能体调用自定义 Python 智能体。A2A 规范化。
-   **类型化工件。** 视频结果、结构化 JSON、音频——全都是一等公民。
-   **长时间运行的任务。** 不透明生命周期 + 轮询使数小时的任务变得简单。

### A2A 在哪里困难

-   **延迟敏感的微调用。** A2A 的生命周期是异步的。亚毫秒级的智能体到智能体不适合；使用直接 RPC。
-   **紧耦合进程内智能体。** 如果两个智能体在同一 Python 进程中运行，A2A 的 HTTP 往返是多余的。
-   **小团队。** 规范开销是真实的；仅内部智能体可能不需要这种形式。

### A2A vs ACP、ANP、NLIP

2024-2026 年间出现了几个相关规范：

-   **ACP**（IBM/Linux Foundation）——A2A 的前身，范围更窄。
-   **ANP**（Agent Network Protocol）——以对等发现为主，去中心化优先。
-   **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）——自然语言内容类型。

截至 2026 年 4 月，A2A 是采用最广泛的对等协议。参见 arXiv:2505.02279（Liu 等人，"A Survey of Agent Interoperability Protocols"）进行比较。

## 构建它

`code/main.py` 使用 `http.server` 和 JSON 实现了一个 A2A 最小化服务器和客户端。服务器：

-   暴露 `/.well-known/agent.json`，
-   接受 `POST /tasks`，
-   管理任务状态，
-   在 `GET /tasks/{id}` 上返回工件。

客户端：

-   获取 Agent Card，
-   提交任务，
-   轮询直到完成，
-   读取工件。

运行：

```
python3 code/main.py
```

脚本在后台线程中启动服务器，然后对其运行客户端。你看到完整的流程：发现、提交、轮询、工件。

## 使用它

`outputs/skill-a2a-integrator.md` 设计 A2A 集成：Agent Card 内容、任务模式、认证选择、流式传输 vs 轮询。

## 交付它

清单：

-   **固定规范版本。** A2A 仍在演进；Agent Card 应声明协议版本。
-   **幂等任务创建。** 重复提交（网络重试）应产生一个任务。
-   **工件模式。** 声明智能体返回什么形状；消费者应验证。
-   **速率限制 + 认证。** A2A 是面向公众的；应用标准 Web 安全。
-   **失败任务的死信队列。** 随时间检查模式以发现重复出现的故障类型。

## 练习

1.  运行 `code/main.py`。确认客户端发现服务器并接收正确的工件。
2.  向服务器添加第二个技能（例如，"summarize"）。更新 Agent Card。编写一个根据任务类型选择技能的客户端。
3.  实现 SSE 流式端点：`/tasks/{id}/events` 发出状态变化。客户端需要做什么不同的事情？
4.  阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。识别规范强制要求但此演示未实现的三件事。
5.  比较 A2A（Agent Card 发现）与 MCP（通过 `listTools` 进行服务器端能力列表）。自描述智能体与能力探测之间的权衡是什么？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| A2A | "智能体到智能体" | 智能体跨系统调用其他智能体的对等协议。Google 2025。 |
| Agent Card | "智能体的名片" | 位于 `/.well-known/agent.json` 的 JSON，描述技能、端点、认证。 |
| Task | "工作单元" | 具有生命周期的异步有状态对象；完成时产生工件。 |
| Artifact | "结果" | 类型化输出：文本、结构化 JSON、图像、视频、音频。一等媒体。 |
| 不透明生命周期 | "如何解决是智能体的事" | 客户端看到状态转换；服务器可以自由选择框架/工具。 |
| 发现 | "找到智能体" | `GET /.well-known/agent.json` 返回卡片。 |
| MCP vs A2A | "工具 vs 对等方" | MCP：垂直智能体 ↔ 工具。A2A：水平智能体 ↔ 智能体。 |
| ACP / ANP / NLIP | "兄弟协议" | 相邻规范；A2A 是 2026 年采用最广泛的。 |

## 扩展阅读

-   [A2A 规范](https://a2a-protocol.org/latest/specification/) — 权威规范
-   [Google Developers Blog — A2A 公告](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布文章
-   [A2A GitHub 仓库](https://github.com/a2aproject/A2A) — 参考实现和 SDK
-   [Liu 等人 — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 比较
