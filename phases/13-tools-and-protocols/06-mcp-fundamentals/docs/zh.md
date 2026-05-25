# MCP 基础 — 原语、生命周期、JSON-RPC 底层

> MCP 出现之前，每个集成都是一次性的。Model Context Protocol（模型上下文协议）由 Anthropic 于 2024 年 11 月首次发布，现由 Linux 基金会旗下的 Agentic AI Foundation 托管，它标准化了发现和调用流程，使任意客户端都能与任意服务器通信。2025-11-25 规范定义了六个原语（服务器端三个、客户端三个）、三阶段生命周期，以及 JSON-RPC 2.0 线格式。掌握这些，本阶段 MCP 章节其余内容就只是阅读了。

**类型：** 学习  
**语言：** Python（stdlib，JSON-RPC 解析器）  
**前置条件：** Phase 13 · 01 至 05（工具接口与函数调用）  
**时长：** ～45 分钟

## 学习目标

- 列举全部六个 MCP 原语（服务器端：tools、resources、prompts；客户端：roots、sampling、elicitation），各给出一个使用场景。
- 走通三阶段生命周期（initialize、operation、shutdown），说明每阶段由谁发送哪条消息。
- 解析和生成 JSON-RPC 2.0 请求、响应与通知信封。
- 解释 `initialize` 阶段的能力协商是什么，以及缺少它会出什么问题。

## 问题

MCP 出现之前，每个使用工具的 Agent 都有自己的协议。Cursor 有一套与 MCP 形似但不兼容的工具系统。Claude Desktop 使用另一套。VS Code 的 Copilot 扩展又是第三套。一个团队做了一个"Postgres 查询"工具，就得针对三个不同宿主 API 写三遍。复用只能靠复制代码。

结果是一次性集成的寒武纪大爆发，以及生态系统速度的天花板。

MCP 通过标准化线格式解决了这一问题。一个 MCP 服务器可以在所有 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，截至 2026 年 4 月超过 300 个客户端。月 SDK 下载量 1.1 亿次。10,000+ 公开服务器。Linux 基金会于 2025 年 12 月在新成立的 Agentic AI Foundation 下接管了治理权。

本阶段使用的规范版本是 **2025-11-25**。该版本新增了异步 Tasks（SEP-1686）、URL 模式 elicitation（SEP-1036）、带工具的 sampling（SEP-1577）、增量 scope 同意（SEP-835）以及 OAuth 2.1 resource-indicator 语义。Phase 13 · 09 到 16 涵盖这些扩展。本课止步于基础。

## 概念

### 三个服务器端原语

1. **Tools（工具）。** 可调用的动作。与 Phase 13 · 01 中相同的四步循环。
2. **Resources（资源）。** 暴露的数据。通过 URI 寻址的只读内容：`file:///path`、`db://query/...`、自定义 scheme。
3. **Prompts（提示模板）。** 可复用的模板。宿主 UI 中的斜杠命令；服务器提供模板，客户端填充参数。

### 三个客户端原语

4. **Roots（根）。** 服务器被允许触及的 URI 集合。客户端声明，服务器遵守。
5. **Sampling（采样）。** 服务器请求客户端的模型执行一次补全。使服务器托管的 Agent 循环无需服务器端 API 密钥。
6. **Elicitation（征询）。** 服务器在执行过程中向客户端用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每项能力都恰好属于这六者之一。Phase 13 · 10 至 14 深入覆盖每一个。

### 线格式：JSON-RPC 2.0

每条消息都是一个 JSON 对象，包含以下字段：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` — 无 `id`，不期望响应。

基础规范约有 ～15 个方法，按原语分组。重要的包括：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：initialize。**

客户端发送 `initialize`，携带自己的 `capabilities` 和 `clientInfo`。服务器以自己的 `capabilities`、`serverInfo` 以及所支持的规范版本响应。客户端消化响应后发送 `notifications/initialized`。此后双方均可按协商好的能力发送请求。

**阶段 2：operation。**

双向通信。客户端调用 `tools/list` 进行发现，再调用 `tools/call` 执行调用。如果服务器声明了 sampling 能力，它可以发送 `sampling/createMessage`。工具集变化时服务器可发送 `notifications/tools/list_changed`。用户更改根范围时客户端可发送 `notifications/roots/list_changed`。

**阶段 3：shutdown。**

任一方关闭传输层。MCP 没有结构化的 shutdown 方法；传输层（stdio 或 Streamable HTTP，Phase 13 · 09）承载连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 就是契约。服务器端示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自己的能力来应答：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端未声明 `sampling`，服务器不得调用 `sampling/createMessage`。反之亦然：如果服务器未声明 `resources.subscribe`，客户端不得尝试订阅。

这就是防止生态漂移的机制。不支持 sampling 的客户端仍然是合法的 MCP 客户端；不调用 `sampling` 的服务器仍然是合法的 MCP 服务器。它们只是不在该功能上合作。

### 结构化内容与错误格式

`tools/call` 返回一个 `content` 数组，包含类型化块：`text`、`image`、`resource`。Phase 13 · 14 在此列表中增加了 MCP Apps（`ui://` 交互式 UI）。

错误使用 JSON-RPC 错误码。规范新增的错误码：`-32002`"Resource not found"、`-32603`"Internal error"，以及通过 `error.data` 承载的 MCP 特定错误数据。

### 客户端能力 vs 工具调用细节

一个常见混淆：`capabilities.tools` 表示的是客户端是否支持 tool-list-changed 通知。客户端是否会调用具体工具是模型在运行时做的选择，而非能力标志。能力标志是规范层面的契约。模型的选择与之正交。

### 为什么选 JSON-RPC 而非 REST？

JSON-RPC 2.0（2010）是一个轻量级双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（sampling、notifications），所以 JSON-RPC 凭借其对称的请求/响应形式天然契合。JSON-RPC 还能干净地组合在 stdio 和 WebSocket/Streamable HTTP 之上，无需重新发明 HTTP 的请求形式。

## 动手用

`code/main.py` 实现了一个最小的 JSON-RPC 2.0 解析器和生成器，然后手动走过 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真实传输层；只有消息格式。对照 Further Reading 中链接的规范来验证每个信封。

关注点：

- `initialize` 双向声明能力；响应包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回 `tools` 数组；每个条目有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应的 `content` 是 `{type, text}` 块的数组。

## 交付物

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一段 MCP 客户端-服务器交互的 pcap 风格记录，该技能标注每条消息属于哪个原语、哪个生命周期阶段，以及依赖哪项能力。

## 练习

1. 运行 `code/main.py`。找到能力协商发生的那行，描述如果服务器不声明 `tools.listChanged` 会有什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息格式：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在一个长时间运行的 `tools/call` 过程中发出它，确认客户端处理器能显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范 — 整个文档约 80 页。找出大多数服务器不需要的那一个能力标志。提示：与资源订阅相关。

4. 在纸上勾画一个假想的"定时任务"功能应归属哪个原语。（提示：服务器希望客户端在预定时间调用它。现有六个原语中没有合适的。）MCP 的 2026 路线图中有对此的 SEP 草案。

5. 解析 GitHub 上某个开放 MCP 服务器的一段会话日志。统计请求 vs 响应 vs 通知消息数。计算生命周期流量占总流量的比例。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| MCP | "Model Context Protocol" | 用于模型到工具发现与调用的开放协议 |
| 服务器原语 | "服务器暴露什么" | tools（动作）、resources（数据）、prompts（模板） |
| 客户端原语 | "客户端让服务器用什么" | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "线格式" | 对称的请求/响应/通知信封 |
| `initialize` 握手 | "能力协商" | 首个消息对；服务器和客户端声明各自支持的功能 |
| `tools/list` | "发现" | 客户端向服务器请求当前工具集 |
| `tools/call` | "调用" | 客户端请求服务器使用参数执行工具 |
| `notifications/*_changed` | "变更事件" | 服务器告诉客户端其原语列表已变化 |
| Content block | "类型化结果" | 工具结果中的 `{type: "text" | "image" | "resource" | "ui_resource"}` |
| SEP | "Spec Evolution Proposal" | 命名的规范演进提案（如 SEP-1686 用于异步 Tasks） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 权威规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布文章
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾与 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 摘要
