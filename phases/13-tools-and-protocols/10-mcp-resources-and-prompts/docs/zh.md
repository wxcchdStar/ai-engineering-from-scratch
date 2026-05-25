# MCP Resources 与 Prompts — 超越工具的上下文暴露

> Tools 获得了 MCP 90% 的关注。另外两个服务器原语解决不同的问题。Resources 暴露用于读取的数据；Prompts 将可复用模板暴露为斜杠命令。许多服务器应该使用 resources 而非将读取包装在 tools 中，使用 prompts 而非在客户端 prompt 中硬编码工作流。本课阐明决策规则并走通 `resources/*` 和 `prompts/*` 消息。

**类型：** 构建  
**语言：** Python（stdlib，resource + prompt 处理器）  
**前置条件：** Phase 13 · 07（MCP 服务器）  
**时长：** ～45 分钟

## 学习目标

- 针对给定领域，决定将一个能力暴露为 tool、resource 还是 prompt。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现 `prompts/list` 和 `prompts/get`，支持参数模板。
- 识别宿主何时将 prompts 呈现为斜杠命令 vs 自动注入的上下文。

## 问题

一个简单化的笔记应用 MCP 服务器将所有东西都暴露为 tools：`notes_read`、`notes_list`、`notes_search`。这将每次数据访问都包装在一个模型驱动的工具调用中。后果：

- 模型需要在每个可能受益于上下文的查询中决定是否调用 `notes_read`。
- 只读内容无法被订阅或流式传输到宿主的侧面板。
- 客户端 UI（Claude Desktop 的资源附加面板、Cursor 的"Include file"选择器）无法呈现数据。

正确的拆分：将数据暴露为 resource，将变更或计算动作暴露为 tool，将可复用的多步工作流暴露为 prompt。每个原语都有其 UX 承载方式和访问模式。

## 概念

### Tools vs resources vs prompts — 决策规则

| 能力 | 原语 |
|------|------|
| 用户想搜索、过滤或转换数据 | tool |
| 用户想让宿主将此数据作为上下文包含 | resource |
| 用户想要一个可重复运行的模板化工作流 | prompt |

指导原则：如果模型在每个相关查询中调用它都有好处，那它是 tool。如果用户将它附加到对话中有好处，那它是 resource。如果一个完整的多步工作流是用户想复用的单元，那它是 prompt。

### Resources

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的内容：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义 scheme）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 同时支持文本和二进制。二进制使用 `blob` 作为 base64 编码字符串加 `mimeType`。

### Resource 订阅

在 capabilities 中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。资源变化时服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

使用场景：一个资源是磁盘文件的笔记服务器；文件监视器触发更新通知；Claude Desktop 在文件被外部编辑时重新拉取到上下文中。

### Resource 模板（2025-11-25 新增）

`resourceTemplates` 让你暴露参数化的 URI 模式：`notes://{id}`，其中 `id` 作为补全目标。客户端可在资源选择器中自动补全 id。

### Prompts

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

Prompt 是一个模板，填充后得到宿主投喂给模型的消息列表。例如，一个 `code_review` prompt 接收 `file_path` 参数，返回一个三消息序列：一条系统消息、一条包含文件内容的用户消息、以及一条带推理模板的助手开头消息。

### 宿主与 prompts

Claude Desktop、VS Code 和 Cursor 将 prompts 暴露为聊天 UI 中的斜杠命令。用户输入 `/code_review` 并从表单中选择参数。服务器的 prompt 是"用户快捷方式"与"发送给模型的完整 prompt"之间的契约。

并非每个客户端都支持 prompts — 检查能力协商。声明了 prompt 能力但客户端不支持 prompt 的服务器，其斜杠命令就不会被看到。

### "list changed" 通知

resources 和 prompts 在集合变化时都会发出 `notifications/list_changed`。一个刚导入了 20 条新笔记的笔记服务器发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 以获取新增项。

### Content type 约定

对于文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
对于二进制：`image/png`、`application/pdf`，加 `blob` 字段。
对于 MCP Apps（Lesson 14）：`ui://` URI 中的 `text/html;profile=mcp-app`。

### 动态 resources

resource URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最新五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可自由动态计算内容。

规则：如果客户端可以按 URI 缓存，URI 必须是稳定的。如果计算是一次性的，URI 应包含时间戳或 nonce，以免客户端缓存过时。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获得服务器推送。订阅前的客户端或不支持订阅的宿主通过重新读取来轮询。两者都符合规范。服务器的能力声明告诉客户端它支持哪种。

订阅的成本：服务器上的每会话状态（谁订阅了什么）。保持订阅集有界；断开的客户端应超时清理。

### Prompts vs system prompts

MCP 中的 prompts 不是 system prompts。宿主的 system prompt（其自身的操作指令）和 MCP prompts（服务器提供的由用户调用的模板）并存。一个行为良好的客户端永远不会让服务器 prompt 覆盖其自身的 system prompt；它将它们分层叠加。

## 动手用

`code/main.py` 在 Lesson 07 的笔记服务器基础上扩展了：

- 每条笔记的 resources（`notes://note-1` 等），支持 `resources/subscribe`。
- 一个 `review_note` prompt，渲染为三消息模板。
- 模拟文件监视器，在笔记被修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` 动态 resource，始终返回最新五条笔记。

运行 demo 查看完整流程。

## 交付物

本课产出 `outputs/skill-primitive-splitter.md`。给定一个拟建的 MCP 服务器，该技能将每个能力分类为 tool / resource / prompt 并给出理由。

## 练习

1. 运行 `code/main.py`。观察初始 resource 列表，然后触发一次笔记编辑并验证 `notifications/resources/updated` 事件是否触发。

2. 添加一个 `resources/list_changed` 发射器：当创建新笔记时发送通知，让客户端重新发现。

3. 为 GitHub MCP 服务器设计三个 prompts：`summarize_pr`、`triage_issue`、`release_notes`。每个带参数 schema。Prompt body 应能无需进一步编辑即可运行。

4. 取 Lesson 07 服务器中的一个现有 tool，分类它应保持为 tool 还是应拆分为 resource + tool 对。用一句话论证。

5. 阅读规范的 `server/resources` 和 `server/prompts` 章节。找出 `resources/read` 中一个很少被填充但规范支持的字段。提示：查看 resource content 上的 `_meta`。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| Resource | "暴露的数据" | 宿主可读取的 URI 可寻址内容 |
| Resource URI | "数据指针" | Scheme 前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | "监视变化" | 客户端选择加入的针对特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | "资源已变化" | 通知客户端已订阅资源有新内容的信号 |
| Resource template | "参数化 URI" | 带补全提示的 URI 模式，用于宿主选择器 |
| Prompt | "斜杠命令模板" | 带参数槽的命名多消息模板 |
| Prompt arguments | "模板输入" | 宿主在渲染前收集的类型化参数 |
| `prompts/get` | "渲染模板" | 服务器返回填充好的消息列表 |
| Content block | "类型化块" | `{type: text | image | resource | ui_resource}` |
| 斜杠命令 UX | "用户快捷方式" | 宿主将 prompts 呈现为以 `/` 开头的命令 |

## 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — resource URI、订阅和模板
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — prompt 模板和斜杠命令集成
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整 `resources/*` 消息参考
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整 `prompts/*` 消息参考
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 扩展官方文档的社区指南
