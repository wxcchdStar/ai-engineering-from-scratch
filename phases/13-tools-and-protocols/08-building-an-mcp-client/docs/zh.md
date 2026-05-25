# 构建 MCP 客户端 — 发现、调用、会话管理

> 大多数 MCP 内容发布的是服务器教程，客户端部分一笔带过。客户端代码才是复杂编排所在：进程启动、能力协商、多服务器工具列表合并、sampling 回调、重连以及命名空间冲突解决。本课构建一个多服务器客户端，将三个不同的 MCP 服务器提升到一个扁平工具命名空间中供模型使用。

**类型：** 构建  
**语言：** Python（stdlib，多服务器 MCP 客户端）  
**前置条件：** Phase 13 · 07（构建 MCP 服务器）  
**时长：** ～75 分钟

## 学习目标

- 将 MCP 服务器作为子进程启动、完成 `initialize`、发送 `notifications/initialized`。
- 维护每服务器的会话状态（capabilities、工具列表、最后看到的通知 id）。
- 跨多个服务器合并工具列表到一个命名空间并处理冲突。
- 将工具调用路由到拥有该工具的服务器并组装响应。

## 问题

一个真实的 Agent 宿主（Claude Desktop、Cursor、Goose、Gemini CLI）同时加载多个 MCP 服务器。用户可能同时运行文件系统服务器、Postgres 服务器和 GitHub 服务器。客户端的任务：

1. 启动每个服务器。
2. 独立握手每一个。
3. 对每个调用 `tools/list` 并拍平结果。
4. 当模型发出 `notes_search` 时，在合并的命名空间中查找并路由到正确的服务器。
5. 处理来自任何服务器的通知（`tools/list_changed`）而不阻塞。
6. 传输层失败时重连。

手动实现所有这些就是"玩具"与"可用"之间的分水岭。官方 SDK 封装了这些，但心智模型必须是你自己的。

## 概念

### 子进程启动

`subprocess.Popen` 配合 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1` 并使用文本模式逐行读取。每个服务器是一个进程；客户端为每个服务器持有一个 `Popen` 句柄。

### 每服务器会话状态

每个服务器对应一个 `Session` 对象，持有：

- `process` — Popen 句柄。
- `capabilities` — 服务器在 `initialize` 时声明的能力。
- `tools` — 最后一次 `tools/list` 的结果。
- `pending` — 请求 id 到等待响应的 promise/future 的映射。

请求本质上是异步的；向服务器 A 发送 `tools/call` 时服务器 B 正在进行中调用不应阻塞。使用带队列的线程或 asyncio。

### 合并命名空间

当客户端看到聚合的工具列表时，名称可能冲突。两个服务器可能都暴露 `search`。客户端有三种选择：

1. **按服务器名前缀。** `notes/search`、`files/search`。清晰但不美观。
2. **静默先到先得。** 后来服务器的 `search` 覆盖先前的。有风险；隐藏了冲突。
3. **冲突拒绝。** 拒绝加载第二个服务器；通知用户。对安全敏感的宿主最安全。

Claude Desktop 使用按服务器名前缀。Cursor 使用冲突拒绝并给出明确错误。VS Code MCP 同样采用按服务器名前缀。

### 路由

合并后，分发表将 `tool_name -> session` 映射。模型按名称发出调用；客户端找到对应 session 并向该服务器的 stdin 写入 `tools/call` 消息，然后等待响应。

### Sampling 回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可能发送 `sampling/createMessage` 请求客户端运行其 LLM。客户端必须：

1. 在 sample 解决前阻止对该服务器的进一步请求，或在实现支持并发时流水线化。
2. 调用其 LLM 提供商。
3. 将响应发回服务器。

Lesson 11 端到端地涵盖 sampling。本课仅做存根处理。

### 通知处理

`notifications/tools/list_changed` 意味着需要重新调用 `tools/list`。`notifications/resources/updated` 意味着如果该资源正在使用则需要重新读取。通知不得产生响应 — 不要尝试确认它们。

一个常见的客户端 bug：在 `tools/call` 上阻塞读取循环，而通知在流中等待。使用后台读取线程将每条消息推入队列；主线程出队并分发。

### 重连

传输层可能失败：服务器崩溃、OS 杀死进程、stdio 管道断裂。客户端在 stdout 上检测到 EOF 并将会话标记为已死。选项：

- 静默重启服务器并重新握手。适用于纯只读服务器。
- 向用户呈现失败。适用于有用户可见会话的有状态服务器。

Phase 13 · 09 涵盖 Streamable HTTP 的重连语义；stdio 更简单。

### Keepalive 与 session id

Streamable HTTP 使用 `Mcp-Session-Id` 头。Stdio 没有 session id — 进程标识就是会话。Keepalive ping 是可选的；stdio 管道在无活动时不会断开。

## 动手用

`code/main.py` 将三个模拟 MCP 服务器作为子进程启动、握手每一个、合并其工具列表，并将工具调用路由到正确的服务器。这些"服务器"实际上是运行玩具响应器的 Python 进程（没有真实 LLM）。运行它可以看到：

- 三次初始化，每个有自己的能力集。
- 三个 `tools/list` 结果合并成一个 7 工具命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀阻止的冲突。

关注点：

- `Session` dataclass 清晰地保存每服务器状态。
- 后台读取线程在不阻塞主线程的情况下排空 stdout 上的每一行。
- 分发表是一个简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个服务器声明相同名称时，后者被重命名加前缀。

## 交付物

本课产出 `outputs/skill-mcp-client-harness.md`。给定一个声明式的 MCP 服务器列表（名称、命令、参数），该技能生成一个工具，启动它们、合并工具列表，并提供带冲突解决的路由函数。

## 练习

1. 运行 `code/main.py` 并观察服务器启动日志。用 SIGTERM 杀死其中一个模拟服务器进程，观察客户端如何检测到 EOF 并将该会话标记为已死。

2. 实现命名空间前缀。当两个服务器暴露 `search` 时，将第二个重命名为 `<server>/search`。更新分发表并验证工具调用能正确路由。

3. 添加连接池式的退避重启：连续失败时指数退避，上限 30 秒，三次失败后向用户发出通知。

4. 设计一个支持 100 个并发 MCP 服务器的客户端。什么数据结构替代简单的分发 dict？（提示：用于前缀命名空间的 trie，加上每服务器工具数的指标。）

5. 将客户端移植到官方 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。代码应从 ～200 行缩减到 ～40 行，同时保持多服务器路由。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| MCP client | "Agent 宿主" | 启动服务器并编排工具调用的进程 |
| Session | "每服务器状态" | Capabilities、工具列表和待处理请求的簿记 |
| 合并命名空间 | "一个工具列表" | 跨所有活跃服务器的扁平工具名集合 |
| 命名空间冲突 | "两个服务器同名工具" | 客户端必须前缀、拒绝或先到先得处理重复 |
| 路由 | "这个调用归谁？" | 从工具名到拥有者服务器的分发 |
| 后台读取器 | "非阻塞 stdout" | 将服务器 stdout 排入队列的线程或任务 |
| Sampling 回调 | "LLM 即服务" | 客户端处理来自服务器的 `sampling/createMessage` 的处理器 |
| `notifications/*_changed` | "原语已变更" | 客户端必须重新发现或重新读取的信号 |
| 重连策略 | "服务器挂了怎么办" | 传输层失败时的重启语义 |
| Stdio session | "进程 = 会话" | 无 session id；子进程生命周期即是会话 |

## 延伸阅读

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — 权威客户端行为规范
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world 客户端教程
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — `ClientSession` 和 `stdio_client` 参考
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TS 对等实现
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个编辑器宿主中多路复用多个 MCP 服务器
