# 异步 Tasks（SEP-1686）— 先调用后获取的长时间运行工作

> 真实的 Agent 工作需要数分钟到数小时：CI 运行、深度研究合成、批量导出。同步工具调用会断连、超时或阻塞 UI。SEP-1686（合并于 2025-11-25）添加了 Tasks 原语：任何请求都可以被增强为 task，结果可以稍后获取或通过状态通知流式推送。漂移风险提示：Tasks 在 2026 H1 之前是实验性的；SDK 表面仍在围绕规范进行设计。

**类型：** 构建  
**语言：** Python（stdlib，异步 task 状态机）  
**前置条件：** Phase 13 · 07（MCP 服务器）、Phase 13 · 09（传输层）  
**时长：** ～75 分钟

## 学习目标

- 识别何时将工具从同步提升为 task 增强（> 30 秒的服务器端工作）。
- 走通 task 生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化 task 状态使崩溃不会丢失正在进行的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题

一个 `generate_report` 工具运行多分钟的提取管道。同步模型下的选项：

1. 保持连接打开三分钟。远程传输层会断开；客户端超时；UI 冻结。
2. 立即返回占位符；要求客户端轮询自定义端点。破坏了 MCP 的统一性。
3. 发后即忘；无结果。

都不好。SEP-1686 添加了第四种：task 增强。任何请求（通常是 `tools/call`）都可以被标记为 task。服务器立即返回 task id。客户端轮询 `tasks/status` 并在完成时获取 `tasks/result`。服务器端状态可在重启后存活。

## 概念

### Task 增强

请求通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定）变为 task。服务器立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器承诺保留状态的时间；ttl 之后 task 结果被丢弃。

### 每工具的选择加入

工具标注可声明 task 支持：

- `taskSupport: "forbidden"` — 此工具始终同步运行。适用于快速工具。
- `taskSupport: "optional"` — 客户端可请求 task 增强。
- `taskSupport: "required"` — 客户端必须使用 task 增强。

`generate_report` 工具应为 `required`。`notes_search` 工具应为 `forbidden`。

### 状态

```
working  -> input_required -> working  (通过 elicitation 循环)
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是只追加的：一旦 `completed`、`failed` 或 `cancelled`，task 即为终态。

### 方法

- `tasks/status {taskId}` — 返回当前状态和进度提示。
- `tasks/result {taskId}` — 阻塞或在未完成时返回 404。
- `tasks/cancel {taskId}` — 幂等；终态忽略。
- `tasks/list` — 可选；枚举活跃和最近完成的 tasks。

### 流式状态变更

当服务器支持时，客户端可订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

使用流式而非轮询的客户端获得更好的 UX。轮询始终被支持作为最小表面。

### 持久状态

规范要求声明 task 支持的服务器持久化状态。崩溃不应在 ttl 内丢失已完成的结果。存储从 SQLite 到 Redis 到文件系统。Lesson 13 的工具使用文件系统。

### 取消语义

`tasks/cancel` 是幂等的。如果 task 正在执行中，服务器尝试停止（检查执行器协作式取消）。如果已是终态，请求为空操作。

### 崩溃恢复

当服务器进程重启时：

1. 加载所有持久化的 task 状态。
2. 将进程已死的任何 `working` tasks 标记为 `failed`，错误为 `CRASH_RECOVERY`。
3. 在 ttl 内保留 `completed` / `failed` / `cancelled`。

### 异步 tasks 加 sampling

Task 自身可以调用 `sampling/createMessage`。这是长时间运行的研究 task 的工作方式：服务器的 task 线程按需 sample 客户端的模型，同时客户端 UI 显示 task 为 `working` 并定期更新进度。

### 为什么这是实验性的

SEP-1686 在 2025-11-25 发布，但更广泛的路线图指出三个开放问题：持久订阅原语、子任务（父子 task 关系）和 result-TTL 标准化。预计规范将在 2026 年继续演进。生产代码应仅对常见情况将 Tasks 视为稳定，并针对子任务的未来 SDK 变更做好防护。

## 动手用

`code/main.py` 实现了一个持久 task 存储（文件系统支持）和一个在后台线程运行的 `generate_report` 工具。客户端调用工具、立即获得 task id、在 worker 更新进度时轮询 `tasks/status`、完成时获取 `tasks/result`。取消有效；崩溃恢复通过杀死 worker 线程并重新加载状态来模拟。

关注点：

- Task 状态 JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- Worker 线程更新 `progress` 字段；轮询显示其前进。
- 客户端发起的取消设置一个事件；worker 检查并提前退出。
- "崩溃"时的状态重载将正在进行的 task 标记为 `failed`，附带 `CRASH_RECOVERY`。

## 交付物

本课产出 `outputs/skill-task-store-designer.md`。给定一个长时间运行的工具（研究、构建、导出），该技能设计 task 存储（状态形状、ttl、持久性），选择正确的 taskSupport 标志，并勾画进度通知。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` task，轮询状态，然后获取结果。

2. 在运行中添加 `tasks/cancel` 调用。验证 worker 遵守它且状态变为 `cancelled`。

3. 模拟崩溃恢复：杀死 worker 线程、重启加载器，观察 `CRASH_RECOVERY` 失败模式。

4. 将存储扩展为 SQLite。持久性优势相同；查询选项打开（列出会话 X 的所有 tasks）。

5. 阅读 MCP 2026 路线图帖子。找出最可能在下一年影响 SDK API 设计的一个 Tasks 相关开放问题。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| Task | "长时间运行的工具调用" | 用 `_meta.task` 增强以异步执行的请求 |
| SEP-1686 | "Tasks 规范" | 在 2025-11-25 添加 Tasks 的 Spec Evolution Proposal |
| `_meta.task` | "Task 信封" | 包含 id、state、ttl 的每请求元数据 |
| taskSupport | "工具标志" | 每工具的 `forbidden` / `optional` / `required` |
| `tasks/status` | "轮询方法" | 获取当前状态和可选进度提示 |
| `tasks/result` | "获取结果" | 返回已完成的有效载荷或未完成时 404 |
| `tasks/cancel` | "停止它" | 幂等取消请求 |
| ttl | "保留预算" | 服务器承诺保留 task 状态的毫秒数 |
| `notifications/tasks/updated` | "状态推送" | 服务器发起的状态变更事件 |
| 持久存储 | "崩溃安全状态" | 文件系统 / SQLite / Redis 持久层 |

## 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 原始提案及完整讨论
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 带理由的设计讲解
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — 机制和状态机
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK 级 task 实现模式
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 包括子任务在内的开放问题和 2026 优先级
