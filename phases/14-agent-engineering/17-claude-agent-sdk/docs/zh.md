# Claude Agent SDK：子 Agent 与会话存储 (Claude Agent SDK: Subagents and Session Store)

> Claude Agent SDK 是 Claude Code harness 的库形式。内置工具、用于上下文隔离的子 Agent、钩子、W3C 追踪传播、会话存储对等。Claude Managed Agents 是用于长时间运行异步工作的托管替代方案。

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 01 (Agent 循环), Phase 14 · 10 (技能库)
**时间：** 约 75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（harness 形态）之间的区别。
- 描述子 Agent——并行化和上下文隔离——以及何时使用它们。
- 列举 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 用标准库实现一个 harness，包含内置工具、带隔离上下文的子 Agent 生成、生命周期钩子和会话存储。

## 问题

原始 LLM API 给你一次往返。生产 Agent 需要工具执行、MCP 服务器、生命周期钩子、子 Agent 生成、会话持久化、追踪传播。Claude Agent SDK 将此形态作为库发布——Claude Code 使用的同一个 harness，暴露给自定义 Agent。

## 概念

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`)。** 原始 Messages API。你拥有循环、工具、状态。
- **Agent SDK (`claude-agent-sdk`)。** 内置工具执行、MCP 连接、钩子、子 Agent 生成、会话存储。Claude Code 循环作为库。

### 内置工具

SDK 开箱即用 10+ 工具：文件读写、shell、grep、glob、网络抓取等。自定义工具通过标准工具 schema 接口注册。

### 子 Agent

Anthropic 记录的两个目的：

1. **并行化。** 并发运行独立工作。"为这 20 个模块分别找到测试文件"是 20 个并行子 Agent 任务。
2. **上下文隔离。** 子 Agent 使用自己的上下文窗口；只有结果返回给编排器。编排器的预算得以保留。

Python SDK 最近新增：`list_subagents()`、`get_subagent_messages()` 用于读取子 Agent 转录。

### 会话存储

与 TypeScript 协议对等：

- `append(session_id, message)` — 添加一个轮次。
- `load(session_id)` — 恢复对话。
- `list_sessions()` — 枚举。
- `delete(session_id)` — 级联到子 Agent 会话。
- `list_subkeys(session_id)` — 列出子 Agent 键。

`--session-mirror` (CLI 标志) 在流式传输时将转录镜像到外部文件，用于调试。

### 钩子

你可以注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` — 门控或审计工具调用。
- `SessionStart`、`SessionEnd` — 设置和清理。
- `UserPromptSubmit` — 在模型看到之前对用户输入采取行动。
- `PreCompact` — 在上下文压缩前运行。
- `Stop` — Agent 退出时清理。
- `Notification` — 侧通道告警。

钩子是 pro-workflow 和类似系统添加横切行为的方式。

### W3C 追踪上下文

调用方上活跃的 OTel span 通过 W3C 追踪上下文头传播到 CLI 子进程。整个多进程追踪在你的后端显示为一条追踪。

### Claude Managed Agents

托管替代方案（beta 头 `managed-agents-2026-04-01`）。长时间运行异步工作，内置 prompt 缓存，内置压缩。用控制换取托管基础设施。

### 这个模式哪里会出错

- **子 Agent 过度生成。** 为 100 个小任务生成 100 个子 Agent。开销占主导。改为批处理。
- **钩子蔓延。** 每个团队添加钩子；启动时间膨胀。每季度审查钩子。
- **会话膨胀。** 会话累积；大小增长。使用 `list_sessions` + 过期策略。

## 构建

`code/main.py` 用标准库实现 SDK 形态：

- `Tool`、`ToolRegistry` 带内置的 `read_file`、`write_file`、`list_dir`。
- `Subagent` — 私有上下文，隔离运行，返回结果。
- `SessionStore` — append、load、list、delete、list_subkeys。
- `Hooks` — `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 演示：主 Agent 并行生成 3 个子 Agent（各自隔离），聚合结果，持久化会话。

运行：

```
python3 code/main.py
```

追踪显示子 Agent 上下文隔离（编排器上下文大小保持有界）、钩子执行和会话持久化。

## 使用

- **Claude Agent SDK** 用于想要 Claude Code harness 形态的 Claude 优先产品。
- **Claude Managed Agents** 用于托管长时间运行异步工作。
- **OpenAI Agents SDK** (第 16 课) 用于 OpenAI 优先的对应物。
- **LangGraph + 自定义工具** 如果你想要图形状状态机。

## 交付

`outputs/skill-claude-agent-scaffold.md` 搭建一个 Claude Agent SDK 应用，包含子 Agent、钩子、会话存储、MCP 服务器附加和 W3C 追踪传播。

## 练习

1. 添加一个子 Agent 生成器，将 20 个任务批处理为每组 5 个并行子 Agent。测量编排器上下文大小 vs 每个任务一个。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用进行速率限制（每个会话每分钟 5 次）。追踪行为。
3. 接入 `list_subkeys` 渲染子 Agent 树。深度嵌套看起来如何？
4. 将玩具移植到真实的 `claude-agent-sdk` Python 包。工具注册有什么变化？
5. 阅读 Claude Managed Agents 文档。你何时会从自托管切换到托管？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Agent SDK | "Claude Code 作为库" | Harness 形态：工具、MCP、钩子、子 Agent、会话存储 |
| 子 Agent (Subagent) | "子 Agent" | 独立上下文，自有预算；结果向上冒泡 |
| 会话存储 (Session store) | "对话数据库" | 持久化、加载、列出、删除轮次，带子 Agent 级联 |
| 钩子 (Hook) | "生命周期回调" | 工具前/后、会话、prompt 提交、压缩、停止 |
| W3C 追踪上下文 (W3C trace context) | "跨进程追踪" | 父 span 传播到 CLI 子进程 |
| Managed Agents | "托管 harness" | Anthropic 托管的长时间运行异步工作 |
| `--session-mirror` | "转录镜像" | 在流式传输时将会话轮次写入外部文件 |
| MCP 服务器 | "工具接口" | 附加到 Agent 的外部工具/资源源 |

## 扩展阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产模式
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对应物
