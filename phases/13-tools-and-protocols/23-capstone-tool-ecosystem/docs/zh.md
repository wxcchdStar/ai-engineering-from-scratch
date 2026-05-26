# 综合项目 — 构建完整的工具生态系统

> Phase 13 教授了每一个组成部分。本综合项目将它们串联为一个生产级系统：一个包含 tools + resources + prompts + tasks + UI 的 MCP 服务器、边缘 OAuth 2.1、一个 RBAC 网关、一个多服务器客户端、A2A 子 agent 调用、发送到收集器的 OTel 追踪、CI 中的工具投毒检测，以及 AGENTS.md + SKILL.md 组合包。结束时你将能为每个架构选择进行辩护。

**类型：** 构建
**语言：** Python（标准库，端到端生态系统框架）
**前置课程：** Phase 13 · 01 至 21
**时长：** 约 120 分钟

## 学习目标

- 组合一个暴露 tools、resources、prompts 和带 `ui://` 应用的 task 的 MCP 服务器。
- 在服务器前使用强制执行 RBAC 和固定哈希的 OAuth 2.1 网关。
- 编写一个使用 OTel GenAI 属性进行端到端追踪的多服务器客户端。
- 将部分工作负载委派给 A2A 子 agent；验证不透明性是否得以保持。
- 用 AGENTS.md + SKILL.md 打包整个技术栈，使其他 agent 可以驱动它。

## 问题背景

交付"研究与报告"系统：

- 用户提问："总结关于 agent 协议的被引用最多的三篇 2026 年 arXiv 论文。"
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要委派给专业写作 agent；聚合结果；将交互式报告渲染为 MCP Apps `ui://` 资源；将每一步记录到 OTel。

Phase 13 的所有原语悉数登场。这不是玩具——Anthropic（Claude Research 产品）、OpenAI（GPTs with Apps SDK）和第三方在 2026 年交付的生产级研究助手系统正是这种形态。

## 核心概念

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### Trace 层级

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

一个 trace id。每个 span 都有正确的 `gen_ai.*` 属性。

### 安全态势

- OAuth 2.1 + PKCE，资源指示器将受众固定到网关。
- 网关持有上游凭证；用户永远看不到它们。
- RBAC：`alice` 有 `research:read`、`research:write`，可以调用所有工具。`bob` 有 `research:read`，不能调用 `generate_report`。
- 固定描述清单：丢弃任何工具哈希发生变化的服务器。
- Rule of Two 审计：没有任何工具同时组合不可信输入、敏感数据和后果性操作。

### 渲染

最终的 `generate_report` 任务返回内容块加上 `ui://report/current` 资源。客户端宿主（Claude Desktop 等）在沙箱 iframe 中渲染交互式仪表盘。仪表盘包含排序的论文列表、引用计数，以及一个按钮，用户点击后调用 `host.callTool('summarize_paper', {arxiv_id})` 对任意论文进行摘要。

### 打包

整个系统以下列形式发布：

```
research-system/
  AGENTS.md                     # 项目约定
  skills/
    run-research/
      SKILL.md                  # 顶层工作流
  servers/
    research-mcp/               # MCP 服务器
      pyproject.toml
      src/
  agents/
    writer/                     # A2A agent
  gateway/
    config.yaml                 # RBAC + 固定清单
```

用户通过 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` skill 来驱动系统。

### 每课贡献清单

| 课程 | 综合项目使用的内容 |
|------|-------------------|
| 01-05 | 工具接口、提供商可移植性、并行调用、schema、linting |
| 06-10 | MCP 原语、服务器、客户端、传输、资源 + 提示词 |
| 11-14 | 采样、roots + elicitation、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册中心 |
| 18 | A2A 子 agent 委派 |
| 19 | OTel GenAI 追踪 |
| 20 | LLM 层路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 动手实践

`code/main.py` 将之前课程的模式缝合为一个可运行的完整演示。全部使用标准库，全部在进程中运行，因此你可以从头到尾阅读理解。它运行研究和报告场景的完整流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、将 generate_report 作为 task、A2A 调用 writer、返回 ui:// 资源、发出 OTel span。

关注点：

- 跨所有跳的单一 trace id。
- 网关策略阻止第二个用户的写操作。
- Task 生命周期走完 working → completed 并返回文本和 ui:// 内容。
- A2A 调用的内部状态对编排器不透明。
- AGENTS.md 和 SKILL.md 是另一个 agent 复现工作流程所需的唯一文件。

## 交付产物

本课产出 `outputs/skill-ecosystem-blueprint.md`。给定产品需求（研究、摘要、自动化），该技能产出完整架构：使用哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测、哪种打包方式。

## 练习

1. 运行 `code/main.py`。注意单一 trace id 以及 span 如何嵌套。统计演示接触了 Phase 13 中多少个原语。

2. 扩展演示：添加第二个后端 MCP 服务器（如 `bibliography`），确认网关将其工具合并到同一命名空间。

3. 将伪造的 A2A writer agent 替换为在子进程上运行的真实 agent。使用 Lesson 19 的框架。

4. 在编排器和 LLM 之间的路由网关中添加 PII 脱敏步骤。确认用户查询中的电子邮件被清洗。

5. 编写一份 AGENTS.md，面向将要维护此系统的队友。应在五分钟内读完，并给出他们在 Cursor 或 Codex 中驱动该综合项目所需的一切。

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Capstone | "Phase 13 集成演示" | 使用每一个原语的端到端系统 |
| Research and report | "场景" | 搜索、总结、渲染模式 |
| Ecosystem | "所有组件一起" | 服务器 + 客户端 + 网关 + 子 agent + 遥测 + 打包 |
| Trace hierarchy | "单一 trace id" | 每个跳的 span 共享 trace；父子关系通过 span id 链接 |
| Gateway-issued token | "传递认证" | 客户端只能看到网关令牌；网关持有上游凭证 |
| Merged namespace | "所有工具在一个扁平列表中" | 网关层面的多服务器合并，冲突时加前缀 |
| Opacity boundary | "A2A 调用隐藏内部细节" | 子 agent 的推理对编排器不可见 |
| Three-layer stack | "AGENTS.md + SKILL.md + MCP" | 项目上下文 + 工作流 + 工具 |
| Defense-in-depth | "多层安全" | 固定哈希、OAuth、RBAC、Rule of Two、审计日志 |
| Spec compliance matrix | "我们交付了什么符合规范要求" | 将交付物映射到 2025-11-25 要求的检查清单 |

## 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 整合参考
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议发展方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 追踪约定标准
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — 生产环境 agent 运行时模式