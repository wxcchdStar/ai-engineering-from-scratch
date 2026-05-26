# Skills 与 Agent SDK — Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说"存在哪些工具"。Skills 说"如何完成一个任务"。2026 年的技术栈将二者分层叠加。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）以 SKILL.md 形式发布，支持渐进式披露。OpenAI 的 Apps SDK 是 MCP 加 widget 元数据。AGENTS.md（现已有 60,000+ 仓库使用）位于仓库根目录，作为项目级 agent 上下文。本课列出每种文件涵盖的内容，并构建一个可跨 agent 迁移的最小 SKILL.md + AGENTS.md 组合包。

**类型：** 学习
**语言：** Python（标准库，SKILL.md 解析器和加载器）
**前置课程：** Phase 13 · 07（MCP 服务器）
**时长：** 约 45 分钟

## 学习目标

- 区分三个层次：AGENTS.md（项目上下文）、SKILL.md（可复用知识）、MCP（工具）。
- 编写一个带有 YAML frontmatter 和渐进式披露的 SKILL.md。
- 以文件系统方式将 skill 加载到 agent 运行时。
- 组合 skill 与 MCP 服务器和 AGENTS.md，使一个包同时在 Claude Code、Cursor 和 Codex 中运行。

## 问题背景

一位工程师将发布说明撰写工作流程提炼为一个多步提示词："读取最近合并的 PR。按领域分组。逐一总结。按团队风格撰写 changelog 条目。发送到 Slack 草稿。"他们将其放在团队 Notion 文档中。

现在他们想在 Claude Code、Cursor 和 Codex CLI 中使用此工作流程。每个 agent 加载指令的方式不同：Claude Code 斜杠命令、Cursor rules、Codex `.codex.md`。该工程师复制工作流程三次，维护三份副本。

AGENTS.md 和 SKILL.md 一起解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的 agent 在会话启动时读取它。"这个项目是如何工作的？有哪些约定？哪些命令运行测试？"
- **SKILL.md** 是一个可移植组合包：YAML frontmatter（名称、描述）+ markdown 正文 + 可选的资源。支持 skill 的 agent 按名称按需加载。
- **MCP** (Phase 13 · 06-14) 处理该 skill 需要调用的工具。

三个层次，一个可移植产出物。

## 核心概念

### AGENTS.md (agents.md)

于 2025 年底推出，到 2026 年 4 月已被 60,000+ 仓库采纳。一个文件位于仓库根目录。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agent 在会话启动时读取此文件，并用其为该项目校准行为。2026 年每个编码 agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（作为开放标准于 2025 年 12 月发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明 skill 的身份。正文是 skill 加载时展示给模型的提示词。

### 渐进式披露

Skill 可引用子资源，agent 仅在需要时才拉取。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 说"see style-guide.md for the style rules." agent 仅在 skill 活跃运行时才拉取 style-guide.md。这避免将模型可能不需要的细节塞满提示词。

### 文件系统发现

Agent 运行时扫描已知目录中的 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载依据是文件夹名和 frontmatter 的 `name`。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨 agent）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk` (TypeScript) 和 `claude-agent-sdk` (Python) 在会话启动时加载 skill，将它们暴露为运行时内可调用的"agent"。Agent 循环在用户调用时将请求分发到 skill。

### OpenAI Apps SDK

于 2025 年 10 月推出；直接构建在 MCP 之上。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到一个开发者表面。一个 Apps SDK 应用即：

- 一个 MCP 服务器（工具、资源、提示词）。
- 加上 ChatGPT UI 的 widget 元数据。
- 加上可选的 MCP Apps `ui://` 资源用于交互式界面。

同一协议，更丰富的用户体验。

### 跨 Agent 可移植性（通过 SkillKit）

SkillKit 和类似跨 agent 分发层工具将单个 SKILL.md 转换为 32 个以上 AI agent（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）各自的本地格式。一个权威来源；多个消费方。

### 三层栈

| 层级 | 文件 | 加载时机 | 用途 |
|------|------|----------|------|
| AGENTS.md | 仓库根目录 | 会话启动时 | 项目级约定 |
| SKILL.md | skills 目录 | skill 被调用时 | 可复用工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用的操作 |

三层组合使用：agent 在会话启动时读取 AGENTS.md，用户调用一个 skill，该 skill 的指令包含 MCP 工具调用，agent 通过 MCP 客户端分发。

## 动手实践

`code/main.py` 提供一个标准库 SKILL.md 解析器和加载器。它在 `./skills/` 下发现 skill，解析 YAML frontmatter 加 markdown 正文，并生成以 skill 名称为键的字典。然后模拟一个按名称调用 `release-notes-writer` 的 agent 循环。

关注点：

- 使用最小化标准库解析器解析 YAML frontmatter（无需 `pyyaml` 依赖）。
- Skill 正文原样存储；agent 在调用时将其附加到系统提示词之前。
- 通过 `read_subresource` 函数演示渐进式披露，按需拉取引用文件。

## 交付产物

本课产出 `outputs/skill-agent-bundle.md`。给定一个工作流程，该技能产出组合的 SKILL.md + AGENTS.md + MCP-server-blueprint 组合包，可跨 agent 移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个 skill，确认加载器能够获取它。

2. 为本课程仓库编写一份 AGENTS.md。包含测试命令、风格约定和 Phase 13 思维模型。

3. 将你团队内部文档中的一个多步工作流程移植为 SKILL.md。验证其在 Claude Code 中加载。

4. 手动将该 skill 翻译为 Cursor 和 Codex 的原生规则格式。计算格式间的差异——这就是 SkillKit 自动化的翻译表面。

5. 阅读 Anthropic Agent Skills 博文。找出 Claude Agent SDK 中本课加载器未涵盖的一项功能。（提示：agent 子调用。）

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| SKILL.md | "skill 文件" | YAML frontmatter 加 markdown 正文，由 agent 运行时加载 |
| AGENTS.md | "仓库根 agent 上下文" | 会话启动时读取的项目级约定文件 |
| Progressive disclosure | "懒加载子资源" | Skill 正文引用的文件仅在需要时拉取 |
| Frontmatter | "顶部的 YAML 块" | `---` 分隔符内的元数据（名称、描述） |
| Claude Agent SDK | "Anthropic 的 skill 运行时" | `@anthropic-ai/claude-agent-sdk`，加载 skill 并路由 |
| OpenAI Apps SDK | "MCP + widget meta" | OpenAI 基于 MCP 加 ChatGPT UI 钩子的开发者表面 |
| Skill discovery | "文件系统扫描" | 遍历已知目录查找 SKILL.md，按名称键控 |
| Cross-agent portability | "一个 skill 多个 agent" | 通过 SkillKit 类工具将一个 SKILL.md 转换为 32+ agent 格式 |
| Agent Skill | "可移植知识" | MCP tool 概念之外的可复用任务模板 |
| Apps SDK | "MCP 加 ChatGPT UI" | Connectors 和 Custom GPTs 在 MCP 上统一 |

## 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式及采纳列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方 skill 示例