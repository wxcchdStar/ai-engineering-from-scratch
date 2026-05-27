# 实战项目 01 — 终端原生编程智能体（Claude Code、Codex CLI、Amp、OpenCode 风格）

> 2026 年的编程智能体（Coding Agent）已不再是演示版。Claude Code、Codex CLI、Amp、OpenCode、Cursor 3 Terminal 和 Gemini CLI 都遵循同一种生产形态：一个在 `git worktree` 中运行的智能体，拥有沙箱化文件系统的读写权限、一个 MCP 工具服务器、一个 LSP 感知的代码库索引，以及一个人类审批门控。本实战项目就是构建这样一个智能体，在 SWE-bench Pro 上运行，并发布一份与 Claude Code 和 Codex CLI 的并排对比报告。

**类型：** 实战项目（Capstone）
**语言：** Python（智能体核心）、TypeScript（TUI）
**前置课程：** 第 11 阶段（LLM 工程）、第 13 阶段（工具与 MCP）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P17
**时间：** 35 小时

## 问题

2026 年，编程智能体已从实验阶段进入生产环境。Claude Code 在 Anthropic 内部被广泛使用。Codex CLI 在 GitHub 上开源。Amp 和 OpenCode 在 SWE-bench Pro 上竞争排行榜位置。Cursor 3 将智能体模式直接嵌入 IDE。Gemini CLI 将 Google 的代码库索引引入终端。它们都遵循同一种形态：一个在隔离工作树中运行的智能体，拥有文件读写权限、一个 MCP 工具服务器、一个代码库索引（LSP 或 tree-sitter），以及一个人类审批门控。

你需要构建的并非演示版，而是一个在 SWE-bench Pro 上可衡量的智能体。你需要发布一份并排对比报告：你的智能体与 Claude Code 和 Codex CLI 在 50 个共享 issue 上的对比。你需要测量成本、延迟和审批摩擦。你需要解释差异所在。

## 概念

该智能体运行一个 ReAct 循环（推理-行动-观察），使用 Claude Opus 4.7 或 GPT-5.4-Codex 作为推理模型。工具包括：`read_file`、`edit_file`、`run_tests`、`git_diff`、`ripgrep`、`tree-sitter` 以及一个 LSP 查询工具。代码库索引是一个 tree-sitter AST 映射加上一个 BM25 文本索引，在启动时构建并缓存在磁盘上。

隔离通过 `git worktree add` 实现——每个任务获得一个独立的工作树，因此多个任务可以并行运行而不会发生文件冲突。沙箱化通过 Daytona 或 E2B 实现，提供网络出口控制和资源上限。审批门控意味着任何超出只读范围的操作（写入文件、运行命令、推送代码）都需要人类确认。

评估在 SWE-bench Pro 上进行，这是一个包含 50 个 issue 的子集，涵盖 Python、TypeScript、Go 和 Rust 仓库。指标包括：pass@1、每次解决 issue 的成本、每次 issue 的审批次数，以及 p50/p95 解决时间。

## 架构

```
用户 prompt / GitHub issue
      |
      v
LangGraph ReAct 智能体（Claude Opus 4.7 / GPT-5.4-Codex）
      |
      +-- 工具 ----+----+----+----+
      |            |    |    |    |
      v            v    v    v    v
read_file   edit_file  run_tests  git_diff  ripgrep  tree-sitter  LSP
      |
      +-- 代码库索引（tree-sitter AST + BM25 文本）
      |
      +-- git worktree（每个任务隔离）
      |
      +-- Daytona / E2B 沙箱（网络出口控制、资源上限）
      |
      v
审批门控（人类确认写入、命令、推送）
      |
      v
SWE-bench Pro 评估（50 个 issue，pass@1，成本，延迟）
```

## 技术栈

- 推理：Claude Opus 4.7（主要）、GPT-5.4-Codex（备选）
- 智能体框架：LangGraph 配合检查点与人工审批
- 代码库索引：tree-sitter AST 映射 + BM25 文本索引（Tantivy）
- 工具传输：FastMCP 通过 StreamableHTTP
- 隔离：`git worktree add` + Daytona 或 E2B 沙箱
- TUI：Textual（Python）或 Ink（TypeScript）
- 评估：SWE-bench Pro 50 个 issue 子集
- 可观测性：Langfuse 配合每任务追踪

## 构建步骤

1. **代码库索引。** 在启动时，遍历仓库，使用 tree-sitter 解析每个文件，构建 AST 节点映射。同时使用 Tantivy 构建 BM25 文本索引。将两者缓存到磁盘。

2. **工具服务器。** 通过 FastMCP 暴露 8 个工具：`read_file`、`edit_file`（精确字符串替换）、`run_tests`（带超时）、`git_diff`、`ripgrep`（正则搜索）、`tree_sitter_query`（AST 查询）、`lsp_goto_def`、`lsp_find_refs`。

3. **ReAct 循环。** LangGraph 配合三个节点：`reason`（模型规划下一步）、`act`（执行工具调用）、`observe`（将结果反馈给模型）。检查点每回合保存状态。

4. **审批门控。** 任何修改文件系统或执行命令的工具调用都需要人类确认。只读工具（read_file、ripgrep、tree_sitter_query）无需审批即可通过。

5. **工作树隔离。** 每个任务执行 `git worktree add` 创建一个独立分支。多个任务可以并行运行。任务完成后，工作树被删除。

6. **沙箱化。** Daytona 或 E2B 沙箱强制执行网络出口控制（仅允许 PyPI/npm/crates.io）和资源上限（8 GB 内存、4 个 CPU、30 分钟超时）。

7. **TUI。** 使用 Textual（Python）或 Ink（TypeScript）构建终端 UI。显示：当前任务、智能体思考过程、文件差异预览、审批提示。

8. **评估。** 在 SWE-bench Pro 的 50 个 issue 子集上运行。记录 pass@1、每次解决 issue 的成本、每次 issue 的审批次数、p50/p95 解决时间。

9. **并排对比。** 在相同的 50 个 issue 上运行 Claude Code 和 Codex CLI。发布一份报告，说明你的智能体在哪些方面胜出、哪些方面落后，以及原因。

## 使用方式

```
$ coding-agent solve --issue https://github.com/acme/widget/issues/842
[索引]   已索引 1,247 个文件，12,844 个 AST 节点，耗时 3.2 秒
[推理]   该 issue 涉及 widget.dedupe() 中的 NPE……
[行动]   read_file src/widget.py:142-180
[行动]   edit_file src/widget.py:155 将 `entry.getKey()` 替换为 `entry?.getKey()`
[行动]   run_tests test_dedupe.py
[观察]   12/12 测试通过
[审批]   提交差异？（y/n）：y
[完成]   已创建分支 fix/npe-dedupe，已暂存 1 个文件
```

## 交付标准

`outputs/skill-coding-agent.md` 描述交付物。一个在 SWE-bench Pro 上可衡量的终端原生编程智能体，附有与 Claude Code 和 Codex CLI 的并排对比报告。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 50 个 issue 子集，pass@1 |
| 20 | 成本与延迟 | 每次解决 issue 的成本、p50/p95 解决时间 |
| 20 | 工具覆盖与正确性 | 8 个工具全部正常工作；AST 查询返回正确结果 |
| 20 | 安全与审批 | 审批门控日志显示每次写入/命令调用均被门控 |
| 15 | 并排分析 | 与 Claude Code 和 Codex CLI 的对比报告 |
| **100** | | |

## 练习

1. 将推理模型从 Claude Opus 4.7 替换为 GPT-5.4-Codex。测量 pass@1 和每次 issue 成本的差异。
2. 添加一个"自动审批"模式，用于差异小于 20 行的安全重构。测量审批次数减少了多少，以及是否引入了任何错误。
3. 实现一个"学习"模式：智能体在每次任务后将成功的差异模式保存到向量存储中，并在后续任务中检索相似的修复方案。
4. 在 SWE-bench Pro 上测量 LSP 工具（goto_def、find_refs）对 pass@1 的提升效果。与仅使用 ripgrep 的基线进行对比。
5. 添加多文件编辑支持：智能体可以在一次回合中编辑多个文件，然后再运行测试。测量这对回合数和成本的影响。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| ReAct 循环 | "推理-行动-观察" | 智能体推理、调用工具、观察结果、重复执行的循环 |
| git worktree | "隔离分支" | 从同一仓库克隆出的独立工作目录，每个目录位于不同分支 |
| MCP 工具服务器 | "工具 API" | 通过 Model Context Protocol 暴露工具的标准方式 |
| tree-sitter | "AST 解析器" | 用于代码库索引的增量、容错解析器 |
| SWE-bench Pro | "编程智能体基准" | 2026 年用于评估编程智能体的标准基准 |
| 审批门控 | "人类确认" | 在允许破坏性操作之前要求人类确认 |
| pass@1 | "首次尝试通过率" | 智能体在单次尝试中解决 issue 的比例 |

## 扩展阅读

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — 参考生产形态
- [Codex CLI 开源](https://github.com/openai/codex) — 备选参考实现
- [Amp](https://ampcode.com) — 备选排行榜竞争者
- [OpenCode](https://github.com/sst/opencode) — 备选开源实现
- [SWE-bench Pro](https://www.swebench.com) — 评估目标
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考智能体编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [Daytona 沙箱](https://daytona.io) — 参考隔离环境
