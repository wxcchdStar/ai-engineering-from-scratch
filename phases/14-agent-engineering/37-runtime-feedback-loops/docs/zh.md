# 运行时反馈循环

> 看不到真实命令输出的 agent 在猜测。反馈运行器将 stdout、stderr、退出码和耗时捕获到下一轮可以读取的结构化记录中。然后 agent 对事实做出反应，而不是对自己对事实的预测做出反应。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** Phase 14 · 32（最小 Workbench），Phase 14 · 35（初始化脚本）
**时间：** 约 50 分钟

## 学习目标

- 区分运行时反馈与可观测性遥测。
- 构建一个反馈运行器，包装 shell 命令并持久化结构化记录。
- 确定性截断大型输出，使循环保持在 token 预算内。
- 当反馈缺失时拒绝推进循环。

## 问题

Agent 说"正在运行测试"。下一条消息说"所有测试通过"。现实是没有测试运行。Agent 想象了输出，或者它运行了命令但从未读取结果，或者它读取了结果并静默截断了失败行。

反馈运行器消除了这个差距。每个命令通过运行器。每条记录携带命令、捕获的 stdout 和 stderr、退出码、墙钟持续时间和一行 agent 注释。Agent 在下一轮读取记录。验证门在任务结束时读取记录。

## 概念

```mermaid
flowchart LR
  Agent[Agent 循环] --> Runner[run_with_feedback.py]
  Runner --> Shell[subprocess]
  Shell --> Capture[stdout / stderr / exit / duration]
  Capture --> Record[feedback_record.jsonl]
  Record --> Agent
  Record --> Gate[验证门]
```

### 反馈记录中包含什么

| 字段 | 为什么重要 |
|------|----------|
| `command` | 确切的 argv，无 shell 展开意外 |
| `stdout_tail` | 最后 N 行，确定性截断 |
| `stderr_tail` | 最后 N 行，与 stdout 分开 |
| `exit_code` | 明确的成功信号 |
| `duration_ms` | 暴露慢探测和失控进程 |
| `started_at` | 用于重放的时间戳 |
| `agent_note` | Agent 写的关于它期望什么的一行 |

### 截断是确定性的

50 MB 的日志会摧毁循环。运行器用 `...truncated N lines...` 标记截断头部和尾部，确定性使相同输出始终产生相同记录。不采样；agent 需要看到的部分（最终错误、最终摘要）位于尾部。

### 反馈与遥测

遥测（Phase 14 · 23，OTel GenAI 约定）是为人类操作员跨时间审查运行。反馈是为本次运行的下一轮。它们共享字段，但存在于不同文件中，具有不同的保留策略。

### 没有反馈拒绝推进

如果运行器在捕获退出之前出错，记录携带 `exit_code: null` 和 `error: <reason>`。Agent 循环必须拒绝在 `null` 退出上声明成功。没有退出，没有进展。

## 构建

`code/main.py` 实现：

- `run_with_feedback(command, agent_note)`，包装 `subprocess.run`，捕获 stdout/stderr/exit/duration，确定性截断，追加到 `feedback_record.jsonl`。
- 一个小型加载器，将 JSONL 流式传输到 Python 列表。
- 一个运行三个命令（成功、失败、慢）的演示，并打印每个命令的最后一条记录。

运行：

```
python3 code/main.py
```

输出：三条反馈记录追加到 `feedback_record.jsonl`，每条的最后一条内联打印。跨重新运行 tail 文件以查看循环累积。

## 实际中的生产模式

三个模式使运行器足够坚固以交付。

**写入时脱敏，而非读取时。** 任何触及 stdout 或 stderr 的记录都可能泄漏密钥。运行器在 JSONL 追加之前进行脱敏：剥离匹配 `^Bearer `、`password=`、`api[_-]?key=`、`AKIA[0-9A-Z]{16}`（AWS）、`xox[baprs]-`（Slack）的行。读取时脱敏是陷阱；磁盘上的文件是攻击者接触到的。每季度对照生产运行时观察到的密钥格式审计脱敏模式。

**轮换策略，而非单个文件。** 将 `feedback_record.jsonl` 限制为每个文件 1 MB；溢出时轮换到 `.1`、`.2`，丢弃 `.5`。Agent 的循环只读取当前文件，因此运行时成本有界。CI 产物存储获取完整的轮换集。没有轮换，文件成为每次加载器调用的瓶颈。

**重试链的父命令 ID。** 每条记录获得 `command_id`；重试携带 `parent_command_id` 指向前一次尝试。审查者的"失败尝试"列表（Phase 14 · 40）和验证门的审计都遵循该链。没有此链接，重试看起来像独立的成功，审计隐藏了失败历史。

## 使用

生产模式：

- **Claude Code Bash 工具。** 该工具已经捕获 stdout、stderr、exit 和 duration。本课中的运行器是任何 agent 产品的框架无关等价物。
- **LangGraph 节点。** 将任何 shell 节点包装在运行器中，使记录在图状态之外持久化。
- **CI 日志。** 将 JSONL 管道传输到 CI 产物存储；审查者可以重放任何命令而无需重新运行会话。

运行器是一个薄包装器，在每次框架迁移中存活，因为它拥有记录的形状。

## 交付

`outputs/skill-feedback-runner.md` 生成项目特定的 `run_with_feedback.py`，具有正确的截断预算、接入 workbench 的 JSONL 写入器，以及 agent 每轮读取的加载器。

## 练习

1. 为每条记录添加 `cwd` 字段，使从不同目录运行的相同命令可区分。
2. 添加脱敏步骤，剥离匹配 `^Bearer ` 或 `password=` 的行。在夹具记录上测试。
3. 通过轮换到 `.1`、`.2` 文件将总 `feedback_record.jsonl` 大小限制为 1 MB。为轮换策略辩护。
4. 添加 `parent_command_id` 使重试链可见：哪个命令产生了下一个命令消费的输入。
5. 将 JSONL 管道传输到一个微型 TUI，高亮最新的非零退出。TUI 必须显示的八个关键特性以在审查中有用。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 反馈记录 | "运行日志" | 结构化 JSONL 条目，包含命令、输出、退出、持续时间 |
| 尾部截断 | "修剪日志" | 确定性头+尾捕获，使记录适合 token 预算 |
| 拒绝 null | "缺失数据时阻止" | 当 `exit_code` 为 null 时循环不得推进 |
| Agent 注释 | "期望标签" | Agent 在读取结果前写的一行预测 |
| 遥测分离 | "两个日志文件" | 反馈用于下一轮，遥测用于操作员 |

## 扩展阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Guardrails AI x MLflow — deterministic safety, PII, quality validators](https://guardrailsai.com/blog/guardrails-mlflow) — 脱敏模式作为回归测试
- [Aport.io, Best AI Agent Guardrails 2026: Pre-Action Authorization Compared](https://aport.io/blog/best-ai-agent-guardrails-2026-pre-action-authorization-compared/) — 工具前/后捕获
- [Andrii Furmanets, AI Agents in 2026: Practical Architecture for Tools, Memory, Evals, Guardrails](https://andriifurmanets.com/blogs/ai-agents-2026-practical-architecture-tools-memory-evals-guardrails) — 可观测性表面
- Phase 14 · 23 — 遥测侧的 OTel GenAI 约定
- Phase 14 · 24 — agent 可观测性平台（Langfuse、Phoenix、Opik）
- Phase 14 · 33 — 要求在声明完成前获得反馈的规则
- Phase 14 · 38 — 读取 JSONL 的验证门
