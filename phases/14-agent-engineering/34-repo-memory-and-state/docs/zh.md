# 仓库记忆与持久状态

> 聊天历史是易失的。仓库是持久的。Workbench 将 agent 状态存储在版本化文件中，以便下一个会话、下一个 agent 和下一个审查者都从同一个真相源读取。

**类型：** 构建
**语言：** Python（标准库 + `jsonschema` 可选）
**前置条件：** Phase 14 · 32（最小 Workbench）
**时间：** 约 60 分钟

## 学习目标

- 定义什么属于仓库记忆，什么属于聊天历史。
- 为 `agent_state.json` 和 `task_board.json` 编写 JSON Schema。
- 构建一个状态管理器，原子地加载、验证、变更和持久化状态。
- 使用 schema 在坏写入破坏 workbench 之前拒绝它们。

## 问题

Agent 完成一个会话。聊天关闭。下一个会话打开并询问从哪里开始。模型说"让我检查文件"，读取过时的笔记，重做已经完成的工作。或者更糟，它重写了一个已完成的文件，因为没有人告诉它文件已完成。

Workbench 的修复是仓库记忆：状态存在于仓库的 JSON 文件中，在 schema 下编写，原子持久化，在代码审查中对 diff 友好。聊天是瞬态输入；仓库是记录系统。

## 概念

```mermaid
flowchart LR
  Agent[Agent 循环] --> Manager[StateManager]
  Manager --> Schema[agent_state.schema.json]
  Schema --> Validate{有效？}
  Validate -- 是 --> Write[agent_state.json]
  Validate -- 否 --> Reject[拒绝 + 抛出]
  Write --> Manager
```

### 什么属于仓库记忆

| 属于 | 不属于 |
|------|-------|
| 活动任务 ID | 原始聊天记录 |
| 本会话修改的文件 | Token 级推理追踪 |
| Agent 做出的假设 | "用户似乎很沮丧" |
| 开放阻塞项 | 采样补全 |
| 下一步行动 | 供应商特定模型 ID |

测试是持久性：三个月后在 CI 重跑中这会有用吗？如果是，仓库。如果否，遥测。

### Schema 优先的状态

JSON Schema 是合约。没有它，每个 agent 发明新字段，每个审查者学习新形状，每个 CI 脚本必须特殊处理过去的版本。有了它，坏写入就是被拒绝的写入。

Schema 覆盖：

- 必需键。
- 允许的 `status` 值。
- 禁止的值（例如数组的 `null`）。
- 模式约束（任务 ID 匹配 `T-\d{3,}`）。
- 用于迁移的版本字段。

### 原子写入

状态写入需要经受部分失败：写入临时文件，fsync，重命名覆盖目标。状态文件是真相源；半写入的文件比没有文件更糟。

### 迁移

当 schema 变更时，在 schema 版本号提升旁边发布迁移脚本。状态文件携带 `schema_version` 字段；管理器拒绝加载无法迁移的版本的文件。

## 构建

`code/main.py` 实现：

- `agent_state.schema.json` 和 `task_board.schema.json`。
- 仅标准库的验证器（JSON Schema 子集：required、type、enum、pattern、items）。
- `StateManager.load`、`StateManager.update`、`StateManager.commit`，带原子临时文件并重命名写入。
- 一个演示：变更状态、持久化、重新加载，并证明往返。

运行：

```
python3 code/main.py
```

脚本写入 `workdir/agent_state.json` 和 `workdir/task_board.json`，跨两轮变更它们，并在每一步打印验证后的状态。

## 实际中的生产模式

四个模式将课程的最小值转变为多 agent 单体仓库可以存活的东西。

**原子临时文件并重命名不是可选的。** 2026 年 3 月的 Hive 项目 bug 报告清晰地记录了失败模式：`state.json` 通过 `write_text()` 写入，异常被捕获并静默。部分写入导致会话在损坏的状态上恢复，没有信号。修复始终是：`tempfile.mkstemp` 在与目标相同的目录中，写入，`fsync`，`os.replace`（POSIX 和 Windows 上的原子重命名）。本课的 `atomic_write` 正是这样做的。

**每个非幂等工具调用的幂等键。** 如果 agent 在调用工具后但在检查点记录结果前崩溃，恢复会重试工具调用。对读取安全；对电子邮件、数据库插入、文件上传危险。模式：在执行前将每个工具调用 ID 记录到 `pending_calls.jsonl`。重试时，检查 ID；如果存在，跳过调用并使用缓存结果。Anthropic 和 LangChain 都在 2026 年指南中指出了这一点；LangGraph 的 checkpointer 出于同样原因持久化待处理写入。

**将大型产物与状态分离。** 不要在 `agent_state.json` 中存储 CSV、长记录或生成的文件。将产物保存为单独文件（或上传到对象存储），仅在状态中保留路径。检查点保持小且快；产物独立增长。

**事件溯源用于审计，快照用于恢复。** 在每次变更时追加到事件日志（`state.events.jsonl`）；定期快照到 `state.json`。恢复读取快照，然后重放快照时间戳之后的任何事件。这消耗更多磁盘，但允许你逐字重放 agent 决策——在调试长周期运行时至关重要。与 Postgres 内部用于 WAL 的形状相同。

**Schema 迁移或拒绝加载。** `schema_version` 整数是合约。当管理器加载未知版本的文件时，拒绝读取。在 schema 版本号提升旁边发布迁移脚本；`tools/migrate_state.py` 在每次启动时幂等运行。

## 使用

在生产中：

- **LangGraph checkpointers。** 相同思想，不同存储。Checkpointer 将图状态持久化到 SQLite、Postgres 或自定义后端。本课教授的 schema 是当 checkpointer 死掉且你需要手动读取状态时使用的东西。
- **Letta memory blocks。** 具有结构化 schema 的持久块（Phase 14 · 08）。相同学科，范围限定为长期角色。
- **OpenAI Agents SDK session store。** 可插拔后端，schema 感知。本课中的状态文件是本地文件后端。

## 交付

`outputs/skill-state-schema.md` 生成项目特定的 JSON Schema 对（状态 + 板），一个接入原子写入的 Python `StateManager`，以及一个迁移脚手架，使下一个 schema 版本号提升不会破坏 workbench。

## 练习

1. 添加 `last_human_touch` 时间戳。拒绝在人类编辑后五秒内的任何 agent 写入。
2. 扩展验证器以支持 `oneOf`，使任务可以是构建任务或审查任务，具有不同的必需字段。
3. 添加 `schema_version` 字段并编写从 v1 到 v2 的迁移（将 `blockers` 重命名为 `risks`）。
4. 将存储后端从本地文件移动到 SQLite。保持 `StateManager` API 相同。
5. 以 50 毫秒写入竞争运行两个 agent 对同一状态文件。什么会出错，原子重命名如何拯救你？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 仓库记忆 | "笔记文件" | 在仓库的跟踪文件中存储的状态，在 schema 下 |
| Schema 优先 | "验证输入" | 在写入者之前定义合约，拒绝漂移 |
| 原子写入 | "只是重命名" | 写入临时文件，fsync，重命名，使部分失败无法损坏 |
| 迁移 | "Schema 版本号提升" | 将 vN 状态转换为 v(N+1) 状态的脚本 |
| 记录系统 | "真相源" | workbench 视为权威的产物 |

## 扩展阅读

- [JSON Schema specification](https://json-schema.org/specification.html)
- [LangGraph checkpointers](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Letta memory blocks](https://docs.letta.com/concepts/memory)
- [Fast.io, AI Agent State Checkpointing: A Practical Guide](https://fast.io/resources/ai-agent-state-checkpointing/) — 带幂等性的 schema 优先检查点
- [Fast.io, AI Agent Workflow State Persistence: Best Practices 2026](https://fast.io/resources/ai-agent-workflow-state-persistence/) — 并发控制、TTL、事件溯源
- [Hive Issue #6263 — 非原子 state.json 写入被静默忽略](https://github.com/aden-hive/hive/issues/6263) — 真实项目中的失败模式
- [eunomia, Checkpoint/Restore Systems: Evolution, Techniques, Applications](https://eunomia.dev/blog/2025/05/11/checkpointrestore-systems-evolution-techniques-and-applications-in-ai-agents/) — 从 OS 历史应用于 agent 的 CR 原语
- [Indium, 7 State Persistence Strategies for Long-Running AI Agents in 2026](https://www.indium.tech/blog/7-state-persistence-strategies-ai-agents-2026/)
- [Microsoft Agent Framework, Compaction](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/compaction) — 供应商检查点管理器
- Phase 14 · 08 — 记忆块和睡眠时计算
- Phase 14 · 32 — 本课模式化的三文件最小值
- Phase 14 · 40 — 从相同 schema 读取的交接包
