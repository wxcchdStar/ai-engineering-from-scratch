# 记忆块与睡眠时计算 (Memory Blocks and Sleep-Time Compute, Letta)

> MemGPT 于 2024 年更名为 Letta。2026 年的演进增加了两个理念：模型可直接编辑的离散功能记忆块，以及一个在主 Agent 空闲时异步整合记忆的睡眠时 Agent。这就是将记忆扩展到单次对话之外的方法。

**类型：** 构建 (Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 07 (MemGPT)
**时间：** 约 75 分钟

## 学习目标

- 列举 Letta 使用的三个记忆层级（core、recall、archival）及其各自的作用。
- 解释记忆块模式：Human 块、Persona 块和用户自定义块作为一等类型对象。
- 描述什么是睡眠时计算，为什么它位于关键路径之外，以及为什么它可以运行比主 Agent 更强的模型。
- 实现一个脚本化的双 Agent 循环：主 Agent 提供响应，睡眠时 Agent 在轮次之间整合块。

## 问题

MemGPT (第 07 课) 解决了虚拟内存的控制流。三个生产问题随之浮现：

1. **延迟。** 每个记忆操作都在关键路径上。如果 Agent 必须在用户等待时修剪、摘要或协调，尾部延迟会爆炸。
2. **记忆腐烂。** 写入不断累积。矛盾的事实保留。检索淹没在过时内容中。
3. **结构丢失。** 扁平的归档存储无法表达"Human 块始终在 prompt 中；Persona 块始终在 prompt 中；Task 块按会话切换。"

Letta (letta.com) 是 2026 年的重写。记忆块使结构显式化；睡眠时计算将整合移出关键路径。

## 概念

### 三个层级

| 层级 | 范围 | 所在位置 | 写入者 |
|------|------|---------|--------|
| Core | 始终可见 | 主 prompt 内部 | Agent 工具调用 + 睡眠时重写 |
| Recall | 对话历史 | 可检索 | 自动轮次记录 |
| Archival | 任意事实 | 向量 + KV + 图 | Agent 工具调用 + 睡眠时摄入 |

Core 即 MemGPT 的核心。Recall 是对话缓冲区及其被驱逐的尾部。Archival 是外部存储。这种拆分清理了 MemGPT 的两层过载。

### 记忆块

块是 core 层级中一个有类型的、持久的、可编辑的区域。原始 MemGPT 论文定义了两个：

- **Human 块** — 关于用户的事实（姓名、角色、偏好、目标）。
- **Persona 块** — Agent 的自我概念（身份、语气、约束）。

Letta 将其推广到任意用户自定义块：用于当前目标的 `Task` 块、用于代码库事实的 `Project` 块、用于硬约束的 `Safety` 块。每个块有 `id`、`label`、`value`、`limit`（字符上限）、`description`（让模型知道何时编辑它）。

块可通过工具接口编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — 压缩接近上限的块。

### 睡眠时计算

2025 年 Letta 新增：在后台运行第二个 Agent，位于关键路径之外。睡眠时 Agent 处理对话转录和代码库上下文，将 `learned_context` 写入共享块，并整合或失效归档记录。

由此产生的特性：

- **无延迟成本。** 主响应不等待记忆操作。
- **允许更强的模型。** 睡眠时 Agent 可以使用更昂贵、更慢的模型，因为它不受延迟约束。
- **自然的整合窗口。** 在用户不等待时去重、摘要、失效矛盾事实。

这种形态与人类的工作方式匹配：你完成任务，睡一觉，长期记忆在夜间沉淀。

### Letta V1 与原生推理

Letta V1 (`letta_v1_agent`, 2026) 弃用了 `send_message`/heartbeat 和内联的 `Thought:` token，转而使用原生推理。Responses API (OpenAI) 和带扩展思考的 Messages API (Anthropic) 在单独的通道上发出推理，跨轮次传递（生产环境中跨提供商加密）。控制循环仍然是 ReAct。思维追踪是结构化的，而非 prompt 形态的。

### 这个模式哪里会出错

- **块膨胀。** 无限的 `block_append` 会迅速达到上限。在写入会超出上限之前接入块摘要器。
- **静默漂移。** 睡眠时 Agent 重写了一个块，主 Agent 从未察觉。对块进行版本控制，并在追踪中展示差异。
- **被污染的整合。** 睡眠时 Agent 将攻击者可触及的内容处理进 core。第 27 课同样适用于睡眠时接口。

## 构建

`code/main.py` 实现：

- `Block` — id、label、value、limit、description。
- `BlockStore` — CRUD + `near_limit(label)` 辅助方法。
- 两个脚本化 Agent — `PrimaryAgent` 服务一个轮次，`SleepTimeAgent` 在轮次之间整合。
- 追踪显示三轮对话（含块写入），外加一次睡眠时传递，摘要了一个块并使一个过时事实失效。

运行：

```
python3 code/main.py
```

转录显示了分工：主轮次快速并产生原始写入；睡眠传递进行压缩和清理。

## 使用

- **Letta** (letta.com) 作为参考实现。自托管或托管云。
- **Claude Agent SDK skills** 作为块形知识——技能是一个命名的、版本化的、可检索的指令块，Agent 按需加载。
- **自定义构建** 适用于希望控制存储后端的团队。使用 Letta API 契约以便后续迁移。

## 交付

`outputs/skill-memory-blocks.md` 为任何运行时生成 Letta 形态的块系统，包含睡眠时钩子、安全规则和引用连接。

## 练习

1. 添加一个 `block_summarize` 工具，当 `near_limit` 返回 true 时用模型生成的摘要替换块值。哪个触发阈值能最小化摘要调用次数和块溢出？
2. 在归档上实现睡眠时去重：文本 token 重叠 >90% 的两条记录合并为一条。仅在睡眠传递中执行，绝不在关键路径上。
3. 对块进行版本控制。每次写入时记录旧值和差异。暴露 `block_history(label)` 以便运维人员调试"为什么 Agent 忘记了 X"。
4. 将睡眠时 Agent 视为不可信的写入者。当它们触及 Persona 或 Safety 块时，要求第二个 Agent 审查后再提交。
5. 将示例移植到 Letta API (`letta_v1_agent`)。块 schema 有什么变化？原生推理如何改变追踪形态？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 记忆块 (Memory block) | "可编辑的 prompt 区域" | core 记忆中有类型的、持久的、LLM 可编辑的段 |
| Human 块 | "用户记忆" | 关于用户的事实，固定在 core 中 |
| Persona 块 | "Agent 身份" | 自我概念、语气、约束，固定在 core 中 |
| 睡眠时计算 (Sleep-time compute) | "异步记忆工作" | 第二个 Agent 在关键路径之外进行整合 |
| Core / Recall / Archival | "层级" | 三层记忆拆分：始终可见 / 对话 / 外部 |
| 块上限 (Block limit) | "容量上限" | 每个块的字符上限；强制进行摘要 |
| 原生推理 (Native reasoning) | "思维通道" | 提供商级别的推理输出，而非 prompt 级别的 `Thought:` |
| 学习到的上下文 (Learned context) | "睡眠输出" | 睡眠时 Agent 写入共享块的事实 |

## 扩展阅读

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — 块模式
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — 异步整合
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — 原生推理重写
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — 起源
