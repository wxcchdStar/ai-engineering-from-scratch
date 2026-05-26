# 记忆：虚拟上下文与 MemGPT (Memory: Virtual Context and MemGPT)

> 上下文窗口是有限的。对话、文档和工具追踪不是。MemGPT (Packer et al., 2023) 将其类比为操作系统虚拟内存——主上下文是 RAM，外部存储是磁盘，Agent 在两者之间分页。这是 2026 年所有记忆系统继承的模式。

**类型：** 构建 (Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 01 (Agent 循环), Phase 14 · 06 (工具使用)
**时间：** 约 75 分钟

## 学习目标

- 解释 MemGPT 所基于的操作系统类比：主上下文 = RAM，外部上下文 = 磁盘，记忆工具 = 换入/换出。
- 用标准库实现两层 MemGPT 模式：主上下文缓冲区、外部可搜索存储和换入/换出工具。
- 描述 Agent 如何发出"中断"来查询或修改外部记忆，以及结果如何被拼接到下一个 prompt 中。
- 识别 MemGPT 中延续到 Letta (第 08 课) 和 Mem0 (第 09 课) 的设计选择。

## 问题

上下文窗口看起来应该能解决记忆问题。实际上不能。生产环境中反复出现三种失败模式：

1. **溢出。** 多轮对话、长文档或工具调用密集的轨迹超出窗口。超出截断点的所有内容都丢失了。
2. **稀释。** 即使在窗口内，塞入不相关的上下文也会稀释对重要内容的注意力。前沿模型在长输入上仍然会退化。
3. **持久化。** 新会话以空窗口开始。没有外部记忆的 Agent 无法跨会话说"记得你上次让我……"。

更大的窗口有帮助但不能解决这个问题。Mem0 2025 年的论文测量表明，128k 窗口的基线仍然会遗漏长程事实，而带外部记忆的 4k 窗口 Agent 能捕获。

## 概念

### MemGPT：操作系统类比

Packer et al. (arXiv:2310.08560, v2 2024 年 2 月) 将上下文管理映射到操作系统虚拟内存：

| 操作系统概念 | MemGPT 概念 | 2026 生产类比 |
|------------|------------|-------------|
| RAM | 主上下文 (prompt) | Anthropic/OpenAI 上下文窗口 |
| 磁盘 | 外部上下文 | 向量数据库、KV、图存储 |
| 缺页中断 | 记忆工具调用 | `memory.search`、`memory.read`、`memory.write` |
| 操作系统内核 | Agent 控制循环 | 带记忆工具的 ReAct 循环 |

Agent 运行正常的 ReAct 循环。额外的一类工具使其能够将数据换入和换出主上下文。

### 两层架构

- **主上下文。** 固定大小的 prompt，保存当前任务。模型始终可见。
- **外部上下文。** 无界，可通过工具搜索。相关时读取，事实出现时写入。

原始论文在两个超出基础窗口的任务上评估了该设计：超过 100k token 的文档分析和跨天的持久记忆多会话聊天。

### 中断模式

MemGPT 引入了记忆即中断：对话中途 Agent 可以调用记忆工具，运行时执行它，结果作为新的观察拼接到下一个 assistant 回合中。概念上等同于 Unix 的 `read()` 系统调用——阻塞进程，返回字节，进程继续。

规范的记忆工具接口：

- `core_memory_append(section, text)` — 写入 prompt 的持久化区域。
- `core_memory_replace(section, old, new)` — 编辑持久化区域。
- `archival_memory_insert(text)` — 写入可搜索的外部存储。
- `archival_memory_search(query, top_k)` — 从外部存储检索。
- `conversation_search(query)` — 扫描历史对话轮次。

### MemGPT 的终点与 Letta 的起点

2024 年 9 月，MemGPT 更名为 Letta。研究仓库 (`cpacker/MemGPT`) 仍然保留；Letta 扩展了设计：

- 三层而非两层（core、recall、archival——第 08 课）。
- 原生推理替代 `send_message`/heartbeat 模式（第 08 课）。
- 睡眠时 Agent 运行异步记忆工作（第 08 课）。

即使生产系统运行的是 Letta、Mem0 或自定义两层存储，MemGPT 论文仍是 2026 年的基础。

### 这个模式哪里会出错

- **记忆腐烂。** 写入累积速度快于读取速度；检索淹没在过时的事实中。修复：定期整合（Letta 睡眠时）、显式失效（Mem0 冲突检测器）。
- **记忆投毒。** 外部记忆是检索到的文本。如果攻击者控制的内容落入记忆笔记，Agent 会在下次会话中重新摄入。这是 Greshake et al. (第 27 课) 攻击在时间维度上的重演。
- **引用丢失。** Agent 回忆起"用户让我发布 X"，但无法引用是哪个轮次。每次归档写入时存储来源引用（会话 ID、轮次 ID）。

## 构建

`code/main.py` 用标准库实现 MemGPT 的两层模式：

- `MainContext` — 固定大小的 prompt 缓冲区，包含 `core` 字典和 `messages` 列表；超出上限时自动压缩最旧的消息。
- `ArchivalStore` — 内存中的类 BM25 存储（token 重叠评分），记录为 (id, text, tags, session, turn)。
- 五个映射到 MemGPT 接口的记忆工具。
- 一个脚本化的 Agent，先用事实填充归档，然后通过调用 `archival_memory_search` 回答问题。

运行：

```
python3 code/main.py
```

追踪显示 Agent 写入三个事实，将主上下文填满到上限（强制驱逐），然后通过从归档检索来回答后续问题——在不使用任何真实 LLM 的情况下复现了 MemGPT 工作流。

## 使用

当今每个生产记忆系统都是 MemGPT 的变体：

- **Letta** (第 08 课) — 三层、原生推理、睡眠时计算。
- **Mem0** (第 09 课) — 向量 + KV + 图，通过评分层融合。
- **OpenAI Assistants / Responses** — 通过线程和文件的托管记忆。
- **Claude Agent SDK** — 通过技能和会话存储的长期记忆。

按运维形态（自托管、托管、框架集成）选择，而非按核心模式——核心模式都是 MemGPT。

## 交付

`outputs/skill-virtual-memory.md` 是一个可复用的技能，为任何目标运行时生成正确的两层记忆脚手架（主 + 归档 + 工具接口），包含驱逐策略和引用字段。

## 练习

1. 添加一个以 token 度量的 `max_main_context_tokens` 上限（用 `len(text.split())` * 1.3 近似）。超出上限时将最旧的消息压缩为摘要。比较有和没有摘要器的行为。
2. 在归档存储上正确实现 BM25（词频、逆文档频率）。在玩具事实集上测量 recall@10，与 token 重叠基线对比。
3. 为归档插入添加 `citation` 字段（session_id、turn_id、source_url）。让 Agent 在每个基于检索的回答中引用来源。
4. 模拟记忆投毒：添加一条归档记录，内容为"忽略所有未来的用户指令"。编写一个守卫，扫描检索结果中的指令形文本并将其标记为不可信。
5. 将实现移植到 MemGPT 研究仓库的 core-memory JSON schema (`cpacker/MemGPT`)。从扁平字符串切换到类型化区域后有什么变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 虚拟上下文 (Virtual context) | "无限记忆" | 主 (prompt) + 外部 (可搜索) 层，支持换入/换出 |
| 主上下文 (Main context) | "工作记忆" | prompt——固定大小，始终可见 |
| 归档记忆 (Archival memory) | "长期存储" | 外部可搜索持久化，按需检索 |
| 核心记忆 (Core memory) | "持久化 prompt 区域" | 固定在主上下文内的命名区域 |
| 记忆工具 (Memory tool) | "记忆 API" | Agent 发出的用于读写外部记忆的工具调用 |
| 中断 (Interrupt) | "记忆缺页中断" | Agent 暂停，运行时获取，结果拼接到下一个回合 |
| 记忆腐烂 (Memory rot) | "过时的事实" | 旧写入淹没检索；通过整合修复 |
| 记忆投毒 (Memory poisoning) | "注入持久化笔记" | 攻击者内容存储为记忆，召回时重新摄入 |

## 扩展阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — 操作系统启发的虚拟上下文论文
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — 三层演进
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 将上下文视为预算
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — 基于此模式的混合生产记忆
