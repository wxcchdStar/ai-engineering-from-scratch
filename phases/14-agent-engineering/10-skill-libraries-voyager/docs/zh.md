# 技能库与终身学习 (Skill Libraries and Lifelong Learning, Voyager)

> Voyager (Wang et al., TMLR 2024) 将可执行代码视为技能。技能是命名的、可检索的、可组合的，并通过环境反馈进行优化。这是 Claude Agent SDK skills、skillkit 以及 2026 年技能库模式的参考架构。

**类型：** 构建 (Build)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta 块)
**时间：** 约 75 分钟

## 学习目标

- 列举 Voyager 的三个组件——自动课程、技能库、迭代提示——及其各自的作用。
- 解释为什么 Voyager 将动作空间设为代码而非原始命令。
- 用标准库实现一个技能库，包含注册、检索、组合和失败驱动的优化。
- 将 Voyager 模式映射到 2026 年的 Claude Agent SDK skills 和 skillkit 生态系统。

## 问题

在每个会话中从头构建所有能力的 Agent 犯了三个错误：

1. **浪费 token。** 每个任务都重新引出相同的推理。
2. **丢失进展。** 在会话 A 中学到的修正不会传递到会话 B。
3. **在长程组合上失败。** 复杂任务需要能力层级；一次性 prompt 无法表达它们。

Voyager 的答案：将每个可复用能力视为存储在库中的命名代码块，按相似度检索，可与其他技能组合，并通过执行反馈进行优化。

## 概念

### 三个组件

Voyager (arXiv:2305.16291) 围绕以下结构组织 Agent：

1. **自动课程 (Automatic curriculum)。** 一个好奇心驱动的提议器，基于 Agent 当前的技能集和环境状态选择下一个任务。探索是自底向上的。
2. **技能库 (Skill library)。** 每个技能是可执行代码。任务成功时添加新技能。技能按查询到描述的相似度检索。
3. **迭代提示机制 (Iterative prompting mechanism)。** 失败时，Agent 接收执行错误、环境反馈和自我验证输出，然后优化技能。

Minecraft 评估 (Wang et al., 2024)：相比基线，独特物品多 3.3 倍，石制工具快 8.5 倍，铁制工具快 6.4 倍，地图穿越距离长 2.3 倍。这些数字是 Minecraft 特定的，但模式是可迁移的。

### 动作空间 = 代码

大多数 Agent 发出原始命令。Voyager 发出 JavaScript 函数。一个技能是：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由子技能组合而成。按键值（描述和嵌入）存储。作为程序检索，而非 prompt。

这就是 2026 年 Claude Agent SDK skill：一个命名的、可检索的代码块加上 Agent 按需加载的指令。

### 技能检索

新任务"制作钻石镐"。Agent：

1. 嵌入任务描述。
2. 在技能库中查询 top-k 相似技能。
3. 检索 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 从检索到的原语 + 新逻辑组合出新技能。

这就是 MCP 资源 (Phase 13) 和 Agent SDK skills 实现的模式：在知识/代码表面上的检索，范围限定在当前任务。

### 迭代优化

Voyager 的反馈循环：

1. Agent 编写一个技能。
2. 技能在环境中运行。
3. 返回三种信号之一：`success`、`error`（带堆栈追踪）、`self-verification failure`。
4. Agent 使用该信号作为上下文重写技能。
5. 循环直到成功或达到最大轮次。

这是 Self-Refine (第 05 课) 应用于代码生成，并带有环境锚定的验证。CRITIC (第 05 课) 是相同的模式，但以外部工具作为验证器。

### 课程与探索

Voyager 的课程模块基于 Agent 已有的和尚未完成的内容，提出诸如"在湖边建造一个庇护所"的任务。提议器使用环境状态 + 技能清单来选择刚好超出当前能力的任务——探索的最佳点。

对于生产 Agent，这转化为"缺少什么"操作符：给定当前技能库和一个领域，我们尚未覆盖哪些技能？团队通常以课程审查的形式手动实现这一点。

### 这个模式哪里会出错

- **技能库腐烂。** 同一个技能被添加了 10 次，描述略有不同。在写入时添加去重；检索只返回一个。
- **组合技能漂移。** 父技能依赖一个已被优化的子技能。对技能进行版本控制；固定在 v1 的父技能不会自动获取 v3。
- **检索质量。** 当库增长到几百个时，基于技能描述的向量检索会退化。用标签过滤和硬约束（"仅 `category=tooling` 的技能"）作为补充。

## 构建

`code/main.py` 用标准库实现一个技能库：

- `Skill` — name、description、code（字符串）、version、tags、dependencies。
- `SkillLibrary` — register、search（token 重叠）、compose（依赖的拓扑排序）和 refine（更新时版本递增）。
- 一个脚本化 Agent，注册三个原始技能，组合出第四个，遇到失败，然后优化。

运行：

```
python3 code/main.py
```

追踪显示库写入、检索、组合、一次失败执行和 v2 优化——Voyager 循环的端到端。

## 使用

- **Claude Agent SDK skills** (Anthropic) — 2026 年参考：每个技能有描述、代码和指令；在 Agent 会话期间按需加载。
- **skillkit** (npm: skillkit) — 面向 32+ AI 编程 Agent 的跨 Agent 技能管理。
- **自定义技能库** — 领域特定（数据 Agent 的 SQL 技能、基础设施 Agent 的 Terraform 技能）。Voyager 模式可以缩小应用。
- **OpenAI Agents SDK `tools`** — 在低端；每个工具是一个轻量级技能。

## 交付

`outputs/skill-skill-library.md` 生成一个 Voyager 形态的技能库，包含注册、检索、版本控制和优化，适用于任何目标运行时。

## 练习

1. 为 `compose()` 添加依赖循环检测器。当技能 A 依赖 B，B 又依赖 A 时会发生什么？报错还是警告？
2. 实现每个技能的版本固定。当父技能组合了子技能 `crafting@1` 时，对 `crafting@2` 的优化不能静默升级父技能。
3. 将 token 重叠检索替换为 sentence-transformers 嵌入（或 BM25 标准库实现）。在 50 个技能的玩具库上测量 retrieval@5。
4. 添加一个"课程"Agent：给定当前库和领域描述，提出 5 个缺失的技能。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK skill 文档。将玩具库移植到 SDK 的 skill schema。可发现性有什么变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 技能 (Skill) | "可复用能力" | 命名的代码块 + 描述，按相似度检索 |
| 技能库 (Skill library) | "Agent 的操作记忆" | 技能的持久化存储，可搜索和可组合 |
| 课程 (Curriculum) | "任务提议器" | 由当前能力差距驱动的自底向上目标生成器 |
| 组合 (Composition) | "技能 DAG" | 技能调用技能；执行时拓扑排序 |
| 迭代优化 (Iterative refinement) | "自我修正循环" | 环境反馈 + 错误 + 自我验证折叠回下一个版本 |
| 动作空间即代码 (Action-space-as-code) | "程序化动作" | 发出函数而非原始命令，实现时间扩展行为 |
| 写入时去重 (Dedup on write) | "技能合并" | 近似重复的描述合并为一个规范技能 |

## 扩展阅读

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — 原始技能库论文
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — 技能作为 2026 年的产品化
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 实践中的技能和子 Agent
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — Voyager 底层的优化循环
