# 群聊（Group Chat）与发言者选择（Speaker Selection）

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个智能体之间共享一个对话；一个选择器函数（LLM、轮转或自定义）选择谁下一个发言。这是涌现式多智能体对话的原型——智能体不知道自己在静态图中的角色，它们只是对共享池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中得以保留；AutoGen v0.4 将其重写为事件驱动的 actor 模型。Microsoft 于 2026 年 2 月将 AutoGen 置于维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（RC 2026 年 2 月）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中都得以保留——学习一次，到处使用。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 第 16 阶段 · 04（原语模型）
**时间：** ~60 分钟

## 问题

静态图（LangGraph）在工作流已知时很棒。真实的对话不是静态的：有时编码器询问审查者，有时询问研究者，有时询问写作者。硬编码每个可能的交接会产生边爆炸。你想要*智能体对共享池做出反应*，由某个函数决定谁下一个发言。

这正是 AutoGen GroupChat 所做的。

## 概念

### 形状

```
              ┌─── 共享池 ────┐
              │  m1  m2  m3...│
              └─────────┬─────┘
                        │ （每个人都读取全部）
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
   智能体 A 智能体 B  智能体 C  智能体 D  选择器
                                           │
                                           ▼
                                  "下一个发言者 = C"
```

每个智能体看到每条消息。每轮调用选择器函数来选择谁下一个发言。

### 三种选择器风格

**轮转（Round-robin）。** 固定循环。确定性。在 N 中线性扩展但忽略上下文——即使主题是法务审查，编码器也会获得发言权。

**LLM 选择。** 调用 LLM 读取最近的池并返回最佳的下一个发言者。上下文感知但慢：每轮增加一次 LLM 调用。AutoGen 的默认设置。

**自定义。** 一个带有你想要的任何逻辑的 Python 函数。典型：LLM 选择带有回退规则（例如，"总是在编码器之后给验证器发言权"）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当智能体完成一轮时，管理者调用选择器，选择器返回下一个智能体。循环继续直到终止条件。

### 终止

三种常见模式：

-   **最大轮数。** 总轮数的硬性上限。
-   **"TERMINATE" 令牌。** 智能体可以发出哨兵消息；管理者在出现时停止。
-   **目标达成检查。** 轻量级验证器每轮运行，完成时停止聊天。

### AutoGen → AG2 分裂与 Microsoft Agent Framework 合并

2025 年初，Microsoft 开始围绕事件驱动的 actor 模型对 AutoGen（v0.4）进行重大重写。社区将 AutoGen v0.2 的 GroupChat 语义分支为 AG2，保留了早期采用者已集成的 API。

2026 年 2 月，Microsoft 宣布 AutoGen 将进入维护模式，事件驱动的 actor 模型合并到 **Microsoft Agent Framework**（RC 2026 年 2 月，现已与 Semantic Kernel 合并）。GroupChat 概念在两个轨道中都得以保留；实现细节不同。AG2 是 v0.2 兼容代码的首选上游。

### GroupChat 何时适合

-   **涌现式对话。** 你不想预先连接每个可能的下一个发言者。
-   **角色混合任务。** 编码器询问研究者，研究者询问档案管理员，档案管理员回问编码器。流程不是 DAG。
-   **探索性问题解决。** 想想"头脑风暴会议"，而不是"装配线"。

### 何时失败

-   **严格确定性。** LLM 选择器可能不一致。相同的提示词，不同的运行，不同的下一个发言者。
-   **谄媚级联。** 智能体服从说话最自信的那个。显式反提示。
-   **上下文膨胀。** 每个智能体读取每条消息；10 轮后上下文巨大。使用投影（Projection，第 15 课）来限定视图范围。
-   **热门发言者。** 一个智能体主导对话，因为选择器偏爱其专长。引入发言者平衡作为选择器特性。

### 群聊 vs 监督者

相同的原语，不同的默认设置：

-   监督者：一个智能体规划，其他执行。选择器是"询问规划器该做什么"。
-   群聊：所有智能体是对等的；选择器是共享池上的函数。

两者都使用第 04 课的四个原语。群聊默认为 LLM 选择的编排和全量池共享状态。

## 构建它

`code/main.py` 用标准库从头实现了一个 GroupChat。三个智能体（编码器、审查者、管理者），轮转和 LLM 选择变体，以及在 `TERMINATE` 令牌上的终止。

演示打印对话记录以及两种变体的选择器决策跟踪。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat 选择器——轮转 vs LLM 选择 vs 自定义，以及使用什么选择器输入（最近消息、智能体专长、轮次计数）。

## 交付它

清单：

-   **最大轮数上限。** 始终设置。典型任务 10-20 轮。
-   **发言者平衡指标。** 跟踪每个智能体的轮次；当不平衡超过阈值时发出警报。
-   **终止令牌。** `TERMINATE` 或专用验证器智能体。
-   **投影或范围记忆。** 大约 10 条消息后，考虑只给每个智能体一个范围视图以防止上下文膨胀。
-   **选择器日志。** 对于 LLM 选择变体，记录选择器的输入和选择。否则调试是不可能的。

## 练习

1.  运行 `code/main.py`。比较轮转和 LLM 选择下的对话。每种情况下哪个智能体占主导？
2.  在选择器中添加"每个智能体最大发言次数"规则。它如何影响记录？
3.  实现目标达成终止：当审查者返回"approved"时停止。它在达到轮数上限之前触发的频率如何？
4.  阅读 AutoGen 稳定版文档中关于 GroupChat 的部分（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。识别 `GroupChatManager` 使用的默认选择器。
5.  阅读 AG2 仓库（https://github.com/ag2ai/ag2）并比较其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 添加了什么具体属性（吞吐量、容错、可组合性）？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| GroupChat | "智能体在一个聊天室" | 共享消息池 + 选择器函数。AutoGen / AG2 原语。 |
| 发言者选择 | "谁下一个发言" | 选择下一个智能体的函数。轮转、LLM 选择或自定义。 |
| GroupChatManager | "会议主持人" | AutoGen 组件，拥有选择器并在轮次上循环。 |
| ConversableAgent | "基础智能体" | AutoGen 基类；可以发送和接收消息的智能体。 |
| 终止令牌 | "'停止'词" | 结束聊天的哨兵字符串（通常是 `TERMINATE`）。 |
| 热门发言者 | "一个智能体占主导" | 故障模式，选择器不断选择同一个智能体。 |
| 上下文膨胀 | "池无限增长" | 每个智能体读取每条先前的消息；上下文随轮次增长。 |
| 投影 | "范围视图" | 共享池的按角色视图，以防止上下文膨胀。 |

## 扩展阅读

-   [AutoGen 群聊文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
-   [AG2 仓库](https://github.com/ag2ai/ag2) — 社区 AutoGen v0.2 延续
-   [Microsoft Agent Framework 文档](https://microsoft.github.io/agent-framework/) — 合并后的继任者，RC 2026 年 2 月
-   [AutoGen v0.4 发布说明](https://microsoft.github.io/autogen/stable/) — 事件驱动 actor 模型重写细节
