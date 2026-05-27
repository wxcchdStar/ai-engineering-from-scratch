# FIPA/ACL 传统——多智能体系统并非始于 LLM

> 多智能体系统（Multi-Agent System, MAS）在 LLM 出现之前已有 30 年的历史。FIPA（Foundation for Intelligent Physical Agents，智能物理智能体基金会）在 1996-2005 年间标准化了智能体通信语言（Agent Communication Language, ACL）、交互协议（Interaction Protocols）和智能体管理。2026 年的每个交接（handoff）、群聊（group chat）和任务市场（task market）都直接继承自 FIPA 规范。Contract Net Protocol（合同网协议，Smith 1980，FIPA 1996）是任务分配（task allocation）的参考协议。本课程将 2026 年的多智能体模式映射回它们的 FIPA 祖先，这样你就能看出哪些是真正的新东西，哪些是换了新包装的旧概念。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 第 16 阶段 · 01（为什么是多智能体）
**时间：** ~45 分钟

## 问题

2026 年的多智能体框架文档读起来像是它们发明了交接、群聊和任务市场。它们没有。FIPA 在 1996 年就标准化了这些概念。不了解这段历史，你会把 30 年前的协议重新发明一遍，并错过那些已经解决了你正在遇到的问题的文献。

## 概念

### FIPA 标准化了什么

FIPA 在 1996-2005 年间产生了四个持久的标准：

1.  **智能体通信语言（ACL, Agent Communication Language）。** 20 种通信行为（communicative acts）：`inform`、`request`、`agree`、`refuse`、`propose`、`cfp`（call for proposal，招标）、`accept-proposal`、`reject-proposal` 等。每条消息都有发送者、接收者和内容。2026 年的交接本质上是带有 `request` 语义的 ACL 消息。

2.  **交互协议（Interaction Protocols, IPs）。** 用于多步对话的预制序列。Contract Net IP（`fipa-contract-net`）：管理者（manager）广播 `cfp`，竞标者（bidders）回复 `propose`，管理者发送 `accept-proposal` / `reject-proposal`。2026 年的任务市场正是这个协议。

3.  **智能体管理（Agent Management）。** 智能体平台（Agent Platform, AP）、目录促进器（Directory Facilitator, DF——黄页服务）、智能体管理系统（Agent Management System, AMS）。智能体向 DF 注册其服务；其他智能体查询 DF 以找到它们。2026 年的 A2A Agent Card（智能体卡片）发现机制正是 DF 的现代版本。

4.  **传输（Transport）。** 智能体间消息的 HTTP 和 IIOP 绑定。2026 年的 A2A 使用 HTTP + JSON；FIPA 使用 HTTP + ACL XML。传输层变了；语义层没变。

### 2026 年模式 → FIPA 祖先映射

| 2026 年模式 | FIPA 祖先 | 变化 |
|-------------|----------|------|
| 交接（Handoff, Swarm） | ACL `request` | 工具调用替代了 XML 消息 |
| 群聊（GroupChat） | 多播 ACL 消息 | LLM 选择器（selector）替代了固定轮转 |
| 任务市场（Task Market） | Contract Net IP | 相同的 `cfp` → `propose` → `accept` 流程 |
| A2A Agent Card | 目录促进器（DF） | `/.well-known/agent.json` 替代了 FIPA DF |
| 监督者（Supervisor） | ACL `request` + `inform` | 相同的委托（delegation）模式 |
| 共享内存（Shared Memory） | 黑板架构（Blackboard Architecture, Hayes-Roth 1985） | JSON 替代了结构化令牌 |

### 为什么这很重要

1.  **你不会重新发明 Contract Net。** 当你的系统需要任务分配时，Contract Net 已经解决了竞标、授予和拒绝的协议。直接使用它。
2.  **ACL 的通信行为是经过实战检验的。** `inform`、`request`、`propose`、`refuse` 涵盖了 80% 的智能体间通信。不要为这些发明新的动词。
3.  **FIPA 的故障模式是有文档记录的。** 竞标者共谋、管理者瓶颈、消息丢失——FIPA 文献记录了所有这些。2026 年的系统会重新遇到它们。

### FIPA 没有解决什么

FIPA 假设智能体是确定性的、基于规则的推理器。它没有解决：

-   **随机输出。** 两个相同的 LLM 调用产生不同的答案。FIPA 假设相同的输入产生相同的输出。
-   **上下文窗口。** FIPA 智能体拥有无限记忆。LLM 智能体在 200k token 后就会遗忘。
-   **幻觉（Hallucination）。** FIPA 智能体不会编造事实。LLM 智能体会。
-   **自然语言语义。** ACL 消息具有精确的语义。LLM 交接是自然语言，接收智能体必须解释。

2026 年的多智能体工程是 FIPA 的结构加上 LLM 的随机性。本阶段的其余部分正是关于如何管理这种组合。

## 构建它

`code/main.py` 实现了一个最小化的 Contract Net：一个管理者广播 `cfp`，三个竞标者回复 `propose`，管理者授予任务。输出显示完整的消息序列。将其与第 16 阶段 · 16（谈判与讨价还价，Negotiation and Bargaining）中的 LLM 驱动版本进行比较。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-fipa-mapper.md` 读取任何多智能体框架的 API 文档，并返回 FIPA 等价映射。在新框架发布时运行它，以便在深入文档之前获得一个段落的理解。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| FIPA | "旧的多智能体标准" | 1996-2005 年 IEEE 标准化组织；ACL、交互协议、智能体管理。 |
| ACL | "智能体通信语言" | 20 种通信行为（`inform`、`request`、`cfp` 等）。2026 年交接的祖先。 |
| Contract Net | "任务市场协议" | Smith 1980，FIPA 1996。`cfp` → `propose` → `accept/reject`。 |
| 目录促进器（DF） | "黄页服务" | 智能体注册服务；其他智能体查询以找到它们。A2A Agent Card 的祖先。 |
| 通信行为 | "消息类型" | ACL 动词，具有精确的发送者/接收者/内容语义。 |
| 交互协议 | "对话模板" | 多步智能体对话的预制序列。 |

## 扩展阅读

-   [FIPA ACL 规范](http://www.fipa.org/specs/fipa00061/) — 20 种通信行为及其语义
-   [FIPA Contract Net 交互协议](http://www.fipa.org/specs/fipa00029/) — 任务市场的参考协议
-   [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — 经典机制，IEEE Transactions on Computers
-   [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — 共享内存的祖先
