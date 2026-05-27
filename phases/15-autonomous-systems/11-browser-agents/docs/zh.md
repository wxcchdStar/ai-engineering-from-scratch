# 浏览器智能体与长周期 Web 任务

> ChatGPT agent（2025 年 7 月）将 Operator 和深度研究合并为一个浏览器/终端智能体，并在 BrowseComp 上达到 SOTA 68.9%。OpenAI 于 2025 年 8 月 31 日关闭了 Operator——产品层的整合。Anthropic 的 Vercept 收购将 Claude Sonnet 在 OSWorld 上从低于 15% 提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的假阴性率，并发布了 258 个任务的 Hard 子集。数字是真实的。攻击面也是真实的：OpenAI 的准备负责人公开表示，对浏览器智能体的间接提示注入"不是一个可以完全修补的 bug"。有记录的 2025-2026 年攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）和 Perplexity Comet 中的一键劫持。

**类型：** 学习
**语言：** Python（标准库，间接提示注入攻击面模型）
**前置条件：** 第 15 阶段 · 10（权限模式），第 15 阶段 · 01（长周期智能体）
**时间：** 约 45 分钟

## 问题

浏览器智能体是一个读取不受信任内容并执行后果性操作的长周期智能体。智能体访问的每个页面都是用户未编写的输入。每个页面上的每个表单都是一个潜在的命令通道。2025-2026 年的攻击语料库显示这不是假设性的：Tainted Memories 让攻击者通过精心制作的页面将恶意指令绑定到智能体的记忆中；HashJack 将命令隐藏在智能体访问的 URL 片段中；Perplexity Comet 劫持在一次点击中命中。

防御图景令人不安。OpenAI 的准备负责人说出了安静的部分：间接提示注入"不是一个可以完全修补的 bug"。这是因为攻击存在于智能体的读取与行动边界中，这在架构上是模糊的——模型读取的每个 token 原则上都可以被读取为指令。

本课命名攻击面，命名基准格局（BrowseComp、OSWorld、WebArena-Verified），并建模一个最小的间接提示注入场景，以便你可以在第 14 和 18 课中推理真正的防御。

## 概念

### 2026 年格局，每个系统一段话

**ChatGPT agent（OpenAI）。** 2025 年 7 月发布。统一了 Operator（浏览）和 Deep Research（多小时研究）。于 2025 年 8 月 31 日关闭了独立的 Operator。BrowseComp SOTA 68.9%；在 OSWorld 和 WebArena-Verified 上表现强劲。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 的 Vercept 收购专注于计算机使用能力。将 Claude Sonnet 在 OSWorld 上从 <15% 提升到 72.5%。Claude Computer Use 作为工具 API 发布。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use 集成发布计算机使用控制；FSF v3（2026 年 4 月，第 20 课）专门跟踪 ML 研发领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个有据可查的问题：原始 WebArena 有约 11.3% 的假阴性率（标记为失败但实际已解决的任务）。Verified 版本使用人工策划的成功标准重新评分，并添加了 258 个任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准 | 测量什么 | 周期 |
|---|---|---|
| BrowseComp | 在时间压力下在开放 Web 上查找特定事实 | 分钟 |
| OSWorld | 智能体操作完整桌面（鼠标、键盘、shell） | 数十分钟 |
| WebArena-Verified | 在模拟站点中的事务性 Web 任务 | 分钟 |
| Hard 子集 | 具有多页面状态转换的 WebArena-Verified 任务 | 数十分钟 |

不同的轴。高 BrowseComp 分数表示智能体找到事实；它不表示智能体可以预订航班。OSWorld 分数更接近"它能在我的桌面上工作吗"。WebArena-Verified 更接近"它能完成一个流程吗"。任何生产决策都需要匹配任务分布的基准。

### 攻击面，命名

1. **间接提示注入。** 不受信任的页面内容包含指令。智能体读取它们。智能体执行它们。公开示例：2024 Kai Greshake 等人，2025 Tainted Memories 论文，2026 HashJack（Cato Networks）。
2. **URL 片段 / 查询注入。** 爬取 URL 的 `#fragment` 或查询字符串包含命令。从不以可见方式渲染；仍在智能体的上下文中。
3. **内存绑定攻击。** 页面指示智能体写入持久内存（第 12 课涵盖持久状态）。下一个会话，内存在没有可见触发器的情况下触发负载。
4. **对已认证会话的 CSRF 形状攻击。** Tainted Memories 类别：智能体在某处已登录；攻击者的页面发出智能体使用用户 cookie 执行的状态更改请求。
5. **一键劫持。** 一个视觉上无害的按钮搭载智能体跟随的负载。Comet 类别。
6. **智能体主机面的内容安全策略漏洞。** 渲染和工具层本身可以是攻击向量；浏览器中的浏览器智能体栈是宽的。

### 为什么"不能完全修补"

攻击与智能体的能力同构。智能体必须读取不受信任的内容来完成其工作。智能体读取的任何内容都可能包含指令。智能体遵循的任何指令都可能与用户的实际请求不对齐。防御（信任边界、分类器、工具允许列表、后果性操作的 HITL）提高了攻击成本并减少了爆炸半径。它们不闭合该类别。

这与 Lob 定理（第 8 课）的推理模式相同：智能体不能证明下一个 token 是安全的；它只能建立一个使不安全 token 更可检测的系统。

### 实际交付的防御姿态

- **读取 / 写入边界。** 读取从不具有后果性。写入（提交表单、发布内容、调用具有副作用的工具）如果发起内容来自信任边界之外，则需要新的人类批准。
- **每个任务的工具允许列表。** 智能体可以浏览；除非该工具被明确为该任务启用，否则它不能发起电汇。第 13 课涵盖预算。
- **会话隔离。** 浏览器智能体会话仅使用限定范围的凭证运行。没有生产认证，没有个人电子邮件。每个 HTTP 请求的日志保留用于审计。
- **内容消毒器。** 获取的 HTML 在连接到模型上下文之前被剥离已知不良模式。（减少简单攻击；不能阻止复杂的负载。）
- **后果性操作的 HITL。** 提议-然后-提交模式（第 15 课）。
- **内存上的金丝雀令牌。** 如果内存条目触发，用户看到它（第 14 课）。

## 使用它

`code/main.py` 对三个合成页面建模一个微型浏览器智能体运行。一个页面是良性的，一个在可见文本中有直接提示注入块，一个有 URL 片段注入（不可见但在智能体的上下文中）。脚本显示（a）天真的智能体会做什么，（b）读取/写入边界捕获什么，（c）消毒器捕获什么，（d）两者都未捕获什么。

## 交付它

`outputs/skill-browser-agent-trust-boundary.md` 界定一个提议的浏览器智能体部署：它触及哪些信任区域，它被授权写入什么，以及在第一次运行之前必须就位的防御措施。

## 练习

1. 运行 `code/main.py`。确定消毒器捕获但读取/写入边界未捕获的攻击，以及只有读取/写入边界捕获的攻击。

2. 扩展消毒器以检测一类 HashJack 风格的 URL 片段注入。测量在具有合法片段的良性 URL 上的误报率。

3. 选择一个你了解的真实浏览器智能体工作流（例如"预订航班"）。列出每个读取和每个写入。标记哪些写入需要 HITL 以及为什么。

4. 阅读 WebArena-Verified ICLR 2026 论文。确定原始 WebArena 评分不可靠的一类任务，并解释 Verified 子集如何解决它。

5. 为浏览器智能体设置设计一个内存金丝雀。你会存储什么，在哪里，什么触发警报？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 间接提示注入 | "不良页面文本" | 智能体读取的页面中的不受信任内容包含智能体执行的指令 |
| Tainted Memories | "内存攻击" | 智能体将攻击者提供的指令写入持久内存；下一个会话触发 |
| HashJack | "URL 片段攻击" | 隐藏在 URL 片段/查询字符串中的负载在智能体上下文中但不可见渲染 |
| 一键劫持 | "不良按钮" | 可见的交互元素搭载智能体执行的后续负载 |
| BrowseComp | "Web 搜索基准" | 在开放 Web 上查找特定事实；分钟级周期 |
| OSWorld | "桌面基准" | 完整操作系统控制；多步 GUI 任务 |
| WebArena-Verified | "修复的 Web 任务基准" | ServiceNow 重新评分的 WebArena，带有 Hard 子集 |
| 读取/写入边界 | "副作用门控" | 读取从不具有后果性；如果内容超出信任范围，写入需要新批准 |

## 扩展阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 和深度研究的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 谱系和成为 ChatGPT agent 的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始基准。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修复子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包括计算机使用智能体的攻击面讨论。
