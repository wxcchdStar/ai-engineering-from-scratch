# 为什么是多智能体系统（Multi-Agent System）？

> 2026 年，多智能体系统（Multi-Agent System, MAS）不再是可选的。Anthropic 的研究系统（Research System）在内部研究评估中比单智能体 Opus 4 高出 +90.2%，但每个查询消耗 15 倍的 token。Cemri 等人（MAST, arXiv:2503.13657）分析了 1642 条执行轨迹，发现 7 个最先进的开源 MAS 的失败率在 41% 到 86.7% 之间。多智能体系统能赢，但也会以可预测的方式失败。本阶段涵盖 25 节课，涵盖从 FIPA 传统到 2026 年最前沿的内容：原语（primitives）、拓扑（topologies）、共识（consensus）、谈判（negotiation）、生成式智能体（generative agents）、心智理论（Theory of Mind）、群体优化（swarm optimization）、MARL、智能体经济（agent economies）、生产规模化（production scaling）、故障分类法（failure taxonomy）以及基准测试（benchmarks）。到本阶段结束时，你将能够阅读任何多智能体框架文档，在一个段落内理解它，并知道它会在哪里出问题。

**类型：** 学习
**语言：** —
**前置条件：** 第 14 阶段（智能体工程，Agent Engineering）
**时间：** ~30 分钟

## 问题

单智能体系统存在上下文窗口限制。一个智能体在 200k token 的窗口中只能容纳有限数量的文档、工具输出和推理步骤。对于研究任务、复杂编码任务或多步骤工作流，单智能体会在步骤 5 之前忘记步骤 1。

多智能体系统通过将工作分配给多个智能体来解决这个问题，每个智能体拥有自己的上下文窗口。但这也引入了单智能体系统中不存在的新故障模式：协调开销（coordination overhead）、状态漂移（state drift）、记忆污染（memory poisoning）、群体思维（groupthink）和级联故障（cascading failures）。

## 概念

### 多智能体系统何时胜出

多智能体系统在以下情况下优于单智能体系统：

1.  **任务可分解。** 研究问题可以拆分为子问题；代码库可以按模块拆分；工作流可以按阶段拆分。
2.  **上下文是瓶颈。** 单智能体在 200k token 后就会遗忘。多智能体系统为每个子任务提供全新的上下文。
3.  **专业化（Specialization）有帮助。** 规划器（planner）加编码器（coder）加审查器（reviewer）胜过三个通用智能体。
4.  **验证（Verification）是独立的。** 一个独立的验证器（verifier）能捕捉到生成器（generator）遗漏的错误。

### 多智能体系统何时失败

MAST 分类法（Cemri 等人，2025）将故障分为三个根本原因：

1.  **规范问题（Specification Problems）——41.77%。** 角色模糊、任务定义不清晰、成功标准不明确。
2.  **协调故障（Coordination Failures）——36.94%。** 通信中断、状态不同步、消息丢失。
3.  **验证缺口（Verification Gaps）——21.30%。** 输出未经独立检查就被接受。

### 2026 年的成本现实

多智能体系统消耗大量 token。Anthropic 的研究系统每个查询使用 15 倍的 token。多智能体系统在准确性上胜出，但在成本上失败——这种权衡是一个商业决策，而非技术决策。

### 本阶段的结构

25 节课分为五个部分：

-   **基础（01-04）：** 为什么是多智能体、FIPA/ACL 传统、通信协议（Communication Protocols）、原语模型（Primitive Model）。
-   **拓扑（05-11）：** 监督者（Supervisor）、层级结构（Hierarchical）、心智社会/辩论（Society of Mind/Debate）、角色专业化（Role Specialization）、群体网络（Swarm Networks）、群聊（GroupChat）、交接（Handoffs）。
-   **协调（12-18）：** A2A 协议、共享内存（Shared Memory）、共识与 BFT、投票与辩论拓扑（Voting and Debate Topology）、谈判（Negotiation）、生成式智能体（Generative Agents）、心智理论（Theory of Mind）。
-   **高级主题（19-22）：** 群体优化（Swarm Optimization）、MARL、智能体经济（Agent Economies）、生产规模化（Production Scaling）。
-   **评估（23-25）：** 故障模式（Failure Modes）、基准测试（Benchmarks）、案例研究（Case Studies）。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| MAS | "多智能体系统" | 多个 LLM 智能体在共享环境中协调工作。 |
| 单智能体 | "一个模型完成所有工作" | 一个 LLM 调用工具并推理；受上下文窗口限制。 |
| 上下文窗口 | "模型一次能看到的量" | 单智能体的硬性限制；多智能体通过并行化绕过它。 |
| 协调开销 | "组织成本" | 规划、交接和综合所消耗的 token。 |
| MAST | "2026 年的故障分类法" | Cemri 2025；3 个根本原因类别 + 14 个子类型。 |
| Token 倍数 | "多智能体消耗更多" | 多智能体系统每个任务使用的 token 是单智能体的 3-15 倍。 |

## 扩展阅读

-   [Cemri 等人 — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分类法，NeurIPS 2025
-   [Anthropic 工程 — 我们如何构建多智能体研究系统](https://www.anthropic.com/engineering/multi-agent-research-system) — 生产级监督者-工作者（supervisor-worker）参考案例
-   [LangGraph 工作流与智能体](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 2026 年生产级默认框架
-   [CrewAI 文档](https://docs.crewai.com/en/introduction) — 基于角色的多智能体框架
