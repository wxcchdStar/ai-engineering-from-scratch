# Phase 14 翻译任务

## 目标

将 `phases/14-agent-engineering/` 下所有 `docs/en.md` 翻译为简体中文，输出为同目录下的 `docs/zh.md`。

## 已完成

| # | Lesson | 状态 |
|---|--------|------|
| 01 | The Agent Loop | ✅ |
| 02 | ReWOO: Plan-and-Execute | ✅ |
| 03 | Reflexion: Verbal RL | ✅ |
| 04 | Tree of Thoughts / LATS | ✅ |
| 05 | Self-Refine and Critic | ✅ |
| 06 | Tool Use and Function Calling | ✅ |
| 07 | Memory: Virtual Context (MemGPT) | ✅ |
| 08 | Memory: Blocks, Sleep-Time Compute | ✅ |
| 09 | Hybrid Memory (Mem0) | ✅ |
| 10 | Skill Libraries (Voyager) | ✅ |
| 11 | Planning: HTN and Evolutionary | ✅ |
| 12 | Anthropic Workflow Patterns | ✅ |
| 13 | LangGraph: Stateful Graphs | ✅ |
| 14 | AutoGen: Actor Model | ✅ |
| 15 | CrewAI: Role-based Crews | ✅ |
| 16 | OpenAI Agents SDK | ✅ |
| 17 | Claude Agent SDK | ✅ |
| 18 | Agno and Mastra Runtimes | ✅ |
| 19 | Benchmarks: SWE-bench / GAIA | ✅ |
| 20 | Benchmarks: WebArena / OSWorld | ✅ |
| 21 | Computer-Use Agents | ✅ |
| 22 | Voice Agents: Pipecat / LiveKit | ✅ |
| 23 | OTel GenAI Conventions | ✅ |
| 24 | Agent Observability Platforms | ✅ |
| 25 | Multi-Agent Debate | ✅ |
| 26 | Failure Modes (Agentic) | ✅ |
| 27 | Prompt Injection Defense | ✅ |
| 28 | Orchestration Patterns | ✅ |
| 29 | Production Runtimes | ✅ |
| 30 | Eval-Driven Agent Development | ✅ |
| 31 | Agent Workbench: Why Models Fail | ✅ |
| 32 | Minimal Agent Workbench | ✅ |
| 33 | Instructions as Executable Constraints | ✅ |
| 34 | Repo Memory and State | ✅ |
| 35 | Initialization Scripts | ✅ |
| 36 | Scope Contracts | ✅ |
| 37 | Runtime Feedback Loops | ✅ |
| 38 | Verification Gates | ✅ |
| 39 | Reviewer Agent | ✅ |
| 40 | Multi-Session Handoff | ✅ |
| 41 | Workbench for Real Repos | ✅ |
| 42 | Agent Workbench Capstone | ✅ |

## 翻译规范

- **保持不变**：Markdown 格式、SVG 图表、Python/JSON/TypeScript 代码块、链接、论文标题/作者名/URL、quiz.json 内容
- **仅翻译**：代码块外的英文叙述性文本
- **术语处理**：首次出现用"中文 (English)"，后续直接用中文；API 名称、框架名、模型名保持英文
- **"What people say / What it actually means" 表格**：左列口语化，右列准确释义

## 项目约束

- 基础路径：`phases/14-agent-engineering/`
- 输入：`{NN-lesson-name}/docs/en.md`
- 输出：`{NN-lesson-name}/docs/zh.md`
- 按序号逐一翻译
