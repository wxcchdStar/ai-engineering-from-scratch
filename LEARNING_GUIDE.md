# 学习指南

## 定级测试结果

| 领域 | 得分 |
|------|------|
| 数学与统计 | 1/2 |
| 经典机器学习 | 1/2 |
| 深度学习 | 2/2 |
| NLP & Transformers | 2/2 |
| 应用 AI | 2/2 |
| **总分** | **8/10** |

**推荐起点：Phase 11: LLM Engineering**

---

## 个性化学习路径

| Phase | 名称 | 状态 | 预估时间 |
|-------|------|------|----------|
| 0 | Setup & Tooling | Skip | -- |
| 1 | Math Foundations | Review | 23 hr |
| 2 | ML Fundamentals | Review | 21 hr |
| 3 | Deep Learning Core | Skip | -- |
| 4 | Computer Vision | Skip | -- |
| 5 | NLP — Foundations to Advanced | Skip | -- |
| 6 | Speech & Audio | Skip | -- |
| 7 | Transformers Deep Dive | Skip | -- |
| 8 | Generative AI | Skip | -- |
| 9 | Reinforcement Learning | Skip | -- |
| 10 | LLMs from Scratch | Skip | -- |
| 11 | LLM Engineering | Do | 17 hr |
| 12 | Multimodal AI | Do | 65 hr |
| 13 | Tools & Protocols | Do | 24.5 hr |
| 14 | Agent Engineering | Do | 42 hr |
| 15 | Autonomous Systems | Do | 20 hr |
| 16 | Multi-Agent & Swarms | Do | 28 hr |
| 17 | Infrastructure & Production | Do | 32 hr |
| 18 | Ethics, Safety & Alignment | Do | 31 hr |
| 19 | Capstone Projects | Do | 500 hr |

**总计：约 803.5 小时，横跨 11 个阶段（含 2 个复习阶段）**

---

## 每课统一结构

```
lesson-name/
├── docs/en.md    ← 核心文档（英文），按 Problem → Concept → Implementation 递进
├── code/         ← 可直接运行的代码示例
├── quiz.json     ← 配套测验，检验理解
└── outputs/      ← 产出物（prompt/skill/agent 模板，部分课有）
```

---

## 每课学习流程

1. **读文档** — 打开 `docs/en.md`，理解 Problem 和 Concept 部分
2. **跑代码** — 进入 `code/` 目录运行示例，动手修改加深理解
3. **做测验** — 用 `quiz.json` 自测，确认知识点掌握
4. **收集产出** — `outputs/` 里的 prompt/skill 模板可直接复用到工作中

---

## Phase 11: LLM Engineering 学习顺序

Phase 11 共 17 课，按依赖关系排列，建议顺序学习：

| 模块 | 课程编号 | 主题 | 重点 |
|------|----------|------|------|
| Prompt 基础 | 01-03 | Prompt 工程、Few-Shot/CoT、结构化输出 | 掌握与 LLM 交互的核心技巧 |
| 检索与上下文 | 04-07 | Embeddings、Context Engineering、RAG、Advanced RAG | 构建知识增强应用的关键 |
| 微调 | 08 | LoRA/QLoRA | 低成本定制模型 |
| 工具调用 | 09 | Function Calling | LLM 调用外部工具的能力 |
| 工程化 | 10-13 | 评估、缓存成本、安全护栏、生产应用 | 从原型到生产的工程实践 |
| 协议与框架 | 14-17 | MCP、Prompt Caching、LangGraph、Agent 框架选型 | 进阶架构与协议 |

---

## 复习建议

- **Phase 1（数学基础）**：重点复习概率与统计（Lesson 06-07）、贝叶斯定理
- **Phase 2（机器学习基础）**：重点复习模型评估指标（Lesson 09）、不平衡数据处理（Lesson 17）

---

## 快速开始

```bash
# 进入第一课
cd phases/11-llm-engineering/01-prompt-engineering

# 阅读文档
cat docs/en.md

# 运行代码（确保已安装依赖：pip install -r requirements.txt）
python code/prompt_engineering.py

# 查看测验
cat quiz.json
```

---

## 注意事项

- 代码运行前需配好 API Key（参考 Phase 0 Lesson 04）
- 如果已熟悉 Prompt Engineering 基础，可快速浏览前 3 课，从 Lesson 04 (Embeddings) 深入
- 每个 Phase 的课程按顺序排列，后续课程依赖前序知识
- 产出物（outputs/）可以作为日常工作的参考模板
