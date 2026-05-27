# 自托管服务选择 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年有四个引擎主导自托管推理。根据硬件、规模和工作负载选择。**llama.cpp** 在 CPU 上最快 — 最广泛的模型支持，对量化和线程的完全控制。**Ollama** 是开发笔记本的一键安装，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP 序列化），在生产级负载下有 3 倍的吞吐量差距。**TGI 于 2025 年 12 月 11 日进入维护模式** — 仅修复 bug，原始吞吐量比 vLLM 慢约 10%，但历史上具有顶级的可观测性和 HF 生态系统集成。该维护状态使其成为有风险的长期选择 — SGLang 或 vLLM 是新项目的更安全默认选择。**vLLM** 是通用生产默认选择 — v0.15.1（2026 年 2 月）添加了 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是 agent 多轮/前缀密集型专家 — 400,000+ GPU 在生产中（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：仅 CPU → 仅 llama.cpp。AMD/非 NVIDIA → 仅 vLLM（TRT-LLM 是 NVIDIA 锁定的）。2026 年流水线模式：开发 = Ollama，预发布 = llama.cpp，生产 = vLLM 或 SGLang。全程使用相同的 GGUF/HF 权重。

**类型：** 学习
**语言：** Python（标准库，引擎决策树演练）
**前置要求：** 所有涉及引擎的第 17 阶段课程（04、06、07、09、18）
**时间：** 约 45 分钟

## 学习目标

- 根据硬件（CPU/AMD/NVIDIA Hopper/Blackwell）、规模（1 用户/100/10,000）和工作负载（通用聊天/agent/长上下文）选择引擎。
- 说出 2026 年 TGI 维护模式状态（2025 年 12 月 11 日）以及为什么它使新项目偏向 vLLM 或 SGLang。
- 描述全程使用相同 GGUF 或 HF 权重的开发/预发布/生产流水线。
- 解释为什么"仅 CPU"强制使用 llama.cpp，以及"AMD"排除 TRT-LLM。

## 问题

你的团队开始一个新的自托管 LLM 项目。一个工程师说 Ollama，另一个说 vLLM，第三个说"TGI 不是开箱即用吗？"三者对不同上下文都是正确的。没有哪个对所有上下文都是正确的。

在 2026 年，选择树很重要：硬件第一，规模第二，工作负载第三。而一个特定的 2025 年事件 — TGI 于 12 月 11 日进入维护模式 — 改变了新项目的默认选择。

## 概念

### 五个引擎

| 引擎 | 最适合 | 备注 |
|------|--------|------|
| **llama.cpp** | CPU/边缘/最小依赖/最广泛模型支持 | CPU 上最快，完全控制 |
| **Ollama** | 开发笔记本、单用户、一键安装 | 比 llama.cpp 慢 15-30%；3 倍生产吞吐量差距 |
| **TGI** | HF 生态系统、受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产、100+ 用户 | 广泛的生产默认选择；v0.15.1 2026 年 2 月 |
| **SGLang** | Agent 多轮、前缀密集型工作负载 | 400,000+ GPU 在生产中 |

### 硬件优先决策

**仅 CPU** → llama.cpp。Ollama 也可以，但更慢。没有其他引擎在 CPU 上有竞争力。

**AMD GPU** → vLLM（AMD ROCm 支持）。SGLang 也可以。TRT-LLM 是 NVIDIA 锁定的，所以排除。

**NVIDIA Hopper（H100/H200）** → vLLM 或 SGLang 或 TRT-LLM。三者都是顶级。

**NVIDIA Blackwell（B200/GB200）** → TRT-LLM 是吞吐量领先者（第 17 阶段 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 封装了它。

### 规模第二决策

**1 用户/本地开发** → Ollama。一个命令，几秒钟内出第一个 token。

**10-100 用户/小团队** → vLLM 单 GPU。

**100-10K 用户/生产** → vLLM production-stack（第 17 阶段 · 18）或 SGLang。

**10K+ 用户/企业** → vLLM production-stack + 分离式（第 17 阶段 · 17）+ LMCache（第 17 阶段 · 18）。

### 工作负载第三决策

**通用聊天/问答** → vLLM 在广泛默认选择上胜出。

**Agent 多轮（工具、规划、记忆）** → SGLang 的 RadixAttention（第 17 阶段 · 06）占主导。

**具有重度前缀复用的 RAG（检索增强生成，Retrieval-Augmented Generation）** → SGLang。

**代码生成** → vLLM 可以；SGLang 在缓存上略好。

**长上下文（128K+）** → vLLM + 分块预填充；SGLang + 分层 KV。

### TGI 维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式 — 此后仅修复 bug。历史上：顶级可观测性、最佳 HF 生态系统集成（模型卡、安全工具）、原始吞吐量略落后于 vLLM。

对于 2026 年的新项目：默认远离 TGI。现有 TGI 部署可以继续，但最终应该迁移。SGLang 和 vLLM 是更安全的默认选择。

### 流水线模式

开发（Ollama）→ 预发布（llama.cpp）→ 生产（vLLM）。全程使用相同的 GGUF 或 HF 权重。工程师在笔记本上快速迭代；预发布镜像生产量化；生产是服务目标。

### Ollama 注意事项

Ollama 对开发很棒。对共享生产不好：Go HTTP 序列化增加开销，并发管理比 vLLM 简单，OpenTelemetry 支持滞后。在 Ollama 擅长的场景使用它 — 一个用户，一个命令 — 并在共享场景切换到 vLLM。

### 自托管 vs 托管是独立的决策

第 17 阶段 · 01（托管超大规模云平台）、· 02（推理平台）涵盖托管。本课假设你已经决定自托管。自托管的理由：数据驻留、自定义微调、大规模下的总拥有成本、托管上不可用的领域模型。

### 你应该记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产足迹：400,000+ GPU。
- Ollama 吞吐量差距 vs llama.cpp：慢 15-30%；生产负载下 3 倍。

## 动手实践

`code/main.py` 是一个决策树演练：给定硬件 + 规模 + 工作负载，选择引擎并解释原因。

## 交付物

本课产出 `outputs/skill-engine-picker.md`。给定约束，选择引擎并编写迁移计划。

## 练习

1. 运行 `code/main.py`，使用你的硬件/规模/工作负载。输出与你的直觉匹配吗？
2. 你的基础设施是 12 个 H100 和 8 个 MI300X AMD。什么引擎？为什么 TRT-LLM 不在考虑范围内？
3. 一个团队想在 2026 年使用 TGI，因为"这是我们所知道的。"论证迁移的理由。
4. Ollama 开发到 vLLM 生产：量化、配置和可观测性方面有什么变化？
5. RAG 产品，P99 前缀长度 8K，跨租户高复用率。选择引擎并与第 17 阶段 · 11 + 18 叠加。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| llama.cpp | "CPU 那个" | 最广泛模型支持，CPU 上最快 |
| Ollama | "笔记本那个" | 一键安装，开发级吞吐量 |
| TGI | "HF 的服务" | 自 2025 年 12 月起维护模式 |
| vLLM | "默认选择" | 2026 年广泛的生产基线 |
| SGLang | "agent 那个" | 前缀密集型，RadixAttention |
| TRT-LLM | "NVIDIA 锁定" | Blackwell 吞吐量领先者，仅 NVIDIA |
| GGUF | "llama.cpp 格式" | 捆绑的 K-quant 变体 |
| Production-stack | "vLLM K8s" | 第 17 阶段 · 18 参考部署 |
| 流水线模式（Pipeline pattern） | "开发→预发布→生产" | 在相同权重上 Ollama → llama.cpp → vLLM |
