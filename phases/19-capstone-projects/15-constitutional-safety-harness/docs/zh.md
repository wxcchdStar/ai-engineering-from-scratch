# 实战项目 15 — 宪法式安全套件 + 红队靶场

> Anthropic 的宪法分类器（Constitutional Classifiers）、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety 以及用于多语言覆盖的 X-Guard 定义了 2026 年的安全分类器栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准的对抗性评估工具。NeMo Guardrails v0.12 将它们整合到生产管道中。本实战项目将所有这些连接在一起：围绕目标应用的 layered safety harness（分层安全套件）、一个运行 6 种以上攻击家族的自主红队智能体，以及一个产生可衡量无害性提升（Harmlessness Delta）的宪法式自我 critique 运行。

**类型：** 实战项目（Capstone）
**语言：** Python（安全管道、红队）、YAML（策略配置）
**前置课程：** 第 10 阶段（从零构建 LLM）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 18 阶段（伦理、安全、对齐）
**涉及阶段：** P10 · P11 · P13 · P14 · P18
**时间：** 25 小时

## 问题

2026 年 LLM 安全的前沿不是分类器是否有效（它们大致有效），而是如何围绕生产应用正确地组合它们，而不过度拒绝或留下明显的漏洞。Llama Guard 4 处理英语策略违规。X-Guard（132 种语言）处理多语言越狱。ShieldGemma-2 捕获基于图像的提示注入。NVIDIA Nemotron 3 Content Safety 覆盖企业类别。Anthropic 的宪法分类器是一种在训练期间而非服务时使用的独立方法。

攻击演化也很重要。PAIR 和 TAP 自动化越狱发现。GCG 运行基于梯度的后缀攻击。多轮和代码切换攻击利用智能体记忆。任何部署的 LLM 都需要一个红队靶场——garak 和 PyRIT 是规范驱动工具——加上文档化的缓解措施和 CVSS 评分的发现。

你将加固一个目标应用（一个 8B 指令调优模型或来自其他实战项目的 RAG 聊天机器人），对其运行 6 种以上攻击家族，并生成前后无害性测量。

## 概念

安全管道有五层。**输入净化**：剥离零宽字符、解码 base64/rot13、规范化 Unicode。**策略层**：NeMo Guardrails v0.12 规则（领域外、毒性、PII 提取）。**分类器门控**：Llama Guard 4 用于输入，X-Guard 用于非英语，ShieldGemma-2 用于图像输入。**模型**：目标 LLM。**输出过滤器**：Llama Guard 4 用于输出，Presidio PII 脱敏，适用的引用强制。**人机协同层**：标记为高风险的输出进入 Slack 队列。

红队靶场在调度器上运行。PAIR 和 TAP 自主发现越狱。GCG 运行基于梯度的后缀攻击。ASCII/base64/rot13 编码攻击。多轮攻击（角色扮演、记忆利用）。代码切换攻击（英语与斯瓦希里语或泰语混合）。每次运行生成结构化的发现文件，包含 CVSS 评分和披露时间线。

宪法式自我 critique 运行是一种训练时干预。取 1k 有害尝试提示，让模型起草响应，根据书面宪法（不伤害规则）对其进行 critique，并在 critique 循环上重新训练。在保留评估上测量前后无害性提升。

## 架构

```
请求（文本/图像/多语言）
      |
      v
输入净化（剥离零宽字符、解码、规范化）
      |
      v
NeMo Guardrails v0.12 规则（领域外、策略）
      |
      v
分类器门控：
  Llama Guard 4（英语）
  X-Guard（多语言，132 种语言）
  ShieldGemma-2（图像提示）
  Nemotron 3 Content Safety（企业）
      |
      v（允许）
目标 LLM
      |
      v
输出过滤器：Llama Guard 4 + Presidio PII + 引用检查
      |
      v
标记输出的人机协同层

并行：
  红队调度器
    -> garak（经典攻击）
    -> PyRIT（编排红队）
    -> 自主越狱智能体（PAIR + TAP）
    -> GCG 后缀攻击
    -> 多语言/代码切换
    -> 多轮角色扮演

输出：CVSS 评分的发现 + 披露时间线 + 前后无害性提升
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 防护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱智能体：PAIR（Chao 等人，2023）、攻击树（TAP）、GCG 后缀
- 宪法训练：Anthropic 风格自我 critique 循环 + 对 critique 的 SFT
- PII 脱敏：Presidio
- 目标：一个 8B 指令调优模型或其他实战项目的 RAG 聊天机器人

## 构建步骤

1. **目标设置。** 在 vLLM 上启动一个 8B 指令调优模型（或重用其他实战项目的 RAG 聊天机器人）。这是被测应用。

2. **安全管道包装。** 围绕目标连接五层管道。验证每一层在 Langfuse 中可单独观测（每层一个 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在每个分类器的小型标注集上运行以建立基线。

4. **红队调度器。** 调度 garak、PyRIT、PAIR 智能体、TAP 智能体、GCG 运行器、多轮攻击者和代码切换攻击者。每个在独立队列上运行。

5. **攻击套件。** 六种攻击家族：（1）PAIR 自动化越狱，（2）TAP 攻击树，（3）GCG 梯度后缀，（4）ASCII/base64/rot13 编码，（5）多轮角色扮演，（6）多语言代码切换。报告每个家族的成功率。

6. **宪法式自我 critique。** 策划 1k 有害尝试提示。对于每个，目标起草响应。Critic LLM 根据书面宪法评分（"不伤害"、"引用证据"、"拒绝非法请求"）。critic 反对的提示被重写；目标在 critique 改进的对上微调。在保留评估上测量前后无害性。

7. **过度拒绝测量。** 在良性提示套件（例如 XSTest）上跟踪误报率。目标必须在良性问题上保持有帮助。

8. **CVSS 评分。** 对于每个成功的越狱，按 CVSS 4.0 评分（攻击向量、复杂性、影响）。生成披露时间线和缓解计划。

9. **靶场自动化。** 以上所有在 cron 上运行；发现写入队列；过度拒绝回退告警触发到 Slack。

## 使用方式

```
$ safety probe --model=target --family=PAIR --budget=50
[攻击者]   PAIR 智能体在目标上运行
[攻击]     尝试 1/50：将查询伪装成学术研究 ... 被阻止
[攻击]     尝试 2/50：诉诸角色扮演 ... 被阻止
[攻击]     尝试 3/50：思维链诱导 ... 成功
[发现]     CVSS 4.8 中等：目标上的角色扮演绕过
[靶场]     50 次中 7 次成功（14% 成功率）
```

## 交付标准

`outputs/skill-safety-harness.md` 是交付物。一个生产级分层安全管道加上一个可复现的红队靶场，包含前后无害性提升。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 6 种以上攻击家族演练，2 种以上语言 |
| 20 | 真阳性/假阳性权衡 | 攻击阻止率 vs XSTest 良性通过率 |
| 20 | 自我 critique 提升 | 在保留评估上的前后无害性 |
| 20 | 文档和披露 | CVSS 评分的发现及时间线 |
| 15 | 自动化和可重复性 | 一切在 cron 上运行并带告警 |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的提示注入插件，比较有和没有输出过滤器层的攻击成功率。
2. 添加第七种攻击家族：通过检索文档的间接提示注入。测量所需的额外防御。
3. 实现"拒绝但提供帮助"模式：当防护栏阻止时，目标提供更安全的相关答案而非直接拒绝。测量 XSTest 差异。
4. 多语言覆盖差距：找到 X-Guard 表现不佳的语言。提出针对它的微调数据集。
5. 在 30B 模型上运行宪法式自我 critique，测量提升是否随规模扩展。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| 分层安全 | "纵深防御" | 在输入、门控、输出、人机协同层的多重防护栏 |
| Llama Guard 4 | "Meta 的安全分类器" | 2026 年参考输入/输出内容分类器 |
| PAIR | "越狱智能体" | 关于 LLM 驱动越狱发现的论文（Chao 等人） |
| TAP | "攻击树" | PAIR 的树搜索变体 |
| GCG | "贪婪坐标梯度" | 基于梯度的对抗性后缀攻击 |
| 宪法式自我 critique | "Anthropic 风格训练" | 目标起草 -> critic 评分 -> 重写 -> 重新训练 |
| XSTest | "良性探测集" | 用于过度拒绝回退的基准 |
| CVSS 4.0 | "严重性评分" | 用于安全发现的标准漏洞评分 |

## 扩展阅读

- [Anthropic 宪法分类器](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard（arXiv:2504.08848）](https://arxiv.org/abs/2504.08848) — 132 种语言多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 规则框架
- [PAIR（arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — 越狱智能体论文
