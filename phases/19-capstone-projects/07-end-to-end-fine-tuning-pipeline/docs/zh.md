# 实战项目 07 — 端到端微调管道（数据到 SFT 到 DPO 到服务）

> 一个 8B 模型，使用你自己的数据进行训练，根据你自己的偏好进行 DPO 对齐，量化，使用推测解码（Speculative Decoding）进行推理，并以可衡量的 $/1M tokens 提供服务。2026 年的开源技术栈是 Axolotl v0.8、TRL 0.15、Unsloth 用于迭代、GPTQ/AWQ/GGUF 用于量化、vLLM 0.7 配合 EAGLE-3 用于服务。本实战项目要求可复现地运行整个管道——YAML 输入，服务端点输出——并根据 2026 年模型开放性框架（Model Openness Framework）发布模型卡（Model Card）。

**类型：** 实战项目（Capstone）
**语言：** Python（管道）、YAML（配置）、Bash（脚本）
**前置课程：** 第 2 阶段（机器学习）、第 3 阶段（深度学习）、第 7 阶段（Transformers）、第 10 阶段（从零构建 LLM）、第 11 阶段（LLM 工程）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间：** 35 小时

## 问题

2026 年，每个严肃的 AI 团队都随时备有一个微调管道。不是因为他们要发布前沿基础模型，而是因为下游适配——领域 SFT（监督微调）、针对标注偏好的 DPO（直接偏好优化）、用于推测解码的蒸馏草稿模型、配合 EAGLE-3 的服务——才是可衡量收益所在。Axolotl v0.8 处理多 GPU SFT 配置。TRL 0.15 处理 DPO 和 GRPO（组相对策略优化）。Unsloth 让你快速进行单 GPU 迭代。vLLM 0.7 配合 EAGLE-3 将解码吞吐量提升 2-3 倍而无质量损失。工具是现成的；精髓在于 YAML 配置、数据卫生和评估纪律。

你将把一个 8B 基础模型（Llama 3.3、Qwen3 或 Gemma 3）通过 SFT 然后 DPO 在任务特定数据上运行，量化后用于服务，并使用 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 测量收益。你将根据 2026 年模型开放性框架生成模型卡。关键在于可复现性——一条命令端到端重跑整个管道。

## 概念

该管道有五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 脱敏、针对公共基准污染的拆分卫生检查。**SFT**：Axolotl YAML，ZeRO-3 在 8xH100 上，余弦调度，打包序列（Packed Sequences），2-3 个 epoch。**DPO 或 GRPO**：TRL 配置，1 个 epoch，偏好对可以是人工标注或模型评判的，beta 调优。**量化**：GPTQ + AWQ + GGUF 以支持部署灵活性。**服务**：vLLM 0.7 配合 EAGLE-3 推测头（或 SGLang 配合 SpecForge），K8s 部署，基于队列等待时间的 HPA（水平自动扩缩）。

消融实验是交付物：SFT-only vs SFT+DPO vs SFT+GRPO 在三个任务特定基准上的对比。服务指标：batch 1/8/32 下的 tokens/s、EAGLE-3 接受率、$/1M tokens。安全评估：Llama Guard 4 通过率。模型卡：偏差评估、可复现性种子、数据许可。

## 架构

```
原始数据（HF 数据集 + 内部数据）
    |
    v
Datatrove 去重 + Nemotron-CC 质量过滤 + PII 脱敏
    |
    v
拆分卫生检查（MMLU-Pro 污染检查）
    |
    v
Axolotl SFT 配置（YAML）  ---> 8xH100，ZeRO-3
    |
    v
TRL DPO / GRPO 配置        ---> 4xH100，1 个 epoch
    |
    v
GPTQ + AWQ + GGUF 量化
    |
    v
vLLM 0.7 + EAGLE-3 推测解码
    |
    v
K8s 部署，基于队列等待时间的 HPA
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
模型卡（2026 MOF）+ 安全评估（Llama Guard 4）
```

## 技术栈

- 数据：Datatrove 用于去重，Nemotron-CC 分类器用于质量，Presidio 用于 PII
- 基础模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8 配合 ZeRO-3、Flash Attention 3、打包序列
- 偏好调优：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 迭代
- 量化：GPTQ（Marlin）、AWQ、GGUF 通过 llama.cpp
- 服务：vLLM 0.7 配合 EAGLE-3 推测解码（或 SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA 设备插件，基于队列等待时间指标的 HPA
- 可观测性：W&B 用于训练，Langfuse 用于推理

## 构建步骤

1. **数据管道。** 对原始语料运行 Datatrove 去重。应用 Nemotron-CC 风格的质量分类器。Presidio 脱敏 PII。使用显式种子写入训练/验证拆分。

2. **污染检查。** 对于每个验证拆分，计算与 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集的 MinHash。拒绝任何重叠。

3. **Axolotl SFT。** YAML 配置 ZeRO-3、FA3、序列打包。在 8xH100 上训练 2-3 个 epoch。记录到 W&B。

4. **TRL DPO / GRPO。** 取 SFT 检查点，在偏好对上运行一个 epoch 的 DPO（或在数学/代码上使用可验证奖励的 GRPO）。扫描 beta 参数。

5. **量化。** 生成三种量化格式：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M 用于 llama.cpp。记录大小和标称吞吐量。

6. **使用推测解码服务。** vLLM 0.7 配置 EAGLE-3 草稿头，通过 Red Hat Speculators 训练。在 batch 1/8/32 下测量接受率和尾部延迟。报告与 Anthropic/OpenAI 在相同评估上的 $/1M tokens。

7. **评估矩阵。** 在基础模型、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成表格。

8. **安全评估。** 在开发集上的 Llama Guard 4 通过率。ShieldGemma-2 输出过滤器。

9. **模型卡。** MOF 2026 模板：数据、训练、评估、安全、许可、包含 YAML 和提交 SHA 的可复现性部分。

## 使用方式

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[数据]    300k 去重后，12k 被过滤，280k 接受（种子=7）
[SFT]     3 个 epoch，8xH100，6h12m，验证损失 1.42 -> 1.03
[DPO]     1 个 epoch，beta=0.08，4xH100，1h40m
[量化]    GPTQ-INT4 4.6 GB，AWQ-INT4 4.8 GB，GGUF-Q4_K_M 5.1 GB
[服务]    vLLM 0.7，EAGLE-3 接受率 0.74，p99 126ms @ bs=8
[评估]    MMLU-Pro +3.2，MT-Bench-v2 +0.41，RewardBench-2 +0.08
[模型卡]  根据 2026 MOF 生成 model-card.md
```

## 交付标准

`outputs/skill-finetuning-pipeline.md` 描述交付物。一条命令运行数据到 SFT 到 DPO 到量化到服务到评估，并输出模型卡 + 服务端点。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 相对于基础模型的评估提升 | 在目标任务（MMLU-Pro、MT-Bench-v2、任务特定）上测量的收益 |
| 20 | 管道可复现性 | 一条命令使用相同种子端到端重跑 |
| 20 | 数据卫生 | 去重率、PII 脱敏覆盖率、污染检查绿色通过 |
| 20 | 服务效率 | bs=1/8/32 下的 tokens/s、EAGLE-3 接受率、$/1M tokens |
| 15 | 模型卡 + 安全评估 | 2026 MOF 完整性 + Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在同一任务特定基准上运行 SFT-only vs SFT+DPO vs SFT+GRPO。报告哪种偏好方法胜出以及胜出多少。
2. 将 Llama 3.3 8B 替换为 Qwen3 14B。在匹配质量下测量 $/1M tokens。
3. 在领域数据 vs 通用 ShareGPT 上测量 EAGLE-3 接受率。报告差异及其对延迟预算的意义。
4. 注入 1% 的污染（将 MMLU-Pro 答案泄露到训练数据中）并重新运行评估。观察 MMLU-Pro 准确率不切实际地飙升。构建一个能捕获此问题的污染检查 CI 门控。
5. 添加 LoRA SFT 作为全量微调的替代方案。在内存降低 10 倍的情况下测量质量差距。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | "SFT 训练器" | 统一的 YAML 驱动训练器，用于 SFT、DPO 和蒸馏 |
| TRL | "偏好调优器" | Hugging Face 库，用于 LLM 的 DPO、GRPO、PPO |
| GRPO | "组相对策略优化" | DeepSeek R1 的 RL 配方，使用可验证奖励 |
| EAGLE-3 | "推测解码草稿" | 预测未来 N 个 token 的草稿头；vLLM 使用目标模型验证 |
| MOF | "模型开放性框架" | 2026 年用于对模型发布在数据、代码、许可方面进行评分的标准 |
| 污染检查 | "拆分卫生" | 基于 MinHash 的检测，用于发现测试集泄露到训练数据中 |
| 接受率 | "EAGLE / MTP 指标" | 目标模型接受的草稿 token 比例 |

## 扩展阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — 参考 SFT / DPO 训练器
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 迭代参考
- [DeepSeek R1 论文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 备选推测解码训练器
- [模型开放性框架 2026](https://isocpp.org/) — 开放发布评分标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 规范评估运行器
