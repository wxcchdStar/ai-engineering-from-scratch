# 具身 VLA：RT-2、OpenVLA、π0、GR00T

> 第一次有模型从网站上读取食谱并在厨房机器人中执行的是 RT-2（Google DeepMind, 2023年7月）。RT-2 将动作离散化为文本 token，在网络数据加机器人动作数据上对 VLM 进行联合微调，证明了网络规模的视觉-语言知识可以迁移到机器人控制。OpenVLA（2024年6月）发布了开源7B参考模型。Physical Intelligence 的 π0 系列（2024-2025）加入了流匹配动作专家。NVIDIA 的 GR00T N1（2025年3月）为人形机器人提供了大规模双系统（System 1 / System 2）控制。VLA 原语——视觉-语言-动作 (vision-language-action)，一个看、读、行的单一模型——是本阶段理解模型与 Phase 15 自主系统之间的桥梁。

**类型：** 学习（Learn）
**语言：** Python（标准库，动作分词器 + VLA 推理骨架）
**前置课程：** Phase 12 · 05（LLaVA）, Phase 15（自主系统，引用）
**时长：** ~180 分钟

## 学习目标

- 描述动作分词：离散 bin 编码（RT-2）、FAST 高效动作 token、连续流匹配动作（π0）。
- 解释为什么在网络 + 机器人数据上联合微调能保留通用知识向新任务的迁移。
- 比较 OpenVLA（开源 7B Llama+VLM）、π0（流匹配）和 GR00T N1（双系统）在相同机器人任务上的表现。
- 列举 Open X-Embodiment 数据集及其作为 RT-X 训练语料库的角色。

## 问题

一个能根据自然语言指令做家务的机器人自1970年代以来就是研究目标。2020年代的答案：视觉-语言-动作（VLA）模型。与 VQA 使用相同的 VLM 架构，但输出是动作（关节力矩、末端执行器位姿、离散命令）而非文本。

VLA 的特有挑战：

1. 动作空间是连续的（关节角度、力）且高维的（7自由度机械臂 + 3自由度夹爪 = 10维，30 Hz）。
2. 机器人专用训练数据稀缺。Open X-Embodiment 有约100万条轨迹；网络文本-图像有50亿+。
3. 控制频率很重要。30 Hz 控制循环意味着每个动作33ms预算。
4. 安全性。错误动作会损坏硬件、伤害人类或破坏财物。

## 核心概念

### 动作分词（RT-2）

RT-2 的技巧：将每个关节目标表示为量化文本 token。将归一化的 [-1, 1] 范围离散化为256个 bin，每个 bin 映射到一个词表 ID。10自由度动作在每个控制步变成10个 token。

在混合数据上对 PaLM-X VLM 联合微调：

- 网络图像-文本对（描述、VQA）。
- 机器人演示，动作作为 token。

模型看到"拿起红色方块"（语言）→ 图像（视觉）→ 10-token 动作序列（离散化关节目标）。网络预训练保留了通用知识迁移：RT-2 可以执行"向快速移动的物体移动"，即使"快速移动"不在训练数据中。

RT-2 论文中推理频率为 3-5 Hz，受限于 VLM 自回归解码。

### OpenVLA — 开源7B参考模型

OpenVLA（Kim et al., 2024年6月）是开源权重的 RT-2 等价物。7B Llama 主干，DINOv2 + SigLIP 双视觉编码器，256 bin 动作分词。

在 Open X-Embodiment（22个机器人的97万条轨迹）上训练。提供 LoRA 微调支持以适配新机器人。

推理：量化后在 A100 上 4-5 Hz。对慢速操作足够快，但不适合高频控制。

### FAST 分词器 — 更快的动作解码

Pertsch et al.（2024）表明离散 bin 分词效率低——大多数动作聚集在 bin 空间的小区域。FAST（频域动作序列分词器，Frequency-domain Action Sequence Tokenizer）通过 DCT 压缩动作序列并量化系数。

一条30步动作轨迹变成约10个 FAST token，而非300个离散 bin token。推理加速3-5倍且无质量损失。

### π0 与流匹配动作

Physical Intelligence 的 π0（Black et al., 2024年10月）用流匹配动作专家替代离散动作 token：

- 一个小型动作 Transformer 读取 VLM 的隐藏状态，通过整流流 (rectified flow) 输出连续50步动作序列。
- 动作头用流匹配损失训练；VLM 预训练保持不变。
- 推理：完整动作序列在约5个去噪步中输出，实际达到50 Hz 控制。

π0 的声明：在广泛的操作任务套件上超越 OpenVLA 和 Octo。连续动作表示保留了离散化所破坏的平滑性。

π0.5 和 π0-FAST 是增量升级。π0-FAST 结合了 FAST 分词与流匹配。

### GR00T N1 — 人形机器人双系统

NVIDIA 的 GR00T N1（2025年3月）为人形机器人（>30自由度，全身）设计：

- System 2：大型 VLM 读取场景 + 指令，以约1 Hz 产生高层子目标。
- System 1：小型动作头 Transformer 基于子目标以50-100 Hz 产生低层关节命令。

这种拆分映射到 Kahneman 的快慢思维：System 2 规划，System 1 执行。好处：慢速 VLM 级规划不阻塞快速控制；System 1 保持小规模以降低延迟。

GR00T N1.7（2025年末）改进了数据扩展。GR00T 使用 Omniverse 的 sim-to-real 数据微调。

### Open X-Embodiment

训练数据。RT-X（2023年10月）汇集了22个数据集，覆盖22个机器人的100万条轨迹。Open X-Embodiment 是所有人使用的语料库：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每个样本：（机器人状态，摄像头视角，指令，动作序列）。
- 训练卫生：统一动作空间、归一化关节范围、调整摄像头尺寸。

OpenVLA 和 π0 在 Open X-Embodiment 上训练。与特定机器人的领域差距通过在100-1000个任务特定演示上 LoRA 微调来弥合。

### 联合微调 vs 纯机器人训练

联合微调混合网络 VQA 数据与机器人轨迹。比例很重要：VQA 太多模型会忘记动作；机器人数据太多模型会丢失通用知识。

RT-2 的比例：约1:1。OpenVLA：约0.5:1 网络对机器人。π0：类似。精确比例是每个数据集大小需要调整的超参数。

纯机器人训练产生的任务特定模型在分布外指令上失败。联合微调是"拿起红色方块（演示中有）"与"从左边拿起第三大的物体（新表述）"之间的区别。

### 安全与动作限制

每个生产级 VLA 都配备：

- 硬关节限制（不能超过规格扭矩）。
- 速度限制（软裁剪）。
- 工作空间边界（末端执行器不能离开桌面）。
- 人类在环审批（针对新任务）。

这些作为控制层检查存在于 VLA 之外。VLA 的输出是建议，不是命令。

## 动手实践

`code/main.py`：

- 实现256 bin 动作分词和反分词。
- 勾画基于 DCT + 量化的 FAST 分词器。
- 比较（离散 bin、FAST、连续流）各种方案每个动作步的 token 数。
- 打印 RT-2 → OpenVLA → π0 → GR00T 的谱系总结。

## 交付产出

本课产出 `outputs/skill-vla-action-format-picker.md`。给定一个机器人任务（操作、导航、人形全身），选择离散 bin + RT-2、FAST + OpenVLA、流匹配 + π0 或双系统 + GR00T。

## 练习

1. 10自由度机械臂，30 Hz 控制频率。256 bin 离散分词每秒输出多少 token？7B VLM 能跟上吗？

2. FAST 分词将30步轨迹压缩到约10个 token。如果轨迹有高频运动（如打鼓），用户会损失什么？

3. π0 的流匹配头在约5步中去噪。将吞吐量与 OpenVLA 4-5 Hz 的自回归解码比较。

4. GR00T 的 System 1 / System 2 拆分映射到 Kahneman。提出一种不同的拆分（System 3？）可能有助于双足行走。

5. 阅读 Open X-Embodiment 第4节关于数据整理的内容。列举防止领域泄漏的三条整理规则。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| VLA | "视觉-语言-动作" | 接收图像 + 指令并输出动作命令的模型 |
| 动作分词 (Action tokenization) | "离散 bin" | 将连续关节目标量化为每维256个 bin，每个 bin 是一个词表 ID |
| FAST 分词器 (FAST tokenizer) | "频域动作 token" | DCT + 量化将30步轨迹压缩到约10个 token |
| 联合微调 (Co-fine-tune) | "混合网络 + 机器人" | 同时在网络 VQA 数据和机器人演示上训练以保留通用知识 |
| 流匹配动作头 (Flow-matching action head) | "π0 连续输出" | 通过整流流输出50步动作序列的小型 Transformer |
| System 1 / System 2 | "双系统控制" | 大型 VLM 慢速规划，小型动作头快速执行；GR00T 模式 |
| Open X-Embodiment | "RT-X 数据集" | 100万条跨机器人轨迹数据集；通用训练语料库 |

## 延伸阅读

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
