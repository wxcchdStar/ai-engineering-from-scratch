# 直接偏好优化家族（The Direct Preference Optimization Family）

> Rafailov 等人（2023）表明 RLHF 的最优解在偏好数据方面有闭式形式，因此你可以跳过显式奖励模型，直接优化策略。这一洞察催生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每个都修复了 DPO 的一个失败模式。在 2026 年，直接对齐算法比 PPO 交付了更多的前沿后训练运行。但第 2 课的过度优化曲线仍然适用：DAA 不能逃脱古德哈特，它们只是改变了它咬合的位置。

**类型：** 学习
**语言：** Python（标准库，六变体偏好损失比较器）
**前置要求：** 第 18 阶段 · 01（InstructGPT），第 18 阶段 · 02（奖励欺骗），第 10 阶段 · 08（DPO 基础）
**时间：** 约 75 分钟

## 学习目标

- 从带 KL 的 RLHF 最优解推导 DPO 闭式形式。
- 陈述 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 的哪个失败模式。
- 区分「隐式奖励差距」和「偏好强度」，并解释为什么 IPO 的恒等映射很重要。
- 解释为什么 Rafailov 等人（NeurIPS 2024）证明 DAA 尽管没有显式 RM 也会过度优化。

## 问题

RLHF 目标（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此奖励由最优策略与参考策略的比率隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将其代入 Bradley-Terry 偏好似然，配分函数 `Z(x)` 因为只依赖于 `x` 而抵消。剩下的是一个仅含策略参数的损失——不需要奖励模型。这就是 DPO。

问题：推导假设最优解可达、偏好数据在分布内、参考策略是真正的模式锚点。这些都不完全成立。每个家族成员修复了一个不同的被违反的假设。

## 概念

### DPO（Rafailov 等人，2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出错的地方：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。一个微小的偏好可以产生任意大的差距。
- 损失将被选择和被拒绝的对数概率推向相反方向。只要被拒绝的下降更快，它就可以将被选择的绝对对数概率向下推。这就是退化被选择响应（Degraded Chosen Response）现象。
- 分布外偏好（罕见对 vs 罕见对）产生任意的隐式奖励。

### IPO（Azar 等人，2024）

恒等偏好优化（Identity Preference Optimization）将对数-sigmoid 替换为偏好概率上的恒等映射。损失变为有界目标上的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边际由 `1/(2 beta)` 界定。偏好强度和隐式奖励差距成比例。不会爆炸。

### KTO（Ethayarajh 等人，2024）

Kahneman-Tversky 优化完全放弃了成对结构。给定单个标注输出和一个二元「可取」或「不可取」信号，它映射到前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失使用不同的权重（损失厌恶）。好处：你可以使用不成对的数据，这要丰富得多。

### SimPO（Meng 等人，2024）

简单偏好优化（Simple Preference Optimization）将训练信号与生成对齐。完全移除参考策略，并按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

带有边际 `gamma` 以稳定。长度归一化移除了利用 DPO 长度偏差失败模式的激励（更长的 `y_w` 从构造上给出更大的对数概率差距）。

### ORPO（Hong 等人，2024）

优势比偏好优化（Odds-Ratio Preference Optimization）在标准 SFT 负对数似然上添加一个偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有参考策略——SFT 项就是正则化器。从基础模型到对齐模型单阶段训练。不需要单独的 SFT 检查点。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

识别了退化被选择响应问题：DPO 保持了排序 `y_w > y_l`，但 `y_w` 的绝对对数概率可能下降。BPO 添加了一个单行修正，惩罚被选择响应的向下移动。报告在 Llama-3.1-8B-Instruct 上数学推理比 DPO 提高 +10.1% 准确率。

### 普遍结果：DAA 仍然过度优化

Rafailov 等人「Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms」（NeurIPS 2024）在多个数据集上跨 KL 预算训练了 DPO、IPO、SLiC 策略。真实奖励-vs-KL 曲线具有相同的 Gao 等人峰值和崩溃形状。隐式奖励在训练期间查询分布外样本；KL 正则化不能稳定这一点。

DAA 不能逃脱古德哈特。它们改变了它咬合的表面，从「奖励模型过度优化」变为「参考策略比率过度优化」。通用修复——更好的数据、集成、提前停止——对两者都适用。

### 在它们之间选择（2026）

- 如果你有大量成对偏好数据：DPO 配合保守的 beta，如果长度偏差明显则用 SimPO。
- 如果你有不成对的二元反馈：KTO。
- 如果你想要从基础模型开始的单阶段流程：ORPO。
- 如果你在 DPO 日志中看到退化的被选择对数概率：BPO。
- 如果偏好强度变化很大且 DPO 饱和：IPO。

每个实验室都在一组测试上运行所有五种方法，并按任务选择胜者。没有理由最优解对数学推理和安全是相同的。

## 使用它

`code/main.py` 在玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度因对而异。每种损失在相同的 500 对样本上用小型 softmax 策略进行优化。绘制每种方法的最终胜率、被选择对数概率漂移和隐式奖励分布。

## 交付它

本课产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计（成对 vs 不成对，可变 vs 均匀偏好强度，长度分布）和目标（单阶段或 SFT-然后-偏好），推荐一个偏好损失并报告它保护的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终被选择对数概率下降。BPO 应该保持更高的被选择绝对概率——验证这一点。

2. 修改偏好数据使所有对具有相等的强度。六种方法中哪种最鲁棒？哪种退化？解释 IPO 在这里的优势。

3. 使被拒绝的响应平均比被选择的长 2 倍。不改变其他任何东西，用数字展示 DPO 的长度利用和 SimPO 的修复。

4. Rafailov 等人（NeurIPS 2024）声称 DAA 过度优化。重现一个单点版本：绘制被选择减被拒绝的 KL 散度，并观察 DPO 在大 beta 下的过度优化。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 添加到 DPO 的单行修正。对照 `code/main.py` 中的实现进行确认。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|------|-----------|-----------------|
| DPO | 「没有奖励模型的 RLHF」 | 从 RLHF 闭式最优解推导的损失；仅含策略参数 |
| 隐式奖励 | 「对数比率」 | `beta * log(pi(y|x) / pi_ref(y|x))`——DPO 隐含的奖励 |
| IPO | 「有界 DPO」 | 将对数-sigmoid 替换为恒等映射；隐式奖励差距由 `1/(2 beta)` 限制 |
| KTO | 「不成对 DPO」 | 在单个标签上使用前景理论效用和损失厌恶 |
| SimPO | 「无参考 DPO」 | 长度归一化对数似然 + 边际；无参考策略 |
| ORPO | 「单阶段 DPO」 | NLL + 优势比偏好项；从基础模型一次训练 |
| BPO | 「保留被选择的 DPO」 | DPO 加上对降低被选择响应绝对对数概率的惩罚 |
| 退化被选择 | 「被选择的下降」 | DPO 降低被选择的对数概率，只要被拒绝的下降更快 |
| DAA | 「直接对齐算法」 | 任何跳过显式 RM 的偏好损失方法 |

## 进一步阅读

- [Rafailov 等人——Direct Preference Optimization（NeurIPS 2023, arXiv:2305.18290）](https://arxiv.org/abs/2305.18290)
- [Azar 等人——A General Theoretical Paradigm to Understand Learning from Human Preferences（AISTATS 2024, arXiv:2310.12036）](https://arxiv.org/abs/2310.12036)——IPO
- [Ethayarajh 等人——KTO: Model Alignment as Prospect Theoretic Optimization（arXiv:2402.01306）](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen——SimPO（NeurIPS 2024, arXiv:2405.14734）](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne——ORPO（EMNLP 2024, arXiv:2403.07691）](https://arxiv.org/abs/2403.07691)
- [BPO——Behavior Preservation Optimization（ICLR 2026 OpenReview b97EwMUWu7）](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等人——Scaling Laws for RM Overoptimization in DAAs（NeurIPS 2024, arXiv:2406.02900）](https://arxiv.org/abs/2406.02900)
