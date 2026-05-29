# 数值稳定性 (Numerical Stability)

> 浮点数是抽象有漏洞。它会在训练时反咬你一口，而你根本看不到它来。

**类型：** 构建 (Build)
**语言：** Python
**前置要求：** 第一阶段，第 01-04 课
**时间：** ~120 分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的溢出、下溢和灾难性抵消
- 使用中心有限差分验证解析梯度与数值梯度
- 解释为什么 bfloat16 比 float16 更适合训练，以及损失缩放如何防止梯度下溢

## 问题

你的模型训练了三个小时，然后损失变成了 NaN。你加了 print 语句。在第 9,000 步时 logits 还正常，到了第 9,001 步就变成了 `inf`，到第 9,002 步所有梯度都是 `nan`，训练彻底死了。

或者：你的模型训练完成了，但精度比论文声称的低了 2%。你检查了所有东西——架构吻合，超参数吻合，数据吻合。问题在于论文用了 float32，而你用 float16 却没做正确的缩放。32 位的累积舍入误差悄悄吃掉了你的精度。

或者：你从零实现了交叉熵损失。它对小的 logits 有效，但当 logits 超过 100 时，它返回 `inf`。因为 `exp(100)` 超过了 float32 能表示的范围，softmax 溢出了。每个 ML 框架都用两行代码的技巧处理这个问题。你不知道这个技巧的存在。

数值稳定性不是理论上的担忧，它决定了训练是成功还是无声地失败。你调试的每一个严重的 ML bug 最终都归结于浮点数。

## 概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 (IEEE 754) 标准将实数存储为浮点值。一个浮点数由三部分组成：符号位、指数和尾数（有效数）。

```
Float32 布局（共 32 位）：
[1 符号位] [8 指数位] [23 尾数位]

值 = (-1)^符号位 * 2^(指数 - 127) * 1.尾数
```

尾数决定精度（有多少有效数字），指数决定范围（数字可以多大或多小）。

```
格式        位数   指数位  尾数位  十进制有效位数  范围（约）
float64    64     11      52      ~15-16           +/- 1.8e308
float32    32     8       23      ~7-8             +/- 3.4e38
float16    16     5       10      ~3-4             +/- 65,504
bfloat16   16     8       7       ~2-3             +/- 3.4e38
```

float32 给你大约 7 位十进制精度，意味着它可以区分 1.0000001 和 1.0000002，但无法区分 1.00000001 和 1.00000002——超过 7 位就全是舍入噪声了。

float16 给你大约 3 位精度。它能表示的最大数字是 65,504，这在 ML 中令人不安地小，因为 logits、梯度和激活值经常超过这个值。

bfloat16 是 Google 解决 float16 范围问题的方案。它有与 float32 相同的 8 位指数（相同范围，可达 3.4e38），但只有 7 位尾数（比 float16 精度低）。对于训练神经网络，范围比精度更重要，所以 bfloat16 通常胜出。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法在二进制浮点中精确表示。在二进制中，它是一个循环小数：

```
二进制下的 0.1 = 0.0001100110011001100110011...（无限循环）
```

Float32 将它截断为 23 位尾数，存储值约为 0.100000001490116。类似地，0.2 存储约为 0.200000002980232。它们的和是 0.300000004470348，不是 0.3。

```
在 Python 中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 很重要，因为：

1. 如 `if loss < threshold` 这样的损失比较可能给出错误答案
2. 累积大量小数值（如数千步的梯度更新）会偏离真实和
3. 如果你用 `==` 比较浮点数，校验和与可复现性测试会失败

修复方法：永远不要用 `==` 比较浮点数。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消 (Catastrophic Cancellation)

当你减去两个几乎相等的浮点数时，有效数字相互抵消，舍入噪声被提升为前导数字。

```
a = 1.0000001    （在 float32 中存储为 1.00000011920929）
b = 1.0000000    （在 float32 中存储为 1.00000000000000）

真实差值：  0.0000001
计算值：    0.00000011920929

相对误差：19.2%
```

单次减法就产生了 19% 的相对误差。在 ML 中，这种情况发生在你：

- 计算均值很大的数据的方差：当 E[x] 很大时求 `E[x^2] - E[x]^2`
- 减去近乎相等的对数概率
- 使用过小的 epsilon 计算有限差分梯度

修复方法：重新组织公式避免减去大的、近乎相等的数。对于方差，使用 Welford 算法或先对数据中心化。对于对数概率，全程在 log 空间中计算。

### 溢出 (Overflow) 和下溢 (Underflow)

溢出发生在结果太大无法表示时，下溢发生在结果太小（比最小可表示正数更接近零）时。

```
Float32 边界：
  最大值：              3.4028235e+38
  最小正数（规格化）：    1.175e-38
  最小正数（非规格化）：  1.401e-45
  溢出：  任何 > 3.4e38 的值变为 inf
  下溢：  任何 < 1.4e-45 的值变为 0.0
```

`exp()` 函数是 ML 中溢出的主要源头：

```
exp(88.7)  = 3.40e+38   （勉强在 float32 范围内）
exp(89.0)  = inf         （溢出）
exp(-87.3) = 1.18e-38   （刚好不下溢）
exp(-104)  = 0.0         （下溢为零）
```

`log()` 函数则走向另一个方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （正常）
log(1e-46) = -inf        （输入下溢为零，然后 log(0) = -inf）
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。`log(exp(x))` 的组合如果没有正确的技巧，就是一个雷区。

### Log-Sum-Exp 技巧 (Log-Sum-Exp Trick)

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何 `x_i` 很大，`exp(x_i)` 会溢出。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 都下溢为零，`log(0)` 就是 `-inf`。

技巧：在指数化之前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去 `max(x)` 后，最大的指数是 `exp(0) = 1`，不可能溢出。和中至少有一项是 1，所以和至少为 1，且 `log(1) = 0`，不可能下溢为 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                  （加减 c）
= log(sum(exp(x_i - c) * exp(c)))             （exp(a+b) = exp(a)*exp(b)）
= log(exp(c) * sum(exp(x_i - c)))             （提取公因子 exp(c)）
= c + log(sum(exp(x_i - c)))                  （log(a*b) = log(a) + log(b)）
```

令 `c = max(x)` 即可消除溢出。

这个技巧在 ML 中无处不在：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 Softmax 需要最大值减法技巧

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有技巧时，logits [100, 101, 102] 会导致溢出：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

exp(88.7) 已经达到 float32 的极限。
exp(100) 在 float32 中 = inf。
```

使用技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全相同，计算安全。这不是优化，而是正确性的要求。

### NaN 和 Inf：检测与防范

`nan`（不是一个数字）和 `inf`（无穷）会在计算中病毒式传播。梯度更新中一个 `nan` 就让权重变成 `nan`，进而让后续每个输出变成 `nan`，训练一步就死了。

`inf` 的出现方式：
- 对大正数取 `exp()`
- 除零：`1.0 / 0.0`
- 累积中的 `float32` 溢出

`nan` 的出现方式：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数取 `sqrt()`
- 对负数取 `log()`
- 任何涉及已存在 `nan` 的算术运算

检测：

```python
import math

math.isnan(x)       # 如果 x 是 nan 则为 True
math.isinf(x)       # 如果 x 是 +inf 或 -inf 则为 True
math.isfinite(x)    # 如果 x 既不是 nan 也不是 inf 则为 True
```

防范策略：

1. 截断 `exp()` 的输入：`exp(clamp(x, -80, 80))`
2. 给分母加 epsilon：`x / (y + 1e-8)`
3. 给 `log()` 内部加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、稳定 softmax）
5. 梯度截断防止权重爆炸
6. 调试期间在每次前向传播后检查 `nan`/`inf`

### 数值梯度检查 (Numerical Gradient Checking)

解析梯度（来自反向传播）可能有 bug。数值梯度检查通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精度，远好于只有 O(h) 精度的前向差分 `(f(x+h) - f(x)) / h`。

选择 h：太大则近似不准确，太小则灾难性抵消破坏结果。通常取 `h = 1e-5` 到 `1e-7`。

检查方法：计算解析梯度和数值梯度之间的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，大概率正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

实现新层或损失函数时一定要检查梯度。PyTorch 为此提供了 `torch.autograd.gradcheck()`。

### 混合精度训练 (Mixed Precision Training)

现代 GPU 有专门的硬件（张量核心 Tensor Cores）能以 float32 的 2-8 倍速度计算 float16 矩阵乘法。混合精度训练利用这一点：

```
1. 维护 float32 主权重副本
2. 前向传播使用 float16（快）
3. 计算损失使用 float32（防止溢出）
4. 反向传播使用 float16（快）
5. 将梯度缩放回 float32
6. 更新 float32 主权重
```

纯 float16 训练的问题：梯度通常很小（1e-8 或更小），float16 将小于约 6e-8 的值都下溢为零，模型停止学习，因为所有梯度更新都是零。

修复方案是损失缩放 (Loss Scaling)：

```
1. 将损失乘以一个大缩放因子（如 1024）
2. 反向传播计算 (loss * 1024) 的梯度
3. 所有梯度都大了 1024 倍（推到 float16 下溢阈值以上）
4. 更新权重前将梯度除以 1024
5. 净效果：相同更新，但没有下溢
```

动态损失缩放 (Dynamic Loss Scaling) 自动调整缩放因子。从一个大的值（65536）开始，如果梯度溢出为 `inf` 就减半，如果 N 步没有溢出就加倍。

### bfloat16 vs float16：为什么 bfloat16 在训练中胜出

```
float16:   [1 符号位] [5 指数位]  [10 尾数位]
bfloat16:  [1 符号位] [8 指数位]  [7 尾数位]
```

float16 有更高精度（10 位尾数 vs 7 位）但范围受限（最大值 ~65,504）。bfloat16 精度较低但范围与 float32 相同（最大值 ~3.4e38）。

对于训练神经网络：

- 在训练过程中的激活值和 logits 经常超过 65,504。float16 溢出，bfloat16 能处理。
- float16 需要损失缩放，但 bfloat16 通常不需要，因为它的范围覆盖了梯度幅度谱。
- bfloat16 是 float32 的简单截断：丢弃尾数的低 16 位。转换简单，指数无损。

float16 在推理中（值有界、精度更重要）更受欢迎。bfloat16 在训练中（范围更重要）更受欢迎。这就是为什么 TPU 和现代 NVIDIA GPU（A100、H100）都有原生 bfloat16 支持。

### 梯度截断 (Gradient Clipping)

当梯度在多层之间呈指数增长（常见于 RNN、深度网络和 Transformer），就会发生梯度爆炸。单步大梯度就能毁掉所有权重。

两种截断方式：

**按值截断：** 独立截断每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单但可能改变梯度向量的方向。

**按范数截断：** 缩放整个梯度向量使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保持梯度方向不变。这就是 `torch.nn.utils.clip_grad_norm_()` 做的事情，是标准选择。

典型取值：Transformer 用 `max_norm=1.0`，强化学习用 `max_norm=0.5`，简单网络用 `max_norm=5.0`。

梯度截断不是 hack，而是一种安全机制。没有它，单次异常批次就可能产生大到足以毁掉数周训练的梯度。

### 归一化层作为数值稳定器

批归一化（Batch Normalization）、层归一化（Layer Normalization）和 RMS 归一化通常被描述为帮助训练收敛的正则化手段，但它们也是数值稳定器。

没有归一化，激活值在层间可能指数级增长或收缩：

```
第 1 层：值在 [0, 1]
第 5 层：值在 [0, 100]
第 10 层：值在 [0, 10,000]
第 50 层：值在 [0, inf]
```

归一化在每一层重新中心化和重缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常 1e-5）防止所有激活值相同时除零。可学习参数 `gamma` 和 `beta` 让网络恢复它需要的任何尺度。

这使值在整网络中保持在数值安全范围内，既防止前向传播中的溢出，也防止反向传播中的梯度爆炸。

### 常见 ML 数值错误

**Bug：损失在几个 epoch 后变成 NaN。**
原因：logits 增长过大，softmax 溢出；或者学习率太高，权重发散。
修复：使用稳定 softmax（最大值减法），降低学习率，添加梯度截断。

**Bug：损失卡在 log(num_classes)。**
原因：模型输出接近均匀概率，通常意味着梯度消失或模型根本没在学习。
修复：检查数据标签是否正确，验证损失函数，检查是否存在死 ReLU。

**Bug：验证精度比预期低 1-3%。**
原因：混合精度没有适当的损失缩放。梯度下溢悄悄将小更新归零。
修复：启用动态损失缩放，或者换用 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：死 ReLU 神经元（所有输入为负），或 float16 下溢。
修复：使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**Bug：模型在一个 GPU 上工作但在另一个上得到不同结果。**
原因：不确定性的浮点累积顺序。GPU 并行归约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
修复：接受微小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 接受速度损失。

**Bug：损失计算中 `exp()` 返回 `inf`。**
原因：原始 logits 传入 `exp()` 而没有使用最大值减法技巧。
修复：使用 `torch.nn.functional.log_softmax()`，它在内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示小于 6e-8 的梯度或大于 65,504 的激活值。
修复：使用带损失缩放的混合精度（AMP），或换用 bfloat16。

## 构建

### 步骤 1：演示浮点精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤 2：实现朴素 vs 稳定 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 步骤 3：实现稳定 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 步骤 4：实现稳定交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤 5：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度截断

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现（含所有边界情况演示）见 `code/numerical.py`。

## 交付

本课产出：
- `code/numerical.py`：包含稳定 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`：用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现将在第三阶段构建训练循环和第四阶段实现注意力机制时再次出现。

## 练习

1. **灾难性抵消。** 使用朴素公式 `E[x^2] - E[x]^2` 在 float32 中计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后使用 Welford 在线算法计算。比较与真实方差（0.6667）的误差。

2. **精度寻宝。** 找到最小的正 float32 值 `x`，使得 `1.0 + x == 1.0` 在 Python 中成立。这就是机器 epsilon。验证它是否匹配 `numpy.finfo(numpy.float32).eps`。

3. **Log-sum-exp 边界情况。** 测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他值，(c) 所有值非常负（-1000）。验证它在朴素版本失败的地方给出正确结果。

4. **检查神经网络层的梯度。** 实现一个单层线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 对 3x2 权重矩阵验证正确性。

5. **损失缩放实验。** 模拟 float16 训练：创建范围 [1e-9, 1e-3] 的随机梯度，转换为 float16，测量有多少比例变为零。然后应用损失缩放（乘以 1024），转换为 float16，缩放回来，再次测量归零比例。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每个现代 CPU 和 GPU 都实现了它。 |
| 机器 epsilon | "精度极限" | 在给定浮点格式下使得 1.0 + e != 1.0 的最小值 e。对 float32 约为 1.19e-7。 |
| 灾难性抵消 | "减法导致精度损失" | 当两个近乎相等的浮点数相减时，有效数字相互抵消，结果由舍入噪声主导。 |
| 溢出 | "数字太大" | 结果超出最大可表示值，变成 inf。exp(89) 在 float32 中溢出。 |
| 下溢 | "数字太小" | 结果比最小可表示正数更接近零，变成 0.0。exp(-104) 在 float32 中下溢。 |
| Log-sum-exp 技巧 | "先减最大值" | 通过提取 exp(max(x)) 来计算 log(sum(exp(x)))，防止溢出和下溢。用于 softmax、交叉熵和对数概率计算。 |
| 稳定 softmax | "不会爆炸的 softmax" | 在指数化之前减去 max(logits)。数值上结果相同，不可能溢出。 |
| 梯度检查 | "验证你的反向传播" | 将来自反向传播的解析梯度与来自有限差分的数值梯度进行比较，以发现实现 bug。 |
| 混合精度 | "float16 前向，float32 后向" | 对速度关键的操作使用低精度浮点，对数值敏感的操作使用高精度浮点。典型加速 2-3 倍。 |
| 损失缩放 | "防止梯度下溢" | 在反向传播前将损失乘以一个大常数，使梯度保持在 float16 可表示范围内，然后在权重更新前除以相同的常数。 |
| bfloat16 | "Brain 浮点" | Google 的 16 位格式，有 8 位指数（与 float32 相同范围）和 7 位尾数（比 float16 精度低）。训练中更受欢迎。 |
| 梯度截断 | "限制梯度范数" | 缩放梯度向量使其范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | "不是一个数字" | 来自未定义操作（0/0、inf-inf、sqrt(-1)）的特殊浮点值。会在所有后续算术运算中传播。 |
| Inf | "无穷" | 来自溢出或除零的特殊浮点值。可以组合产生 NaN（inf - inf、inf * 0）。 |
| 数值梯度 | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。慢但可靠，用于验证。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) —— 权威参考文献，密集但完整
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) —— 为 float16 训练引入损失缩放的 NVIDIA 论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) —— PyTorch 混合精度实践指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) —— Google 为什么为 TPU 选择这种格式
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) —— 减少浮点求和舍入误差的算法