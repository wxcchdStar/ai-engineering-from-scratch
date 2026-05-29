# AI 中的复数 (Complex Numbers for AI)

> -1 的平方根不是虚的。它是旋转、频率和信号处理半壁江山的关键。

**类型：** 学习 (Learn)
**语言：** Python
**前置要求：** 第一阶段，第 01-04 课（线性代数、微积分）
**时间：** ~60 分钟

## 学习目标

- 在直角坐标和极坐标形式下执行复数算术（加、乘、除、共轭）
- 应用欧拉公式 (Euler's Formula) 在复指数和三角函数之间转换
- 使用单位复根实现离散傅里叶变换 (DFT)
- 解释复数旋转如何支撑 Transformer 中的 RoPE 和正弦位置编码

## 问题

你打开一篇关于傅里叶变换的论文，到处都是 `i`。你查看 Transformer 的位置编码，看到不同频率的 `sin` 和 `cos`——复指数的实部和虚部。你阅读量子计算，发现一切都在复向量空间中表达。

复数看起来很抽象。一个建立在 -1 平方根上的数系感觉像数学把戏。但它不是把戏，它是旋转和振荡的自然语言。每当有东西旋转、振动或振荡时，复数就是正确的工具。

不理解复数，你就无法理解离散傅里叶变换，无法理解 FFT，无法理解 RoPE（旋转位置嵌入 Rotary Position Embedding）在现代语言模型中如何工作，无法理解为什么原始 Transformer 论文中的正弦位置编码使用那些特定的频率。

本课从零构建复数算术，将其与几何联系起来，并准确展示复数在机器学习中的出现位置。

## 概念

### 什么是复数？

复数有两部分：实部和虚部。

```
z = a + bi

其中：
  a 是实部
  b 是虚部
  i 是虚数单位，定义为 i^2 = -1
```

就是这样。你将数轴扩展为一个平面。实数位于一个轴上，虚数位于另一个轴上。每个复数都是这个平面中的一个点。

### 复数算术

**加法。** 实部加实部，虚部加虚部。

```
(a + bi) + (c + di) = (a + c) + (b + d)i

示例：(3 + 2i) + (1 + 4i) = 4 + 6i
```

**乘法。** 使用分配律并记住 i^2 = -1。

```
(a + bi)(c + di) = ac + adi + bci + bdi^2
                 = ac + adi + bci - bd
                 = (ac - bd) + (ad + bc)i

示例：(3 + 2i)(1 + 4i) = 3 + 12i + 2i + 8i^2
                        = 3 + 14i - 8
                        = -5 + 14i
```

**共轭。** 翻转虚部的符号。

```
(a + bi) 的共轭 = a - bi
```

一个复数与其共轭的乘积始终是实数：

```
(a + bi)(a - bi) = a^2 + b^2
```

**除法。** 分子和分母同时乘以分母的共轭。

```
(a + bi) / (c + di) = (a + bi)(c - di) / (c^2 + d^2)
```

这消除了分母中的虚部，给你一个干净的复数。

### 复平面

复平面将每个复数映射为一个 2D 点。水平轴是实轴，垂直轴是虚轴。

```
z = 3 + 2i  对应点 (3, 2)
z = -1 + 0i 对应实轴上的点 (-1, 0)
z = 0 + 4i  对应虚轴上的点 (0, 4)
```

一个复数同时是一个点和一个从原点出发的向量。这种双重解释使复数对几何有用。

### 极坐标形式

平面中的任何点都可以用其到原点的距离和与正实轴的夹角来描述。

```
z = r * (cos(theta) + i*sin(theta))

其中：
  r = |z| = sqrt(a^2 + b^2)     （模长 magnitude，或 modulus）
  theta = atan2(b, a)             （辐角 phase，或 argument）
```

直角坐标形式 (a + bi) 适合加法，极坐标形式 (r, theta) 适合乘法。

**极坐标形式的乘法。** 模长相乘，辐角相加。

```
z1 = r1 * e^(i*theta1)
z2 = r2 * e^(i*theta2)

z1 * z2 = (r1 * r2) * e^(i*(theta1 + theta2))
```

这就是为什么复数对旋转来说是完美的。乘以模长为 1 的复数就是纯旋转。

### 欧拉公式 (Euler's Formula)

复指数和三角函数之间的桥梁：

```
e^(i*theta) = cos(theta) + i*sin(theta)
```

这是本课最重要的公式。当 theta = pi 时：

```
e^(i*pi) = cos(pi) + i*sin(pi) = -1 + 0i = -1

因此：e^(i*pi) + 1 = 0
```

五个基本常数（e、i、pi、1、0）被一个方程联系起来。

### 为什么欧拉公式对 ML 重要

欧拉公式说 `e^(i*theta)` 随着 theta 变化在单位圆上描点。theta = 0 时在 (1, 0)，theta = pi/2 时在 (0, 1)，theta = pi 时在 (-1, 0)，theta = 3*pi/2 时在 (0, -1)。完整旋转是 theta = 2*pi。

这意味着复指数就是旋转。而旋转在信号处理和 ML 中无处不在。

### 与 2D 旋转的联系

将复数 (x + yi) 乘以 e^(i*theta) 将点 (x, y) 绕原点旋转角度 theta。

```
通过复数乘法的旋转：
  (x + yi) * (cos(theta) + i*sin(theta))
  = (x*cos(theta) - y*sin(theta)) + (x*sin(theta) + y*cos(theta))i

通过矩阵乘法的旋转：
  [cos(theta)  -sin(theta)] [x]   [x*cos(theta) - y*sin(theta)]
  [sin(theta)   cos(theta)] [y] = [x*sin(theta) + y*cos(theta)]
```

它们产生相同的结果。复数乘法就是 2D 旋转。旋转矩阵只是用矩阵符号写的复数乘法。

### 相量 (Phasors) 和旋转信号

复指数 e^(i*omega*t) 是一个以角频率 omega 绕单位圆旋转的点。随着 t 增加，该点描出圆。

这个旋转点的实部是 cos(omega*t)，虚部是 sin(omega*t)。正弦信号是旋转复数的影子。

```
e^(i*omega*t) = cos(omega*t) + i*sin(omega*t)

实部：      cos(omega*t)    —— 余弦波
虚部：      sin(omega*t)    —— 正弦波
```

这就是相量表示。与其追踪一条扭动的正弦波，不如追踪一个平滑旋转的箭头。相移变成角度偏移，幅度变化变成模长变化，信号相加变成向量加法。

### 单位根 (Roots of Unity)

N 次单位根是单位圆上均匀分布的 N 个点：

```
w_k = e^(2*pi*i*k/N)    对 k = 0, 1, 2, ..., N-1
```

对 N = 4，根是：1, i, -1, -i（四个罗盘点）。
对 N = 8，你得到四个罗盘点加上四个对角线点。

单位根是离散傅里叶变换的基础。DFT 将信号分解为这 N 个等间距频率上的分量。

### 与 DFT 的联系

信号 x[0], x[1], ..., x[N-1] 的离散傅里叶变换是：

```
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)
```

每个 X[k] 衡量信号与第 k 个单位根——频率为 k 的复正弦波——的相关程度。DFT 将信号分解为 N 个旋转相量，并告诉你每个相量的幅度和相位。

### 为什么 i 不是虚的

"虚数"这个词是历史偶然。笛卡尔轻蔑地使用了它。但 i 并不比负数在人们最初拒绝它们时更"虚"。负数回答"从 3 中减去 5 得到什么？"，虚数单位回答"什么数的平方是 -1？"

更有用的理解：i 是一个 90 度旋转算子。将实数乘以 i 一次，旋转 90 度到虚轴。再乘以 i（i^2），再旋转 90 度——现在你指向负实轴方向。这就是为什么 i^2 = -1。这不神秘，这是由两个四分之一转组成的半转。

这就是为什么复数在工程中无处不在。任何旋转的东西——电磁波、量子态、信号振荡、位置编码——都自然地用复数描述。

### 复指数 vs 三角函数

在欧拉公式之前，工程师将信号写为 A*cos(omega*t + phi)——幅度 A、频率 omega、相位 phi。这可行但使算术痛苦。将两个不同相位的余弦相加需要三角恒等式。

使用复指数，同样的信号是 A*e^(i*(omega*t + phi))。将两个信号相加只是将两个复数相加。乘法（调制）只是模长相乘、辐角相加。相移变成角度加法，频移变成乘以相量。

整个信号处理领域转向复指数符号，因为数学更简洁。"真实信号"始终只是复数表示的实部，虚部作为簿记被携带，使所有代数自然成立。

### 与 Transformer 的联系

**正弦位置编码**（原始 Transformer 论文）：

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

sin 和 cos 对是不同频率复指数的实部和虚部。每个频率为编码位置提供不同的"分辨率"。低频变化慢（粗粒度位置），高频变化快（细粒度位置）。它们一起给每个位置一个独特的频率指纹。

**RoPE（旋转位置嵌入 Rotary Position Embedding）** 更进一步。它显式地将查询和键向量乘以复数旋转矩阵。两个 token 之间的相对位置变成一个旋转角度。注意力使用这些旋转后的向量计算，使模型通过复数乘法对相对位置敏感。

| 操作 | 代数形式 | 几何含义 |
|------|---------|---------|
| 加法 | (a+c) + (b+d)i | 平面中的向量加法 |
| 乘法 | (ac-bd) + (ad+bc)i | 旋转并缩放 |
| 共轭 | a - bi | 关于实轴反射 |
| 模长 | sqrt(a^2 + b^2) | 到原点的距离 |
| 辐角 | atan2(b, a) | 与正实轴的夹角 |
| 除法 | 乘以共轭 | 反向旋转并重新缩放 |
| 幂 | r^n * e^(i*n*theta) | 旋转 n 次，缩放 r^n |

## 构建

### 步骤 1：复数类

构建一个支持算术、模长、辐角以及直角坐标和极坐标形式之间转换的复数类。

```python
import math

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return Complex(r, i)

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        return Complex(self.real, -self.imag)
```

### 步骤 2：极坐标转换和欧拉公式

```python
def to_polar(z):
    return z.magnitude(), z.phase()

def from_polar(r, theta):
    return Complex(r * math.cos(theta), r * math.sin(theta))

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))
```

### 步骤 3：单位根和 DFT

```python
def roots_of_unity(N):
    return [Complex(math.cos(2*math.pi*k/N), math.sin(2*math.pi*k/N))
            for k in range(N)]

def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            total = total + signal[n] * w
        result.append(total)
    return result
```

### 步骤 4：使用复数旋转

```python
def rotate_2d(point, angle):
    z = Complex(point[0], point[1])
    rotator = Complex(math.cos(angle), math.sin(angle))
    rotated = z * rotator
    return (rotated.real, rotated.imag)
```

完整实现见 `code/complex_numbers.py`。

## 使用

使用 NumPy 的生产版本：

```python
import numpy as np

z1 = 3 + 2j
z2 = 1 + 4j
print(f"Sum: {z1 + z2}")
print(f"Product: {z1 * z2}")
print(f"Magnitude: {abs(z1):.4f}")
print(f"Phase: {np.angle(z1):.4f}")

N = 8
roots = np.exp(2j * np.pi * np.arange(N) / N)
print(f"8th roots of unity: {roots}")

signal = np.array([1.0, 2.0, 3.0, 4.0])
dft_result = np.fft.fft(signal)
print(f"DFT: {dft_result}")
```

## 练习

1. 手工计算 (3 + 4i)(2 - i) 和 (3 + 4i)/(2 - i)。验证你的答案。

2. 使用欧拉公式推导 cos(2*theta) 和 sin(2*theta) 的倍角公式。提示：e^(i*2*theta) = (e^(i*theta))^2。

3. 实现一个函数，将点 (x, y) 绕原点旋转 45 度。在点 (1, 0)、(0, 1) 和 (1, 1) 上测试。验证旋转后的点与预期位置匹配。

4. 为 N = 4 和 N = 8 计算单位根。在复平面上绘制它们。验证它们均匀分布在单位圆上。

5. 使用你的 Complex 类实现 DFT，并在简单信号上测试：[1, 0, 0, 0]（脉冲）、[1, 1, 1, 1]（常数）、[1, -1, 1, -1]（最高频率）。解释每个信号的 DFT 输出。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| 复数 | "实部加虚部" | 形式为 a + bi 的数，其中 i^2 = -1。将数轴扩展为平面 |
| 虚数单位 i | "-1 的平方根" | 90 度旋转算子。i^2 = -1 因为两次 90 度旋转 = 180 度 |
| 复平面 | "实轴和虚轴" | 复数的 2D 表示。水平 = 实部，垂直 = 虚部 |
| 共轭 | "翻转虚部符号" | a + bi 的共轭是 a - bi。关于实轴的反射 |
| 模长 | "到原点的距离" | sqrt(a^2 + b^2)。复数的大小 |
| 辐角 | "与实轴的夹角" | atan2(b, a)。复数的角度 |
| 极坐标形式 | "模长和角度" | z = r * e^(i*theta)。对乘法有用 |
| 欧拉公式 | "指数 = 三角函数" | e^(i*theta) = cos(theta) + i*sin(theta)。连接指数和三角函数 |
| 相量 | "旋转箭头" | 表示正弦信号的复指数。将振荡转化为旋转 |
| 单位根 | "单位圆上的等间距点" | e^(2*pi*i*k/N)。DFT 的基础 |
| DFT | "离散傅里叶变换" | 将信号分解为复正弦波。使用单位根 |
| RoPE | "旋转位置嵌入" | 使用复数旋转编码相对位置。在现代 LLM 中使用 |
| 正弦位置编码 | "sin/cos 位置编码" | 原始 Transformer 使用不同频率的 sin/cos 对来编码位置 |

## 延伸阅读

- [Visual Complex Analysis (Needham)](https://global.oup.com/academic/product/visual-complex-analysis-9780198534464) —— 复数的几何直觉，配有精美插图
- [3Blue1Brown: Euler's formula with introductory group theory](https://www.3blue1brown.com/lessons/eulers-formula-via-group-theory) —— 欧拉公式的直观视觉解释
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864) —— 引入 RoPE 的论文