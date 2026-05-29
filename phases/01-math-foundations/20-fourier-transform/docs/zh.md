# 傅里叶变换 (The Fourier Transform)

> 每个信号都是正弦波之和。傅里叶变换告诉你有哪些正弦波。

**类型：** 构建 (Build)
**语言：** Python
**前置要求：** 第一阶段，第 01-04 课、第 19 课（复数）
**时间：** ~90 分钟

## 学习目标

- 从零实现 DFT，并对照 O(N log N) 的 Cooley-Tukey FFT 进行验证
- 解释频率系数：从信号中提取幅度、相位和功率谱
- 应用卷积定理通过 FFT 乘法执行卷积
- 将傅里叶频率分解与 Transformer 位置编码和 CNN 卷积层联系起来

## 问题

音频录音是随时间变化的压力测量序列。股票价格是随时间变化的值序列。图像是空间上的像素强度网格。所有这些都是时域（或空间域）中的数据——你看到值随某个索引变化。

但许多模式在时域中是不可见的。这个音频信号是纯音还是和弦？这个股票价格有周周期吗？这个图像有重复纹理吗？这些问题是关于频率内容的，而时域隐藏了它。

傅里叶变换将数据从时域转换到频域。它取一个信号并将其分解为不同频率的正弦波。每个正弦波有幅度（多强）和相位（从哪里开始）。傅里叶变换告诉你两者。

这对 ML 很重要，因为频域思维无处不在。卷积神经网络执行卷积，而卷积在频域中是乘法。Transformer 位置编码使用频率分解来表示位置。音频模型（语音识别、音乐生成）在频谱图——声音的频率表示——上操作。时间序列模型寻找周期模式。理解傅里叶变换给你处理所有这些的词汇。

## 概念

### DFT 定义

给定 N 个样本 x[0], x[1], ..., x[N-1]，离散傅里叶变换产生 N 个频率系数 X[0], X[1], ..., X[N-1]：

```
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)

对 k = 0, 1, ..., N-1
```

每个 X[k] 是一个复数。其模长 |X[k]| 告诉你频率 k 的幅度，其辐角 angle(X[k]) 告诉你该频率的相位偏移。

关键洞察：`e^(-2*pi*i*k*n/N)` 是频率为 k 的旋转相量。DFT 计算信号与 N 个等间距频率中每个频率之间的相关性。如果信号在频率 k 处包含能量，相关性就大；如果不包含，就接近零。

### 每个系数的含义

**X[0]：直流分量 (DC Component)。** 这是所有样本之和——与均值成正比。它表示信号的常数（零频率）偏移。

```
X[0] = sum_{n=0}^{N-1} x[n] * e^0 = 所有样本之和
```

**X[k]，1 <= k <= N/2：正频率。** X[k] 表示每 N 个样本中 k 个周期的频率。k 越高意味着频率越高（振荡越快）。

**X[N/2]：奈奎斯特频率 (Nyquist Frequency)。** 用 N 个样本能表示的最高频率。超过这个频率，你会得到混叠 (aliasing)——高频伪装成低频。

**X[k]，N/2 < k < N：负频率。** 对于实值信号，X[N-k] = conj(X[k])。负频率是正频率的镜像。这就是为什么有用信息在前 N/2 + 1 个系数中。

### 逆 DFT

逆 DFT 从频率系数重建原始信号：

```
x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^(2*pi*i*k*n/N)

对 n = 0, 1, ..., N-1
```

与正向 DFT 的唯一区别：指数中的符号是正的（不是负的），并且有一个 1/N 的归一化因子。

逆 DFT 是完美重建。没有信息丢失。你可以从时域到频域再回来，没有任何误差。DFT 是基变换——它在不同的坐标系中重新表达相同的信息。

### FFT：让它变快

上面定义的 DFT 是 O(N^2)：对 N 个输出系数中的每一个，你对 N 个输入样本求和。对 N = 100 万，那是 10^12 次操作。

快速傅里叶变换 (FFT) 在 O(N log N) 内计算相同的结果。对 N = 100 万，那是大约 2000 万次操作而不是一万亿次。这就是使频率分析变得实用的原因。

Cooley-Tukey 算法（最常见的 FFT）通过分治工作：

1. 将信号分为偶数索引和奇数索引样本。
2. 递归计算每半的 DFT。
3. 使用"旋转因子" e^(-2*pi*i*k/N) 组合两个半尺寸 DFT。

```
X[k] = E[k] + e^(-2*pi*i*k/N) * O[k]          对 k = 0, ..., N/2 - 1
X[k + N/2] = E[k] - e^(-2*pi*i*k/N) * O[k]    对 k = 0, ..., N/2 - 1

其中 E = 偶数索引样本的 DFT
      O = 奇数索引样本的 DFT
```

对称性意味着每层递归做 O(N) 的工作，共有 log2(N) 层。总计：O(N log N)。

FFT 要求信号长度是 2 的幂。实践中，信号被零填充到下一个 2 的幂。

### 频谱分析

**功率谱 (Power Spectrum)** 是 |X[k]|^2——每个频率系数模长的平方。它显示每个频率有多少能量。

**相位谱 (Phase Spectrum)** 是 angle(X[k])——每个频率的相位偏移。对于大多数分析任务，你关心功率谱而忽略相位。

```
频率 k 处的功率：  P[k] = |X[k]|^2 = X[k].real^2 + X[k].imag^2
频率 k 处的相位：  phi[k] = atan2(X[k].imag, X[k].real)
```

### 频率分辨率

DFT 的频率分辨率取决于样本数 N 和采样率 fs。

```
bin k 的频率：          f_k = k * fs / N
频率分辨率：            delta_f = fs / N
最大频率：              f_max = fs / 2  （奈奎斯特）
```

要分辨两个接近的频率，你需要更多样本。要捕获高频，你需要更高的采样率。

### 卷积定理

这是信号处理中最重要的结果之一，与 CNN 直接相关。

**时域中的卷积等于频域中的逐点乘法。**

```
x * h = IFFT(FFT(x) . FFT(h))

其中 * 是卷积，. 是逐元素乘法
```

为什么这很重要：

- 长度为 N 和 M 的两个信号的直接卷积需要 O(N*M) 次操作。
- 基于 FFT 的卷积需要 O(N log N)：变换两者，相乘，变换回来。
- 对于大核，FFT 卷积快得多。
- 这正是具有大感受野的卷积层中发生的事情。

注意：DFT 计算循环卷积（信号环绕）。对于线性卷积（无环绕），在计算前将两个信号零填充到长度 N + M - 1。

### 加窗 (Windowing)

DFT 假设信号是周期的——它将 N 个样本视为无限重复信号的一个周期。如果信号在开始和结束处不是相同的值，这会在边界处产生不连续性，表现为虚假的高频内容。这称为频谱泄漏 (Spectral Leakage)。

加窗通过在 DFT 之前将信号两端逐渐减小到零来减少泄漏。

常见窗函数：

| 窗函数 | 形状 | 主瓣宽度 | 旁瓣水平 | 用例 |
|--------|------|---------|---------|------|
| 矩形窗 | 平坦（无窗） | 最窄 | 最高 (-13 dB) | 当信号在 N 个样本中恰好是周期的 |
| Hann 窗 | 升余弦 | 中等 | 低 (-31 dB) | 通用频谱分析 |
| Hamming 窗 | 修正余弦 | 中等 | 更低 (-42 dB) | 音频处理、语音分析 |
| Blackman 窗 | 三重余弦 | 宽 | 非常低 (-58 dB) | 当旁瓣抑制至关重要时 |

```
Hann 窗：    w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
Hamming 窗： w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
```

在 DFT 之前将窗与信号逐元素相乘：`X = DFT(x * w)`。

### DFT 性质

| 性质 | 时域 | 频域 |
|------|------|------|
| 线性 | a*x + b*y | a*X + b*Y |
| 时移 | x[n - k] | X[f] * e^(-2*pi*i*f*k/N) |
| 频移 | x[n] * e^(2*pi*i*f0*n/N) | X[f - f0] |
| 卷积 | x * h | X * H（逐点） |
| 乘法 | x * h（逐点） | X * H（循环卷积，缩放 1/N） |
| Parseval 定理 | sum \|x[n]\|^2 | (1/N) * sum \|X[k]\|^2 |
| 共轭对称（实输入） | x[n] 为实 | X[k] = conj(X[N-k]) |

Parseval 定理说总能量在两个域中相同。能量在变换中守恒。

### 与位置编码的联系

原始 Transformer 使用正弦位置编码：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

每个维度对 (2i, 2i+1) 以不同频率振荡。频率从高（维度 0,1）到低（最后维度）呈几何间隔分布。这给每个位置一个跨所有频带的独特模式——类似于傅里叶系数如何唯一地标识一个信号。

这提供的关键性质：

- **唯一性：** 没有两个位置有相同的编码。
- **有界值：** sin 和 cos 始终在 [-1, 1] 内。
- **相对位置：** 位置 p+k 的编码可以表示为位置 p 编码的线性函数。模型可以学习关注相对位置。

### 与 CNN 的联系

卷积层通过在信号或图像上滑动学习到的滤波器（核）来应用它。数学上，这就是卷积操作。

根据卷积定理，这等价于：
1. 对输入做 FFT
2. 对核做 FFT
3. 在频域中相乘
4. 对结果做 IFFT

标准 CNN 实现使用直接卷积（对小 3x3 核更快）。但对于大核或全局卷积，基于 FFT 的方法快得多。某些架构（如 FNet）完全用 FFT 替代注意力，以 O(N log N) 而非 O(N^2) 的复杂度实现有竞争力的精度。

### 频谱图和短时傅里叶变换 (STFT)

单次 FFT 给你整个信号的频率内容，但不告诉你这些频率何时出现。一个啁啾信号（频率随时间增加的信号）和一个和弦（所有频率同时出现）可以有相同的幅度谱。

短时傅里叶变换 (STFT) 通过在信号的重叠窗口上计算 FFT 来解决这个问题。结果是一个频谱图 (Spectrogram)：一个 2D 表示，时间在一个轴上，频率在另一个轴上。每个点的强度显示该时间该频率的能量。

```
STFT 流程：
1. 选择窗口大小（如 1024 个样本）
2. 选择跳跃大小（如 256 个样本——75% 重叠）
3. 对每个窗口位置：
   a. 提取加窗片段
   b. 应用 Hann/Hamming 窗
   c. 计算 FFT
   d. 将幅度谱存储为频谱图的一列
```

频谱图是音频 ML 模型的标准输入表示。语音识别模型（Whisper、DeepSpeech）在梅尔频谱图 (mel-spectrograms) 上操作——频率映射到梅尔刻度的频谱图，这更好地匹配人类音高感知。

### 混叠 (Aliasing)

如果信号包含高于 fs/2（奈奎斯特频率）的频率，以速率 fs 采样将产生混叠副本。以 100 Hz 采样的 90 Hz 信号看起来与 10 Hz 信号完全相同。仅从样本无法区分它们。

```
示例：
  真实信号：90 Hz 正弦波
  采样率：100 Hz
  表观频率：100 - 90 = 10 Hz

  以 100 Hz 采样率采样的 90 Hz 信号的样本
  与 10 Hz 信号的样本完全相同。
  没有任何数学方法能恢复原始的 90 Hz。
```

这就是为什么模数转换器包含抗混叠滤波器，在采样前移除高于奈奎斯特的频率。在 ML 中，当没有适当低通滤波就下采样特征图时会出现混叠——某些架构通过抗混叠池化层来解决这个问题。

### 零填充不增加分辨率

一个常见误解：在 FFT 之前零填充信号能提高频率分辨率。它不能。零填充在现有频率 bin 之间插值，给你一个看起来更平滑的频谱。但它不能揭示原始样本中不存在的频率细节。

真正的频率分辨率仅取决于观测时间 T = N / fs。要分辨两个相隔 delta_f 的频率，你至少需要 T = 1 / delta_f 秒的数据。无论多少零填充都不能改变这个基本限制。

## 构建

### 步骤 1：从零实现 DFT

O(N^2) 的 DFT 直接从定义得出。

```python
import math

class Complex:
    ...

def dft(x):
    N = len(x)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            xn = x[n] if isinstance(x[n], Complex) else Complex(x[n])
            total = total + xn * w
        result.append(total)
    return result
```

### 步骤 2：Cooley-Tukey FFT

```python
def fft(x):
    N = len(x)
    if N <= 1:
        return [x[0] if isinstance(x[0], Complex) else Complex(x[0])]
    even = fft(x[0::2])
    odd = fft(x[1::2])
    result = [Complex(0, 0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = Complex(math.cos(angle), math.sin(angle))
        result[k] = even[k] + twiddle * odd[k]
        result[k + N // 2] = even[k] - twiddle * odd[k]
    return result
```

### 步骤 3：频谱分析

```python
def power_spectrum(X):
    return [x.magnitude() ** 2 for x in X]

def phase_spectrum(X):
    return [x.phase() for x in X]
```

### 步骤 4：基于 FFT 的卷积

```python
def fft_convolve(a, b):
    n = len(a) + len(b) - 1
    n_padded = 1
    while n_padded < n:
        n_padded *= 2
    a_pad = a + [0] * (n_padded - len(a))
    b_pad = b + [0] * (n_padded - len(b))
    A = fft(a_pad)
    B = fft(b_pad)
    C = [A[i] * B[i] for i in range(n_padded)]
    c = ifft(C)
    return [z.real for z in c[:n]]
```

完整实现见 `code/fourier.py`。

## 使用

使用 NumPy 的生产版本：

```python
import numpy as np

t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)

X = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), t[1] - t[0])

power = np.abs(X) ** 2
top_idx = np.argsort(power)[-5:]
for idx in top_idx:
    print(f"Frequency: {freqs[idx]:.1f} Hz, Power: {power[idx]:.2f}")
```

## 练习

1. 实现 DFT 和逆 DFT。验证对随机信号，`idft(dft(x))` 在浮点容差内恢复 x。

2. 创建一个包含两个频率（如 5 Hz 和 20 Hz）的信号。计算 DFT 并绘制功率谱。验证峰值出现在正确的频率处。

3. 使用 FFT 卷积实现一个简单的低通滤波器：创建一个高斯核，通过 FFT 将其与信号卷积，比较结果与直接卷积。

4. 创建一个啁啾信号（频率随时间线性增加）。计算其 STFT 并绘制频谱图。验证频率随时间增加。

5. 实现正弦位置编码函数。绘制不同位置的编码向量。验证位置 p+k 的编码是位置 p 编码的线性函数。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| DFT | "离散傅里叶变换" | 将 N 个时域样本转换为 N 个频域系数。O(N^2) 直接计算 |
| FFT | "快速傅里叶变换" | Cooley-Tukey 分治算法。O(N log N)。使频率分析实用 |
| 直流分量 (DC) | "零频率" | X[0]，与信号均值成正比。信号的常数偏移 |
| 奈奎斯特频率 | "最高可表示频率" | fs/2。高于此频率会出现混叠 |
| 功率谱 | "每个频率的能量" | \|X[k]\|^2。显示每个频率有多少能量 |
| 相位谱 | "每个频率的偏移" | angle(X[k])。每个频率分量的相位偏移 |
| 卷积定理 | "时域卷积 = 频域乘法" | x * h = IFFT(FFT(x) · FFT(h))。使快速卷积成为可能 |
| 频谱泄漏 | "能量扩散到相邻 bin" | 当信号在边界处不连续时发生。加窗减少泄漏 |
| 加窗 | "在 FFT 前逐渐减小到零" | 将信号乘以窗函数以减少频谱泄漏 |
| STFT | "短时傅里叶变换" | 在重叠窗口上计算 FFT。产生频谱图（时间 vs 频率） |
| 频谱图 | "频率随时间变化的热力图" | STFT 的 2D 输出。音频 ML 的标准输入 |
| 混叠 | "高频伪装成低频" | 当采样率太低时发生。需要抗混叠滤波器 |
| 零填充 | "在 FFT 前添加零" | 在现有 bin 之间插值。不提高真实频率分辨率 |
| 旋转因子 | "FFT 中的复指数" | e^(-2*pi*i*k/N)。Cooley-Tukey 中用于组合半尺寸 DFT |

## 延伸阅读

- [The Scientist and Engineer's Guide to DSP (Smith)](https://www.dspguide.com/) —— 免费在线，最好的 DSP 入门书
- [FNet: Mixing Tokens with Fourier Transforms (Lee-Thorp et al., 2022)](https://arxiv.org/abs/2105.03824) —— 用 FFT 替代 Transformer 中的注意力
- [3Blue1Brown: But what is the Fourier Transform?](https://www.3blue1brown.com/lessons/fourier-transforms) —— 傅里叶变换的直观视觉解释