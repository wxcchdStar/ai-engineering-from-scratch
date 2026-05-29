# 视频理解（Video Understanding）— 时序建模（Temporal Modeling）

> 视频是一系列图像加上连接它们的物理规律。每个视频模型要么将时间视为额外轴（3D 卷积），要么视为需要关注的序列（Transformer），要么视为提取一次然后池化的特征（2D+池化）。

**类型：** 学习 + 构建（Learn + Build）
**语言：** Python
**前置知识：** 第 4 阶段第 03 课（CNN）、第 4 阶段第 04 课（图像分类）
**时间：** 约 45 分钟

## 学习目标

- 区分三种主要的视频建模方法（2D+池化、3D 卷积、时空 Transformer）并预测它们的成本和准确率权衡
- 在 PyTorch 中实现帧采样（Frame Sampling）、时序池化（Temporal Pooling）和一个 2D+池化基线分类器
- 解释为什么 I3D 的"膨胀（Inflated）"3D 卷积核能从 ImageNet 权重良好迁移，以及分解式 (2+1)D 卷积的不同之处
- 阅读标准动作识别数据集和指标：Kinetics-400/600、UCF101、Something-Something V2；片段级（Clip-Level）和视频级（Video-Level）的 top-1 准确率

## 问题

一段 30 秒、30 fps 的视频是 900 张图像。朴素地看，视频分类就是运行 900 次图像分类然后进行某种聚合。当动作在几乎每一帧中都可见时（体育、烹饪、健身视频），这种方法有效；但当动作本身由运动定义时，它严重失败："将某物从左推到右"在每一帧中看起来都是两个静止的物体。

每个视频架构的核心问题是：时序结构（Temporal Structure）在何时被建模，以及如何建模？答案驱动着其他一切 — 计算成本、预训练策略、能否复用 ImageNet 权重、模型在哪些数据集上训练。

本课有意比静态图像课程更短。核心图像机制已经就位，视频理解主要关乎时序故事：采样、建模和聚合。

## 概念

### 三种架构家族

```mermaid
flowchart LR
    V["视频片段<br/>（T 帧）"] --> A1["2D + 池化<br/>每帧运行 2D CNN，<br/>在时间上取平均"]
    V --> A2["3D 卷积<br/>在<br/>T x H x W 上卷积"]
    V --> A3["时空<br/>Transformer<br/>对 (t, h, w)<br/>token 做注意力"]

    A1 --> C["Logits"]
    A2 --> C
    A3 --> C

    style A1 fill:#dbeafe,stroke:#2563eb
    style A2 fill:#fef3c7,stroke:#d97706
    style A3 fill:#dcfce7,stroke:#16a34a
```

### 2D + 池化

取一个 2D CNN（ResNet、EfficientNet、ViT）。在每一帧采样帧上独立运行。对每帧嵌入取平均（或最大池化，或注意力池化）。将池化后的向量输入分类器。

优点：
- ImageNet 预训练直接迁移。
- 实现最简单。
- 便宜：T 帧 * 单图像推理成本。

缺点：
- 无法建模运动。动作 = 外观的聚合。
- 时序池化是顺序无关的；"开门"和"关门"看起来一样。

适用场景：外观为主的任务、小型视频数据集上的迁移学习、初始基线。

### 3D 卷积

将 2D (H, W) 卷积核替换为 3D (T, H, W) 卷积核。网络在空间和时间上同时卷积。早期家族：C3D、I3D、SlowFast。

I3D 技巧：取一个预训练的 2D ImageNet 模型，通过沿新的时间轴复制每个 2D 卷积核来"膨胀"它。一个 3x3 的 2D 卷积变成 3x3x3 的 3D 卷积。这为 3D 模型提供了强大的预训练权重，而不是从头训练。

优点：
- 直接建模运动。
- I3D 膨胀提供免费的迁移学习。

缺点：
- 比 2D 对应模型多 T/8 的 FLOPs（对于堆叠 3 次的时序卷积核为 3）。
- 时序卷积核很小；长程运动需要金字塔或双流方法。

适用场景：运动是信号的动作识别（Something-Something V2、运动密集型类别的 Kinetics）。

### 时空 Transformer

将视频标记化为时空块（Space-Time Patch）的网格，并在所有块之间进行注意力计算。TimeSformer、ViViT、Video Swin、VideoMAE。

重要的注意力模式：
- **联合（Joint）** — 对 (t, h, w) 做一次大注意力。与 `T*H*W` 呈二次方；昂贵。
- **分离（Divided）** — 每个块两次注意力：一次在时间上，一次在空间上。近似线性缩放。
- **分解（Factorised）** — 时间注意力与空间注意力跨块交替。

优点：
- 在每个主要基准上达到 SOTA 准确率。
- 通过块膨胀从图像 Transformer（ViT）迁移。
- 通过稀疏注意力支持长上下文视频。

缺点：
- 计算量大。
- 需要仔细选择注意力模式，否则运行时会爆炸。

适用场景：大数据集、高保真视频理解、多模态视频+文本任务。

### 帧采样

一段 10 秒、30 fps 的片段是 300 帧；将所有 300 帧输入任何模型都是浪费。标准策略：

- **均匀采样（Uniform Sampling）** — 在片段中均匀选取 T 帧。2D+池化的默认方式。
- **密集采样（Dense Sampling）** — 随机连续的 T 帧窗口。3D 卷积常用，因为运动需要相邻帧。
- **多片段（Multi-Clip）** — 从同一视频采样多个 T 帧窗口，分别分类，测试时平均预测。

T 通常为 8、16、32 或 64。更高的 T = 更多的时序信号，但计算量更大。

### 评估

两个级别：
- **片段级准确率（Clip-Level Accuracy）** — 模型看到一个 T 帧片段，报告 top-k。
- **视频级准确率（Video-Level Accuracy）** — 对每个视频的多个片段取平均片段级预测；更高且更稳定。

始终报告两者。一个得分 78% 片段 / 82% 视频的模型严重依赖测试时平均；一个得分 80% / 81% 的模型在每个片段上更鲁棒。

### 你会遇到的数据集

- **Kinetics-400 / 600 / 700** — 通用动作数据集。40 万个片段；YouTube URL（许多现已失效）。
- **Something-Something V2** — 运动定义的动作（"将 X 从左移到右"）。无法用 2D+池化解决。
- **UCF-101**、**HMDB-51** — 更旧、更小，仍被报告。
- **AVA** — 空间和时间上的动作*定位*；比分类更难。

## 构建它

### 步骤 1：帧采样器

在帧列表（或视频张量）上工作的均匀和密集采样器。

```python
import numpy as np

def sample_uniform(num_frames_total, T):
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    step = num_frames_total / T
    return [int(i * step) for i in range(T)]


def sample_dense(num_frames_total, T, rng=None):
    rng = rng or np.random.default_rng()
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    start = int(rng.integers(0, num_frames_total - T + 1))
    return list(range(start, start + T))
```

两者都返回 `T` 个索引，用于切片视频张量。

### 步骤 2：2D+池化基线

在每一帧上运行 2D ResNet-18，平均池化特征，分类。

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FramePool(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))  # 保留全局平均池化
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (N, T, 3, H, W)
        N, T = x.shape[:2]
        x = x.view(N * T, *x.shape[2:])
        feats = self.features(x).view(N, T, -1)
        pooled = feats.mean(dim=1)
        return self.head(pooled)

model = FramePool(num_classes=10)
x = torch.randn(2, 8, 3, 224, 224)
print(f"output: {model(x).shape}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
```

一千一百万参数，ImageNet 预训练，逐帧运行，平均，分类。这个基线在外观为主的任务上通常与真正的 3D 模型相差 5-10 个点以内 — 有时甚至更好，因为它复用了更强的 ImageNet 骨干网络。

### 步骤 3：I3D 风格的膨胀 3D 卷积

通过沿新的时间轴重复权重，将单个 2D 卷积变为 3D 卷积。

```python
def inflate_2d_to_3d(conv2d, time_kernel=3):
    out_c, in_c, kh, kw = conv2d.weight.shape
    weight_3d = conv2d.weight.data.unsqueeze(2)  # (out, in, 1, kh, kw)
    weight_3d = weight_3d.repeat(1, 1, time_kernel, 1, 1) / time_kernel
    conv3d = nn.Conv3d(in_c, out_c, kernel_size=(time_kernel, kh, kw),
                        padding=(time_kernel // 2, conv2d.padding[0], conv2d.padding[1]),
                        stride=(1, conv2d.stride[0], conv2d.stride[1]),
                        bias=False)
    conv3d.weight.data = weight_3d
    return conv3d

conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
conv3d = inflate_2d_to_3d(conv2d, time_kernel=3)
print(f"2D weight shape:  {tuple(conv2d.weight.shape)}")
print(f"3D weight shape:  {tuple(conv3d.weight.shape)}")
x = torch.randn(1, 3, 8, 56, 56)
print(f"3D output shape:  {tuple(conv3d(x).shape)}")
```

除以 `time_kernel` 保持激活值幅度大致恒定 — 这对于不在第一次传递时破坏批归一化统计量很重要。

### 步骤 4：分解式 (2+1)D 卷积

将 3D 卷积拆分为 2D（空间）和 1D（时序）卷积。相同的感受野，更少的参数，在某些基准上更好的准确率。

```python
class Conv2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        mid_c = (in_c * out_c * kernel_size * kernel_size * kernel_size) \
                // (in_c * kernel_size * kernel_size + out_c * kernel_size)
        self.spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1, kernel_size, kernel_size),
                                 padding=(0, kernel_size // 2, kernel_size // 2), bias=False)
        self.bn = nn.BatchNorm3d(mid_c)
        self.act = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(mid_c, out_c, kernel_size=(kernel_size, 1, 1),
                                  padding=(kernel_size // 2, 0, 0), bias=False)

    def forward(self, x):
        return self.temporal(self.act(self.bn(self.spatial(x))))

c = Conv2Plus1D(3, 64)
x = torch.randn(1, 3, 8, 56, 56)
print(f"(2+1)D output: {tuple(c(x).shape)}")
```

完整的 R(2+1)D 网络与 ResNet-18 相同，只是将每个 3x3 卷积替换为 `Conv2Plus1D`。

## 使用它

两个库覆盖了生产级视频工作：

- `torchvision.models.video` — R(2+1)D、MViT、Swin3D，带有预训练的 Kinetics 权重。与图像模型相同的 API。
- `pytorchvideo`（Meta）— 模型库、Kinetics / SSv2 / AVA 的数据加载器、标准变换。

对于视觉-语言视频模型（视频字幕、视频问答），使用 `transformers`（`VideoMAE`、`VideoLLaMA`、`InternVideo`）。

## 交付它

本课产出：

- `outputs/prompt-video-architecture-picker.md` — 一个提示词，根据外观 vs 运动、数据集大小和计算预算选择 2D+池化 / I3D / (2+1)D / Transformer。
- `outputs/skill-frame-sampler-auditor.md` — 一个技能，检查视频流水线的采样器并标记常见错误：差一错误索引、`num_frames < T` 时的不均匀采样、缺少保持宽高比的裁剪等。

## 练习

1. **（简单）** 计算 T=8 的 FramePool 与 T=8 的 I3D 风格 3D ResNet 的 FLOPs（近似）。论证为什么 2D+池化便宜 3-5 倍。
2. **（中等）** 生成一个合成视频数据集：随机球体向随机方向移动，按运动方向标记（"从左到右"、"从右到左"、"对角线向上"）。在其上训练 FramePool。展示它达到接近随机猜测的准确率，证明仅靠外观不足以完成运动任务。
3. **（困难）** 通过将 ResNet-18 中的每个 Conv2d 替换为 `Conv2Plus1D` 来构建 R(2+1)D-18。从 ImageNet 预训练的 ResNet-18 膨胀第一个卷积的权重。在练习 2 的运动数据集上训练并击败 FramePool。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 2D + 池化（2D + Pool） | "逐帧分类器" | 在每帧采样帧上运行 2D CNN，跨时间平均池化特征，分类 |
| 3D 卷积（3D Convolution） | "时空卷积核" | 在 (T, H, W) 上卷积的卷积核；可以原生建模运动 |
| 膨胀（Inflation） | "将 2D 权重提升到 3D" | 通过沿新的时间轴重复 2D 卷积的权重来初始化 3D 卷积权重，然后除以 kernel_T 以保持激活值尺度 |
| (2+1)D | "分解式卷积" | 将 3D 拆分为 2D 空间 + 1D 时序；更少参数，中间有额外的非线性 |
| 分离注意力（Divided Attention） | "先时间后空间" | 每层两次注意力的 Transformer 块：一次对同一帧的 token，一次对同一位置的 token |
| 片段（Clip） | "T 帧窗口" | 采样的 T 帧子序列；视频模型消费的单元 |
| 片段 vs 视频准确率 | "两种评估设置" | 片段 = 每个视频一个样本，视频 = 多个采样片段的平均 |
| Kinetics | "视频的 ImageNet" | 400-700 个动作类别，30 万+ YouTube 片段，标准视频预训练语料库 |

## 扩展阅读

- [I3D: Quo Vadis, Action Recognition (Carreira & Zisserman, 2017)](https://arxiv.org/abs/1705.07750) — 引入膨胀和 Kinetics 数据集
- [R(2+1)D: A Closer Look at Spatiotemporal Convolutions (Tran et al., 2018)](https://arxiv.org/abs/1711.11248) — 分解式卷积，至今仍是强大的基线
- [TimeSformer: Is Space-Time Attention All You Need? (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) — 第一个强大的视频 Transformer
- [VideoMAE (Tong et al., 2022)](https://arxiv.org/abs/2203.12602) — 视频的掩码自编码器预训练；当前主流的预训练方案
