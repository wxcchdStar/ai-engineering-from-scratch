# 图像分类（Image Classification）

> 分类器是从像素到类别概率分布的函数。其他一切都是管道。

**类型：** 构建（Build）
**语言：** Python
**前置要求：** 第二阶段第 9 课（模型评估），第三阶段第 10 课（迷你框架），第四阶段第 3 课（CNN）
**时间：** 约 75 分钟

## 学习目标

- 在 CIFAR-10 上构建端到端的图像分类管线：数据集、数据增强（Augmentation）、模型、训练循环、评估
- 解释每个组件（数据加载器、损失、优化器、调度器、数据增强）的角色，并预测破坏其中任何一个在损失曲线上的表现
- 从零实现 mixup、cutout 和标签平滑（Label Smoothing），并论证每种方法何时值得添加
- 阅读混淆矩阵（Confusion Matrix）和每类精确率/召回率表，以诊断超出总体准确率的数据集和模型失败

## 问题

每个交付的视觉任务在某个层面上都归结为图像分类。检测对区域进行分类。分割对像素进行分类。检索按与类质心的相似度排序。把分类做对——数据集循环、数据增强策略、损失、评估——是迁移到本阶段其他每个任务的技能。

大多数分类 bug 不在模型中。它们存在于管线中：一个损坏的归一化、一个未打乱的训练集、扭曲标签的数据增强、被训练数据污染的验证集划分、在第 30 个 epoch 后静默发散的 learning rate。一个在正确设置下能在 CIFAR-10 上达到 93% 的 CNN，在损坏的设置下通常只能得到 70-75%，而损失曲线全程看起来都合理。

本课手工连接整个管线，使每个部分都可检查。你不会使用 `torchvision.datasets` 中任何可能隐藏 bug 的东西。

## 概念

### 分类管线

```mermaid
flowchart LR
    A["数据集<br/>（图像 + 标签）"] --> B["数据增强<br/>（随机变换）"]
    B --> C["归一化<br/>（均值/标准差）"]
    C --> D["DataLoader<br/>（批次 + 打乱）"]
    D --> E["模型<br/>（CNN）"]
    E --> F["Logits<br/>（N, C）"]
    F --> G["交叉熵损失"]
    F --> H["评估时<br/>取 Argmax"]
    G --> I["反向传播"]
    I --> J["优化器步进"]
    J --> K["调度器步进"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

这个循环中的每一行都是 bug 可能藏身的地方。交叉熵接受原始 logits，而不是 softmax 输出，所以在损失之前的任何 `model(x).softmax()` 都会静默地计算错误的梯度。数据增强仅应用于输入，不应用于标签——除了 mixup，它混合两者。`optimizer.zero_grad()` 必须每步发生一次；跳过它会累积梯度，看起来像一个极不稳定的 learning rate。这些 bug 中的每一个都会压平学习曲线而不抛出错误。

### 交叉熵、logits 和 softmax

分类器为每张图像产生 `C` 个数字，称为 logits。应用 softmax 将它们转换为概率分布：

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

交叉熵衡量正确类别的负对数概率：

```
CE(z, y) = -log( softmax(z)_y )
        = -z_y + log( sum_j exp(z_j) )
```

右侧形式是数值稳定的（log-sum-exp）。PyTorch 的 `nn.CrossEntropyLoss` 在一次操作中融合了 softmax + NLL，并直接接受原始 logits。自己先应用 softmax 几乎总是一个 bug——你计算的是 log(softmax(softmax(z)))，一个无意义的量。

### 为什么数据增强有效

CNN 对平移有归纳偏置（来自权重共享），但对裁剪、翻转、颜色抖动或遮挡没有内置不变性。教它这些不变性的唯一方法是向它展示能锻炼这些不变性的像素。训练期间的每个随机变换都是在说："这两张图像有相同的标签；学习忽略差异的特征。"

```
原始裁剪：  "狗面朝左"
翻转：      "狗面朝右"       <- 相同标签，不同像素
旋转(+15)：  "狗，轻微倾斜"
颜色抖动：  "狗在较暖的光线下"
随机擦除：  "狗有缺失的块"
```

规则：数据增强必须保留标签。对数字的 Cutout 和旋转可能将"6"翻转为"9"；对于该数据集，你使用较小的旋转范围并选择尊重数字特定不变性的数据增强。

### Mixup 和 Cutmix

普通数据增强变换像素但保持标签为 one-hot。**Mixup** 和 **cutmix** 通过插值两者来打破这一点。

```
Mixup：
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix：
  将 x_j 的一个随机矩形粘贴到 x_i 中
  y = x_i 和 x_j 的面积加权混合
```

为什么有帮助：模型停止记忆尖锐的 one-hot 目标，学会在类别之间插值。训练损失上升，测试准确率上升。这是任何分类器最便宜的鲁棒性升级。

### 标签平滑

Mixup 的表亲。不是针对 `[0, 0, 1, 0, 0]` 训练，而是针对 `[eps/C, eps/C, 1-eps, eps/C, eps/C]` 训练，其中 `eps` 是一个小值如 0.1。阻止模型产生任意尖锐的 logits，并以几乎零成本改善校准（Calibration）。自 PyTorch 1.10 起内置于 `nn.CrossEntropyLoss(label_smoothing=0.1)`。

### 超越准确率的评估

总体准确率隐藏了不平衡。一个总是预测多数类的 90-10 二分类器得分为 90%。真正告诉你发生了什么事的工具：

- **每类准确率** — 每个类别一个数字；立即暴露表现不佳的类别。
- **混淆矩阵** — C x C 网格，第 i 行第 j 列 = 真实类别 i 被预测为类别 j 的计数；对角线是正确的，非对角线是你的模型所在的地方。
- **Top-1 / Top-5** — 正确类别是否在前 1 或前 5 个预测中；Top-5 对 ImageNet 很重要，因为像"Norwich terrier"和"Norfolk terrier"这样的类别确实模糊。
- **校准（ECE）** — 0.8 置信度的预测在 80% 的情况下正确吗？现代网络系统性地过度自信；用温度缩放（Temperature Scaling）或标签平滑修复。

## 构建它

### 步骤 1：一个确定性的合成数据集

CIFAR-10 存在于磁盘上。为了使本课可复现且快速，我们构建一个看起来像 CIFAR 的合成数据集——32x32 RGB 图像，具有模型必须学习的类别特定结构。完全相同的管线在真实 CIFAR-10 上无需更改即可工作。

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

每个类别获得自己的调色板和频率模式，加上高斯噪声以迫使模型学习信号而不是记忆像素。十个类别，每个一千张图像，已打乱。

### 步骤 2：归一化和数据增强

每个视觉管线都有的两个变换。

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

裁剪前用反射填充，而不是零填充，因为黑色边框是模型会以非有用方式学会忽略的信号。

### 步骤 3：Mixup

在训练步骤内混合两张图像和两个标签。实现为批次变换，使其位于前向传播旁边而不是数据集内部。

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy` 是针对软标签分布的交叉熵。当目标恰好是 one-hot 时，它退化为通常的 one-hot 情况。

### 步骤 4：训练循环

完整配方：一次数据遍历，每批次一次梯度，每 epoch 一次调度器步进。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # 当 mixup 开启时，针对未混合标签 `y` 的训练准确率只是一个近似值
        # （模型看到的是软目标，而不是 y）。将其视为粗略的进度信号；
        # 依赖验证准确率来获取真实性能。
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

每次编写训练循环时检查的五个不变量：

1. 训练前 `model.train()`，评估前 `model.eval()`——翻转 dropout 和 batchnorm 行为。
2. `.zero_grad()` 在 `.backward()` 之前。
3. 累积指标时使用 `.item()`，以免任何东西保持计算图存活。
4. 评估期间使用 `@torch.no_grad()`——节省内存和时间，防止微妙的意外。
5. 对原始 logits 取 argmax，而不是 softmax——相同结果，少一次操作。

### 步骤 5：组合起来

使用上一课的 `TinyResNet`，训练几个 epoch，评估。

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet 来自上一课（03-cnns-lenet-to-resnet）。
# 将导入路径调整到你存储上一课代码的位置。
from cnns_lenet_to_resnet import TinyResNet  # 示例占位符

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

在合成数据集上，这在五个 epoch 内达到接近完美的验证准确率，这正是重点：管线是正确的，模型可以学习可学习的内容。将数据集换成真实 CIFAR-10，相同的循环无需更改即可训练到约 90%。

### 步骤 6：阅读混淆矩阵

仅靠准确率永远不能告诉你模型在哪里失败。混淆矩阵可以。

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

行是真实类别，列是预测。类别 3 和 5 之间的非对角线计数簇意味着模型混淆了这两个类别，并为你提供了针对性数据收集或类别特定数据增强的起点。

## 使用它

`torchvision` 将以上所有内容包装成惯用组件。对于真实 CIFAR-10，完整管线是四行加一个训练循环。

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

需要注意两点：均值/标准差是**数据集特定的**——在 CIFAR-10 训练集上计算，而不是 ImageNet——反射填充是社区默认的裁剪策略。在这里复制粘贴 ImageNet 统计量是约 1% 的准确率泄漏，直到有人分析模型时才会被发现。

## 交付它

本课产出：

- `outputs/prompt-classifier-pipeline-auditor.md` — 一个提示词，审计训练脚本的上述五个不变量，并指出第一个违规。
- `outputs/skill-classification-diagnostics.md` — 一个技能，给定混淆矩阵和类别名称列表，总结每类失败并提出最有影响力的单一修复方案。

## 练习

1. **（简单）** 在合成数据集上用和不使用 mixup 训练相同模型五个 epoch。绘制两者的训练和验证损失。解释为什么使用 mixup 的训练损失更高，但验证准确率相似或更好。
2. **（中等）** 实现 Cutout——在每张训练图像中清零一个随机的 8x8 方块——并运行消融实验：无数据增强、hflip+crop、hflip+crop+cutout、hflip+crop+mixup。报告每种情况的验证准确率。
3. **（困难）** 构建 CIFAR-100 管线（100 个类别，相同输入尺寸）并复现 ResNet-34 训练运行，使其在已发布准确率的 1% 以内。额外：扫描三个 learning rate 和两个 weight decay，记录到本地 CSV，生成最终的混淆矩阵-顶部混淆表。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Logits | "原始输出" | 每张图像 C 个数字的 softmax 前向量；交叉熵期望这些，而不是 softmax 后的值 |
| 交叉熵（Cross-Entropy） | "损失" | 正确类别的负对数概率；在一次稳定操作中组合 log-softmax 和 NLL |
| DataLoader | "批处理器" | 用打乱、批处理和（可选的）多工作线程加载包装数据集；被归咎于一半的训练 bug |
| 数据增强（Augmentation） | "随机变换" | 训练时保留标签的任何像素级变换；教会 CNN 本身不具有的不变性 |
| Mixup / Cutmix | "混合两张图像" | 混合输入和标签，使分类器学习平滑插值而不是硬边界 |
| 标签平滑（Label Smoothing） | "更软的目标" | 将 one-hot 替换为 (1-eps, eps/(C-1), ...)；改善校准并略微提升准确率 |
| Top-k 准确率 | "Top-5" | 正确类别在 k 个最高概率预测中；用于具有真正模糊类别的数据集 |
| 混淆矩阵（Confusion Matrix） | "错误在哪里" | C x C 表，其中条目 (i, j) 计数真实类别 i 被预测为 j 的图像；对角线是正确的，非对角线告诉你该修复什么 |

## 扩展阅读

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/) — 仍然是单页中最清晰的训练管线导览
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187) — 每个小技巧，合起来为 ResNet 在 ImageNet 上增加 3-4% 准确率
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412) — 原始 mixup 论文；三页理论加令人信服的实验
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599) — 证明现代网络校准不良并用一个标量参数修复的论文
