# 实时视觉（Real-Time Vision）— 边缘部署（Edge Deployment）

> 边缘推理（Edge Inference）是一门让 90 准确率的模型在只有 2 GB 内存的设备上以 30 fps 运行的学科。每一个准确率百分点都与毫秒级的延迟进行权衡。

**类型：** 学习 + 构建（Learn + Build）
**语言：** Python
**前置知识：** 第 4 阶段第 04 课（图像分类）、第 10 阶段第 11 课（量化）
**时间：** 约 75 分钟

## 学习目标

- 测量任何 PyTorch 模型的推理延迟（Inference Latency）、峰值内存（Peak Memory）和吞吐量（Throughput），并理解 FLOPs / 参数 / 延迟的权衡
- 使用 PyTorch 的训练后量化（Post-Training Quantisation）将视觉模型量化为 INT8，并验证准确率损失 < 1%
- 导出到 ONNX 并使用 ONNX Runtime 或 TensorRT 编译；列举三种最常见的导出失败及其修复方法
- 解释何时为边缘约束选择 MobileNetV3、EfficientNet-Lite、ConvNeXt-Tiny 或 MobileViT

## 问题

训练时的视觉模型是一个浮点怪兽。1 亿参数，每次前向传播 10 GFLOPs，2 GB 显存。这些都不适合手机、汽车信息娱乐单元、工业相机或无人机。交付一个视觉系统意味着将相同的预测塞进一个 100 倍小的预算中。

三个旋钮完成大部分工作：模型选择（相同方案下的更小架构）、量化（INT8 而非 FP32）和推理运行时（ONNX Runtime、TensorRT、Core ML、TFLite）。正确设置它们是一个在工作站上运行的演示与一个在 30 美元相机模块上出货的产品之间的区别。

本课首先建立测量规范（你无法优化你无法测量的东西），然后遍历三个旋钮。目标不是学习每个边缘运行时，而是知道存在哪些杠杆以及如何验证每个杠杆是否按预期工作。

## 概念

### 三个预算

```mermaid
flowchart LR
    M["模型"] --> LAT["延迟<br/>每张图像毫秒"]
    M --> MEM["内存<br/>峰值 MB"]
    M --> PWR["功耗<br/>每次推理毫焦"]

    LAT --> SHIP["出货 / 不出货<br/>决策"]
    MEM --> SHIP
    PWR --> SHIP

    style LAT fill:#fecaca,stroke:#dc2626
    style MEM fill:#fef3c7,stroke:#d97706
    style PWR fill:#dbeafe,stroke:#2563eb
```

- **延迟（Latency）**：p50、p95、p99。仅平均 p50 会隐藏对实时系统很重要的尾部行为。
- **峰值内存（Peak Memory）**：设备曾经看到的最大值，而非稳态平均值。很重要，因为在嵌入式目标上 OOM 是致命的。
- **功耗 / 能量（Power / Energy）**：电池供电设备上每次推理的毫焦耳。通常用 CPU/GPU 利用率 * 时间来近似。

一个（模型、延迟、内存、准确率）的表格是边缘决策的依据。每个单元格都在目标设备上测量，而非工作站。

### 测量规范

每个边缘性能分析应遵循的三条规则：

1. **预热**模型，在测量前进行 5-10 次虚拟前向传播。冷缓存和 JIT 编译会产生不具代表性的首次数字。
2. **同步** GPU 工作负载，在计时块前后使用 `torch.cuda.synchronize()`。没有这个，你测量的是内核调度而非内核执行。
3. **固定输入尺寸**为生产分辨率。224x224 上的延迟不是 512x512 上的延迟。

### FLOPs 作为代理指标

FLOPs（每次推理的浮点运算次数）是延迟的一个廉价、设备无关的代理指标。对架构比较有用，作为绝对墙钟时间则具有误导性。一个 FLOPs 多 10% 的模型在实践中可能快 2 倍，因为它使用了硬件友好的操作（深度可分离卷积编译良好，大 7x7 卷积则不然）。

规则：使用 FLOPs 进行架构搜索，使用设备上延迟进行部署决策。

### 量化简述

用 INT8 替换 FP32 权重和激活值。模型大小下降 4 倍，内存带宽下降 4 倍，在具有 INT8 内核的硬件上计算下降 2-4 倍（每个现代移动 SoC、每个带 Tensor Core 的 NVIDIA GPU）。视觉任务上的准确率损失通常为 0.1-1 个百分点，使用训练后静态量化。

类型：

- **动态（Dynamic）** — 权重量化为 INT8，激活值以 FP 计算。简单，小幅加速。
- **静态（训练后，Post-Training）** — 量化权重 + 在小型校准集上校准激活值范围。比动态快得多。
- **量化感知训练（QAT，Quantisation-Aware Training）** — 在训练期间模拟量化，使模型围绕它学习。最佳准确率，需要标注数据。

对于视觉，训练后静态量化以 5% 的努力获得 95% 的收益。仅在 PTQ 的准确率损失不可接受时才使用 QAT。

### 剪枝和蒸馏

- **剪枝（Pruning）** — 移除不重要的权重（基于幅值）或通道（结构化）。在过参数化模型上效果良好；对已经紧凑的架构不太有用。
- **蒸馏（Distillation）** — 训练一个小型学生模型模仿大型教师模型的 logits。通常能恢复因缩小模型而损失的大部分准确率。生产边缘模型的标准做法。

### 推理运行时

- **PyTorch eager** — 慢，不适合部署。仅用于开发。
- **TorchScript** — 遗留。已被 `torch.compile` 和 ONNX 导出取代。
- **ONNX Runtime** — 中性运行时。CPU、CUDA、CoreML、TensorRT、OpenVINO 都有 ONNX 提供程序。从这里开始。
- **TensorRT** — NVIDIA 编译器。在 NVIDIA GPU（工作站和 Jetson）上延迟最佳。与 ONNX Runtime 集成或独立使用。
- **Core ML** — Apple 的 iOS/macOS 运行时。需要 `.mlmodel` 或 `.mlpackage`。
- **TFLite** — Google 的 Android/ARM 运行时。需要 `.tflite`。
- **OpenVINO** — Intel 的 CPU/VPU 运行时。需要 `.xml` + `.bin`。

实践中：导出 PyTorch -> ONNX -> 为目标选择运行时。ONNX 是通用语言。

### 边缘架构选择器

| 预算 | 模型 | 原因 |
|--------|-------|-----|
| < 3M 参数 | MobileNetV3-Small | 到处都能编译，良好的基线 |
| 3-10M | EfficientNet-Lite-B0 | TFLite 上每参数最佳准确率 |
| 10-20M | ConvNeXt-Tiny | 每参数最佳准确率，CPU 友好 |
| 20-30M | MobileViT-S 或 EfficientViT | 具有 ImageNet 准确率的 Transformer |
| 30-80M | Swin-V2-Tiny | 如果技术栈支持窗口注意力 |

除非有特定理由不这样做，否则将所有这些量化为 INT8。

## 构建它

### 步骤 1：正确测量延迟

```python
import time
import torch

def measure_latency(model, input_shape, device="cpu", warmup=10, iters=50):
    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "p99_ms": times[int(len(times) * 0.99)],
        "mean_ms": sum(times) / len(times),
    }
```

预热，同步，使用 `time.perf_counter()`。报告百分位数，而不仅仅是均值。

### 步骤 2：参数和 FLOP 计数

```python
def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def flops_estimate(model, input_shape):
    """
    仅卷积/线性模型的粗略 FLOP 计数。生产环境使用 `fvcore` 或 `ptflops`。
    """
    total = 0
    def conv_hook(m, inp, out):
        nonlocal total
        c_out, c_in, kh, kw = m.weight.shape
        h, w = out.shape[-2:]
        total += 2 * c_in * c_out * kh * kw * h * w
    def linear_hook(m, inp, out):
        nonlocal total
        total += 2 * m.in_features * m.out_features
    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(input_shape))
    for h in hooks:
        h.remove()
    return total
```

对于真实项目，使用 `fvcore.nn.FlopCountAnalysis` 或 `ptflops`；它们正确处理每种模块类型。

### 步骤 3：训练后静态量化

```python
def quantise_ptq(model, calibration_loader, backend="x86"):
    import torch.ao.quantization as tq
    model = model.eval().cpu()
    model.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(model, inplace=True)
    with torch.no_grad():
        for x, _ in calibration_loader:
            model(x)
    tq.convert(model, inplace=True)
    return model
```

三步：配置、准备（插入观察器）、用真实数据校准、转换（融合 + 量化）。需要模型被融合（`Conv -> BN -> ReLU` -> `ConvBnReLU`），由 `torch.ao.quantization.fuse_modules` 处理。

### 步骤 4：导出到 ONNX

```python
def export_onnx(model, sample_input, path="model.onnx"):
    model = model.eval()
    torch.onnx.export(
        model,
        sample_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return path
```

`opset_version=17` 是 2026 年的安全默认值。`dynamic_axes` 允许以任意批次大小运行 ONNX 模型。

### 步骤 5：基准测试和比较方案

```python
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def compare_regimes():
    model = mobilenet_v3_small(weights=None, num_classes=10)
    params = parameter_count(model)
    flops = flops_estimate(model, (1, 3, 224, 224))
    lat_fp32 = measure_latency(model, (1, 3, 224, 224), device="cpu")
    print(f"FP32 MobileNetV3-Small: {params:,} params  {flops/1e9:.2f} GFLOPs  "
          f"p50={lat_fp32['p50_ms']:.2f}ms  p95={lat_fp32['p95_ms']:.2f}ms")
```

对 `resnet50`、`efficientnet_v2_s` 和 `convnext_tiny` 运行相同函数，你就有了部署决策所需的比较表。

## 使用它

生产技术栈收敛到三条路径之一：

- **Web / 无服务器**：PyTorch -> ONNX -> ONNX Runtime（CPU 或 CUDA 提供程序）。最简单，对大多数情况足够好。
- **NVIDIA 边缘（Jetson、GPU 服务器）**：PyTorch -> ONNX -> TensorRT。最佳延迟，最大的工程工作量。
- **移动端**：PyTorch -> ONNX -> Core ML（iOS）或 TFLite（Android）。导出前量化。

对于测量，`torch-tb-profiler`、`nvprof` / `nsys` 和 macOS 上的 Instruments 提供逐层分解。`benchmark_app`（OpenVINO）和 `trtexec`（TensorRT）提供独立的 CLI 数字。

## 交付它

本课产出：

- `outputs/prompt-edge-deployment-planner.md` — 一个提示词，给定目标设备和延迟 SLA，选择骨干网络、量化策略和运行时。
- `outputs/skill-latency-profiler.md` — 一个技能，编写完整的延迟基准测试脚本，包含预热、同步、百分位数和内存追踪。

## 练习

1. **（简单）** 在 CPU 上以 224x224 测量 `resnet18`、`mobilenet_v3_small`、`efficientnet_v2_s` 和 `convnext_tiny` 的 p50 延迟。报告表格并确定哪个架构具有最佳的每毫秒准确率。
2. **（中等）** 对 `mobilenet_v3_small` 应用训练后静态量化。在 CIFAR-10 或类似数据集的保留子集上报告 FP32 vs INT8 延迟和准确率损失。
3. **（困难）** 将 `convnext_tiny` 导出到 ONNX，通过 `onnxruntime` 使用 `CPUExecutionProvider` 运行，并将延迟与 PyTorch eager 基线进行比较。确定 ONNX Runtime 更快的第一个层并解释原因。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 延迟（Latency） | "有多快" | 从输入到输出的时间；p50/p95/p99 百分位数，而非均值 |
| FLOPs | "模型大小" | 每次前向传播的浮点运算次数；计算成本的粗略代理 |
| INT8 量化（INT8 Quantisation） | "8 位" | 用 8 位整数替换 FP32 权重/激活值；约 4 倍更小，2-4 倍更快 |
| PTQ | "训练后量化" | 无需重新训练即可量化已训练模型；简单，通常足够 |
| QAT | "量化感知训练" | 训练期间模拟量化；最佳准确率，需要标注数据 |
| ONNX | "中性格式" | 每个主流推理运行时都支持的模型交换格式 |
| TensorRT | "NVIDIA 编译器" | 将 ONNX 编译为 NVIDIA GPU 的优化引擎 |
| 蒸馏（Distillation） | "教师 -> 学生" | 训练小型模型模仿大型模型的 logits；恢复大部分损失的准确率 |

## 扩展阅读

- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946) — 高效架构的复合缩放
- [MobileNetV3 (Howard et al., 2019)](https://arxiv.org/abs/1905.02244) — 移动优先架构，带 h-swish 和 squeeze-excite
- [A Practical Guide to TensorRT Optimization (NVIDIA)](https://developer.nvidia.com/blog/accelerating-model-inference-with-tensorrt-tips-and-best-practices-for-pytorch-users/) — 如何实际获得论文中的吞吐量数字
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/) — 量化、图优化、提供程序选择
