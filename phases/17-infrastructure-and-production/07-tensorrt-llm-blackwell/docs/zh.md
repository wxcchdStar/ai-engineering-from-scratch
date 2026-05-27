# TensorRT-LLM on Blackwell — FP4 量化、多 GPU 并行与飞行中批处理

> NVIDIA 的 **Blackwell 架构**（B200、GB200）引入了原生 **FP4 精度**，使模型内存占用相比 FP8 减半，相比 FP16 减少 75%。**TensorRT-LLM（TRT-LLM）** 是 NVIDIA 的推理编译器/运行时，针对 Blackwell 进行了优化 — 它将模型图编译为针对特定 GPU 和批处理大小调优的优化 CUDA 内核。关键特性：**飞行中批处理（In-flight batching）** 是 TRT-LLM 的连续批处理版本 — 请求在每次解码迭代时加入/退出批次。**张量并行（Tensor parallelism）** 将权重矩阵分割到多个 GPU 上（每层内）；**流水线并行（Pipeline parallelism）** 将层分割到多个 GPU 上（层间）。Blackwell 上的 FP4 量化使用 NVIDIA 的 **Microscaling（MX）格式** — 每个块有一个共享指数，提供比 INT4 更好的精度，同时保持相同的内存占用。TRT-LLM 还管理 KV 缓存（Key-Value 缓存），具有类似于 vLLM 的块级分配，但使用 NVIDIA 专有的内存管理。Blackwell 上的基准测试显示，相对于 H100 FP8，Llama 3.3 70B 的吞吐量提升 2-3 倍。TRT-LLM 是 NVIDIA 锁定的 — 仅在 NVIDIA GPU 上运行。

**类型：** 学习
**语言：** Python（标准库，TRT-LLM 配置构建器模拟）
**前置要求：** 第 17 阶段 · 04（vLLM 服务内部原理）、第 17 阶段 · 09（生产量化）
**时间：** 约 75 分钟

## 学习目标

- 解释 FP4 精度如何将模型内存占用相比 FP8 减半，以及 Microscaling 格式如何保持精度。
- 区分张量并行（层内）和流水线并行（层间），并计算每种的内存开销。
- 描述飞行中批处理以及它与 vLLM 连续批处理的不同之处。
- 说出 TRT-LLM 的 NVIDIA 锁定约束以及它何时值得接受。

## 问题

你在 8 个 H100 上运行 Llama 3.3 70B FP8。它占用约 70 GB 的 GPU 内存。你想运行更大的批次或更长的上下文，但内存不足。Blackwell B200 提供原生 FP4 — 相同的模型占用 35 GB。你可以在相同的 GPU 内存中容纳 2 倍的批次大小或 2 倍的上下文长度。

但 FP4 精度损失是真实存在的。NVIDIA 的 Microscaling 格式通过为每个块使用共享指数来缓解这一问题 — 类似于块浮点（block floating-point），但针对 transformer 注意力模式进行了优化。结果：FP4 精度在大多数基准测试中与 FP8 相差不到 1%。

## 概念

### Blackwell 架构与 FP4

Blackwell（B200、GB200）引入了第二代 Transformer Engine，原生支持 FP4。关键规格：
- B200：192 GB HBM3e，8 TB/s 带宽。
- GB200（Grace Blackwell）：连接 2 个 B200 + 1 个 Grace CPU。
- FP4 张量核心：2 倍于 FP8 的吞吐量。

FP4 格式：4 位浮点数（1 位符号 + 2 位指数 + 1 位尾数）。范围：±0.0625 到 ±240.0。粒度太粗，无法直接使用。

**Microscaling（MX）格式**：将值分组为块（通常为 32 个元素）。每个块共享一个指数（8 位）。块内的值使用 FP4 尾数。这提供了细粒度的动态范围，同时保持了 FP4 的内存占用。

### TensorRT-LLM 编译流程

1. **图捕获（Graph capture）**：将模型导出为 ONNX 或直接使用 Python API。
2. **图优化（Graph optimization）**：层融合、常量折叠、内核自动调优。
3. **内核选择（Kernel selection）**：TRT-LLM 为每个操作选择最优的 CUDA 内核（针对特定的 GPU 架构和批处理大小）。
4. **引擎序列化（Engine serialization）**：将编译后的引擎保存到磁盘。加载时间 < 1 秒（对比模型加载的 30 秒-5 分钟）。

权衡：编译时间较长（大型模型需要数分钟到数小时），但推理延迟较低。适合生产部署；不适合快速实验。

### 张量并行 vs 流水线并行

**张量并行（Tensor parallelism，TP）**：将每个权重矩阵分割到多个 GPU 上。每个 GPU 持有矩阵的一个分片。前向传播需要 all-reduce 操作来同步激活。通信开销与 TP 度成正比。适合：单节点内的高带宽 NVLink 连接。

**流水线并行（Pipeline parallelism，PP）**：将层分割到多个 GPU 上。每个 GPU 持有一组连续的层。前向传播将激活传递给下一个 GPU。通信开销较低（仅层边界）。适合：跨节点，其中节点间带宽较低。

典型配置：TP=4、PP=2 用于 8 个 GPU 上的 70B 模型。TP 处理层内的大矩阵乘法；PP 处理层间通信。

### 飞行中批处理

TRT-LLM 的连续批处理版本。与 vLLM 的关键区别：
- TRT-LLM 使用编译后的执行图 — 批处理大小在编译时上限固定。
- vLLM 使用动态图 — 批处理大小完全灵活。
- TRT-LLM 的飞行中批处理在编译后的图内运行，因此开销较低，但灵活性较差。

对于稳定的生产工作负载（已知的批处理大小范围），TRT-LLM 的开销较低。对于高度可变的工作负载，vLLM 的灵活性胜出。

### KV 缓存管理

TRT-LLM 使用类似于 vLLM 的块级分配，但采用 NVIDIA 专有的内存管理。块在 GPU 内存中分配；驱逐到 CPU 内存是可选的。TRT-LLM 还支持与 vLLM 的 PagedAttention 类似的跨请求前缀共享。

### 你应该记住的数字

- FP4 内存占用：FP8 的一半，FP16 的 1/4。
- B200 HBM：192 GB HBM3e，8 TB/s。
- Microscaling 块大小：32 个元素。
- TP 通信开销：与 TP 度成正比。
- TRT-LLM 编译时间：大型模型需要数分钟到数小时。

## 动手实践

`code/main.py` 模拟 TRT-LLM 配置：给定模型大小和 GPU 数量，推荐 TP/PP 配置并估算内存占用。

## 交付物

本课产出 `outputs/skill-trt-llm-config.md`。给定硬件和模型，推荐 TRT-LLM 配置。

## 练习

1. 运行 `code/main.py`。对于 8 个 B200 上的 70B 模型，最优 TP/PP 配置是什么？
2. 比较 FP4 Microscaling 与 INT4 GPTQ（第 17 阶段 · 09）。在什么场景下各自胜出？
3. 飞行中批处理 vs 连续批处理：在什么工作负载下 TRT-LLM 的编译方法胜出？
4. 计算 8 个 GPU 上 TP=4、PP=2 的通信开销。瓶颈在哪里？
5. 论证 TRT-LLM 的 NVIDIA 锁定是否值得用它换取 2-3 倍的吞吐量提升。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Blackwell | "NVIDIA 新 GPU" | B200/GB200 架构，原生 FP4 |
| FP4 | "4 位浮点" | 4 位浮点格式，FP8 内存的一半 |
| Microscaling（MX） | "块浮点" | 每个块共享指数，保持精度 |
| TensorRT-LLM | "NVIDIA 推理" | NVIDIA 的推理编译器/运行时 |
| 飞行中批处理（In-flight batching） | "TRT 连续批处理" | TRT-LLM 的动态批处理 |
| 张量并行（Tensor parallelism） | "层内分割" | 将权重矩阵分割到多个 GPU 上 |
| 流水线并行（Pipeline parallelism） | "层间分割" | 将层分割到多个 GPU 上 |
| 图编译（Graph compilation） | "AOT 编译" | 提前编译，加载快，灵活性差 |
| NVLink | "GPU 互连" | 高带宽 GPU 到 GPU 通信 |
