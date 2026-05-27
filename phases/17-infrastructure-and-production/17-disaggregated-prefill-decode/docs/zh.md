# 分离式预填充/解码 — NVIDIA Dynamo 与 llm-d

> 预填充（Prefill）是计算密集型的；解码（Decode）是内存密集型的。在同一 GPU 上同时运行两者会浪费其中一种资源。分离式架构（Disaggregation）将它们拆分到独立的池中，并通过 NIXL（RDMA/InfiniBand 或 TCP 回退）在它们之间传输 KV 缓存（Key-Value 缓存）。**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上 — 其 Planner Profiler + SLA Planner 自动匹配预填充:解码比率以满足 SLO（服务等级目标，Service Level Objective）。NVIDIA 公布的吞吐量增益大致在这个范围内 — developer.nvidia.com（2025-06）显示在 GB200 NVL72 + Dynamo 上，DeepSeek-R1 MoE 在中等延迟场景下提升约 6 倍，Dynamo 产品页面（developer.nvidia.com，未注明日期）宣称在 GB300 NVL72 + Dynamo 上 MoE 吞吐量相比 Hopper 提升高达 50 倍。"30 倍"这个数字是全栈 Blackwell + Dynamo + DeepSeek-R1 报告的社区汇总数据；我们未找到明确声称恰好 30 倍的单一主要来源，因此将其视为方向性声明。**llm-d**（Red Hat + AWS）是 Kubernetes 原生的：预填充/解码/路由器作为独立的 Service，具有按角色的 HPA（水平 Pod 自动扩缩容，Horizontal Pod Autoscaler）。llm-d 0.5 增加了分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容到零。经济学：多个客户披露的内部汇总表明，在恒定 SLA 下从合并服务切换到使用 Dynamo 的分离式服务时，$2M 级别的推理支出可节省 30-40%（即每年 $600-800K）；具体的 $2M→$600-800K 数字是内部综合数据，而非单一已发布的案例研究 — 请将其用作数量级锚点，而非引用来源。短提示词（< 512 token，短输出）不值得付出传输成本。

**类型：** 学习
**语言：** Python（标准库，分离式 vs 合并式模拟器）
**前置要求：** 第 17 阶段 · 04（vLLM 服务内部原理）、第 17 阶段 · 08（推理指标）
**时间：** 约 75 分钟

## 学习目标

- 解释为什么预填充和解码具有不同的最优 GPU 分配，并量化合并式服务下的浪费。
- 绘制分离式架构图：预填充池、解码池、通过 NIXL 传输 KV、路由器。
- 说出分离式架构不划算的条件（短提示词、短输出）。
- 区分 NVIDIA Dynamo（上层堆栈）和 llm-d（Kubernetes 原生），并将每个匹配到运维场景。

## 问题

你在 8 个 H100 上运行 Llama 3.3 70B。在混合工作负载下（长提示词 + 短输出），GPU 在解码期间空闲，因为大部分计算花费在预填充上。在不同的工作负载下（短提示词 + 长输出），情况正好相反。合并式预填充 + 解码意味着你过度配置了两者。

预算影响：20-40% 的 GPU 时间浪费在错误的资源上。你在购买 H100 计算来运行内存密集型的解码，或者购买 H100 HBM 带宽来运行计算密集型的预填充。两者都是昂贵的浪费。

分离式架构将预填充和解码拆分到独立的池中，每个池针对各自的瓶颈进行规模调整。KV 缓存通过高带宽互连从预填充池传输到解码池。

## 概念

### 为什么瓶颈不同

**预填充（Prefill）** — 在一次前向传播中对完整输入提示词运行 transformer。矩阵乘法占主导；计算密集型。H100 FP8 提供约 2000 TFLOPS 的有效吞吐量。批处理效率良好 — 一次前向传播处理许多 token。

**解码（Decode）** — 每次迭代生成一个 token，每次迭代读取完整权重。内存带宽密集型。HBM3 提供约 3 TB/s。仅在高并发下批处理效率良好 — 权重读取在批次中摊销。

合并两者：你购买的是针对两者优化的 GPU。H100 在两者上都表现良好，但无论如何成本相同。在大规模下，你希望预填充池使用 H100/计算密集型；解码池使用 H200/内存密集型，或使用激进量化。

### 架构

```
            ┌──────────────┐
  请求 →    │    路由器    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (仅提示词)                     │
            ┌──────────────┐    KV 缓存     ┌───────▼──────┐
            │  预填充池    │ ─── NIXL ────► │   解码池     │
            │  (计算)      │                │  (内存)      │
            └──────────────┘                └──────┬───────┘
                                                   │ token
                                                   ▼
                                                 客户端
```

NIXL 是 NVIDIA 的节点间传输层。在可用时使用 RDMA/InfiniBand，否则使用 TCP 回退。传输延迟是真实存在的 — 对于 70B FP8 上 4K token 提示词的 KV 缓存，通常为 20-80 ms。这就是为什么短提示词不值得分离式架构：传输税超过了节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 作为编排器位于 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量工作负载，SLA Planner 自动配置预填充:解码比率。
- Rust 核心，Python 可扩展性。
- 吞吐量增益：NVIDIA 报告在 GB200 NVL72 + Dynamo 上，DeepSeek-R1 MoE 在中等延迟场景下提升 6 倍（developer.nvidia.com，2025-06）；社区关于全栈 Blackwell + Dynamo + DeepSeek-R1 上"高达 30 倍"的报告缺乏单一主要来源，应视为方向性数据。
- GB300 NVL72 + Dynamo：根据 Dynamo 产品页面（developer.nvidia.com，未注明日期），MoE 吞吐量相比 Hopper 提升高达 50 倍。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- 预填充/解码/路由器作为独立的 Kubernetes Service。
- 按角色的 HPA，使用队列深度（预填充）/ KV 利用率（解码）信号。
- `topologyConstraint packDomain: rack` 将预填充+解码集群打包在同一机架上，以实现高带宽 KV 传输。
- llm-d 0.5（2026 年）：分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容到零。

如果你想要托管的上层编排器，使用 Dynamo。如果你想要 Kubernetes 原生原语并致力于 CNCF 生态系统，使用 llm-d。

### 经济学

内部综合数据（非单一已发布案例研究 — 数量级锚点）：

- 合并式服务每年 $2M 推理支出。
- 切换到使用 Dynamo 的分离式服务。
- 相同请求量，相同 P99 延迟 SLA。
- 报告的节省：每年 $600K–$800K（减少 30–40%）。
- 无新硬件。

我们综合了多个客户披露的数据，而非单一可引用的案例研究；最接近的已发布数据点是 Baseten 使用 Dynamo KV 路由实现 2 倍更快的 TTFT（首 token 时间，Time To First Token）/ 61% 更高的吞吐量（baseten.co，2025-10），以及 VAST + CoreWeave 预测在 40–60% KV 命中率下每美元 token 数提升 60–130%（vastdata.com，2025-12）。节省来自对每个池进行合理规模调整；预填充密集型工作负载（8K+ 前缀的 RAG（检索增强生成，Retrieval-Augmented Generation））比均衡型工作负载受益更多。

### 什么时候不应该分离

- 提示词 < 512 token 且输出 < 200 token：传输税超过收益。
- 小集群（< 4 个 GPU）：池多样性不足。
- 团队无法运维两个具有按角色扩缩容的 GPU 池：Dynamo 有帮助，但并非易事。
- 无 RDMA 网络：TCP 传输税更重。

### 路由器与第 17 阶段 · 11 集成

分离式路由器是 KV 缓存感知的（第 17 阶段 · 11）。请求落在持有其前缀的解码池上 — 如果不匹配，则流经预填充 → 解码。命中率和分离式架构是复合的 — 缓存感知路由器决定是否需要新的预填充。

### Blackwell 上的 MoE 才是真正的数字所在

GB300 NVL72 + Dynamo 显示 MoE 吞吐量相比 Hopper 基线提升 50 倍。MoE 专家路由在预填充上是计算密集型的，但在解码上是内存密集型的（专家缓存），因此分离式架构是双重胜利。2026 年前沿模型服务以 MoE 为主（DeepSeek-V3、未来的 GPT-5 变体）。

### 你应该记住的数字

基准数字会漂移 — NVIDIA 和推理堆栈每季度发布更新结果。在引用之前重新检查。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo 上：中等延迟场景下吞吐量相比基线提升约 6 倍（developer.nvidia.com，2025-06）；社区关于全栈 Blackwell + Dynamo 上"高达 30 倍"的说法是没有单一主要来源的方向性汇总数据。
- GB300 NVL72 + Dynamo：MoE 吞吐量相比 Hopper 提升高达 50 倍（developer.nvidia.com，未注明日期）。
- 节省锚点（内部综合数据，非单一案例研究）：在恒定 SLA 下，每年 $2M 支出节省 $600-800K。
- 分离阈值：提示词 > 512 token + 输出 > 200 token。
- 通过 NIXL 传输 KV：70B FP8 上 4K 提示词 KV 约 20-80 ms。

## 动手实践

`code/main.py` 模拟合并式 vs 分离式服务。报告吞吐量、每请求成本和提示词长度交叉点。

## 交付物

本课产出 `outputs/skill-disaggregation-decider.md`。给定工作负载和集群，决定是否分离。

## 练习

1. 运行 `code/main.py`。在什么提示词长度下分离式架构优于合并式？
2. 为 P99 前缀长度 8K、输出 300 的 RAG 服务设计预填充池和解码池。
3. Dynamo vs llm-d：为一个纯 Kubernetes 且无 Python 运行时偏好的团队选择。
4. 计算 KV 传输成本：70B FP8 上 4K 预填充 = 约 500 MB KV。在 RDMA 100 GB/s 下，传输 = 5 ms。在 TCP 10 GB/s 下 = 50 ms。哪个对你的 SLA 重要？
5. MoE 专家路由改变了 KV 访问模式。分离式架构在每 token 激活不同专家的 MoE 下表现如何？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 分离式服务（Disaggregated serving） | "拆分预填充/解码" | 每个阶段使用独立的 GPU 池 |
| NIXL | "NVIDIA 传输" | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "编排器" | vLLM/SGLang/TRT-LLM 的上层协调器 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS K8s 分离式堆栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量工作负载，配置池比率 |
| SLA Planner | "Dynamo 策略" | 自动匹配预填充:解码比率以满足 SLO |
| `packDomain: rack` | "llm-d 拓扑" | 将预填充+解码打包在同一机架上以实现快速 KV 传输 |
| UCCL | "统一集合通信" | llm-d 0.5 网络层，用于缩容到零 |
| MoE 专家路由 | "每 token 专家" | DeepSeek-V3 模式；分离式架构有帮助 |
