# vLLM 生产栈与 LMCache KV 卸载

> vLLM 的 production-stack 是参考 Kubernetes 部署 — 路由器、引擎和可观测性连接在一起。**LMCache** 是 KV 卸载层，将 KV 缓存（Key-Value 缓存）从 GPU 内存中提取出来，并在查询和引擎之间复用（CPU DRAM，然后是磁盘/Ceph）。vLLM 0.11.0 KV 卸载连接器（2026 年 1 月）通过连接器 API（Connector API，v0.9.0+）使其异步且可插拔。卸载延迟不面向用户。即使没有共享前缀，LMCache 也有价值 — 当 GPU 的 KV 槽位不足时，被抢占的请求可以从 CPU 恢复，而不是重新计算预填充。已发布的基准测试在 16 个 H100（80GB HBM）上跨 4 个 a3-highgpu-4g 进行：当 KV 缓存超过 HBM 时，原生 CPU 卸载和 LMCache 都显著提高了吞吐量；在低 KV 占用下，所有配置与基线匹配，开销很小。

**类型：** 学习
**语言：** Python（标准库，KV 溢出模拟器）
**前置要求：** 第 17 阶段 · 04（vLLM 服务内部原理）、第 17 阶段 · 06（SGLang/RadixAttention）
**时间：** 约 60 分钟

## 学习目标

- 绘制 vLLM production-stack 层次结构：路由器、引擎、KV 卸载、可观测性。
- 解释 KV 卸载连接器 API（v0.9.0+）以及 0.11.0 异步路径如何隐藏卸载延迟。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM）vs 增加开销（KV 足够小以适应 HBM）。
- 在给定部署约束下，在原生 vLLM CPU 卸载和 LMCache 连接器之间做出选择。

## 问题

你的 vLLM 服务显示 GPU 在 100% HBM（高带宽内存，High Bandwidth Memory）下运行，每当并发升高时就会出现抢占事件。请求被驱逐、重新排队，你在一分钟内对相同的 2K token 提示词重新预填充四次。GPU 计算花费在冗余预填充上；有效吞吐量（goodput）远低于原始吞吐量。

增加更多 GPU 成本线性增长。增加更多 HBM 是不可能的。但 CPU DRAM 很便宜 — 一个插槽有 512 GB+，延迟比 HBM 差几个数量级，但对于"暂时温热"的 KV 缓存来说是可以接受的。

LMCache 将 KV 缓存提取到 CPU DRAM，以便被抢占的请求快速恢复，并且跨引擎的重复前缀共享缓存，而无需每个引擎重新预填充。

## 概念

### vLLM production-stack

`github.com/vllm-project/production-stack` 是参考 Kubernetes 部署：

- **路由器（Router）** — 缓存感知（第 17 阶段 · 11）。消费 KV 事件。
- **引擎（Engines）** — vLLM worker。每个 GPU 或每个 TP/PP 组一个。
- **KV 缓存卸载（KV cache offload）** — LMCache 部署或原生连接器。
- **可观测性（Observability）** — Prometheus 抓取、Grafana 仪表盘、OTel 追踪。
- **控制平面（Control plane）** — 服务发现、配置、滚动更新。

以 Helm chart + operator 形式发布。

### KV 卸载连接器 API（v0.9.0+）

vLLM 0.9.0 引入了连接器 API（Connector API），用于可插拔的 KV 缓存后端。你的引擎将块卸载到连接器；连接器存储它们（RAM、磁盘、对象存储、LMCache）。请求需要一个块，连接器将其加载回来。

vLLM 0.11.0（2026 年 1 月）添加了异步卸载路径 — 卸载可以在后台进行，因此引擎在常见情况下不会阻塞它。端到端延迟和吞吐量仍然取决于工作负载形态、KV 缓存命中率和系统压力；vLLM 自己的说明指出，自定义内核卸载在低命中率下可能降低吞吐量，并且异步调度与推测解码存在已知的交互问题。

### 原生 CPU 卸载 vs LMCache

**原生 vLLM CPU 卸载**：引擎本地。将 KV 块存储在主机 RAM 中。实现快速，零网络跳转。不跨引擎。

**LMCache 连接器**：集群规模。将块存储在共享的 LMCache 服务器中（CPU DRAM + Ceph/S3 层）。块对任何引擎都可访问。已发布 16 个 H100 的基准测试。

当单个引擎面临 HBM 压力时选择原生。当多个引擎共享前缀时选择 LMCache（具有通用系统提示词的 RAG（检索增强生成，Retrieval-Augmented Generation）、具有共享模板的多租户）。

### 基准测试行为

16 个 H100（80 GB HBM）分布在 4 个 a3-highgpu-4g 上的测试：

- 低 KV 占用（短提示词、低并发）：所有配置与基线匹配，LMCache 增加约 3-5% 的开销。
- 中等占用：LMCache 开始在跨引擎前缀复用上发挥作用。
- KV 超过 HBM：原生 CPU 卸载和 LMCache 都显著提高吞吐量；LMCache 增益更大，因为跨引擎共享。

### LMCache 何时具有决定性

- 多租户服务，其中系统提示词在租户之间共享。
- RAG，其中文档块在查询之间重复。
- 同一基础上的微调变体（LoRA），其中基础模型 KV 复用减少了冗余工作。
- 抢占密集型工作负载：从 CPU 恢复比重新预填充更便宜。

### 何时不应启用

- HBM 压力小 — 你支付开销而没有收益。
- 短上下文（< 1K token）— 传输时间 > 重新预填充。
- 单租户单提示词工作负载 — 没有可捕获的复用。

### 与分离式服务的集成

第 17 阶段 · 17 分离式服务 + LMCache 复合：从预填充池传输到解码池的 KV 如果不使用则落在 LMCache 中；后续查询从 LMCache 拉取。第 17 阶段 · 11 缓存感知路由器可以路由到其本地或 LMCache 共享缓存匹配的引擎。

### 你应该记住的数字

- vLLM 0.9.0：连接器 API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（非绝对保证）。
- 16 个 H100 基准测试：当 KV 占用超过 HBM 时 LMCache 有帮助。
- HBM 压力小：3-5% 开销，无收益。

## 动手实践

`code/main.py` 模拟有和没有 LMCache 的抢占密集型工作负载。报告避免的重新预填充次数、吞吐量增益和盈亏平衡 HBM 利用率。

## 交付物

本课产出 `outputs/skill-vllm-stack-decider.md`。给定工作负载形态和 vLLM 部署，决定原生 vs LMCache vs 两者都不用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率下 LMCache 开始产生回报？
2. 一个租户在 200 次查询/小时中共享 6K token 的系统提示词。计算每个租户的预期 LMCache 节省。
3. LMCache 服务器是单点故障。设计 HA（高可用，High Availability）策略（副本、回退到原生）。
4. LMCache 存储到机械硬盘上的 Ceph。对于 70B FP8 上 4K token 的 KV（500 MB），读取时间 vs 重新预填充是多少？
5. 论证 vLLM 0.11.0 异步路径是否"免费" — 开销隐藏在哪里？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Production-stack | "参考部署" | vLLM 的 Kubernetes Helm chart + operator |
| 连接器 API（Connector API） | "KV 后端接口" | vLLM 0.9.0+ 可插拔 KV 存储接口 |
| 原生 CPU 卸载（Native CPU offload） | "引擎本地溢出" | 将 KV 存储在同一引擎的主机 RAM 中 |
| LMCache | "集群 KV 缓存" | 跨引擎 KV 缓存服务器，位于 CPU DRAM + 磁盘上 |
| 0.11.0 异步（0.11.0 async） | "非阻塞卸载" | 卸载隐藏在引擎流之后 |
| 抢占（Preemption） | "驱逐以腾出空间" | HBM 满时的 KV 缓存洗牌 |
| 前缀复用（Prefix reuse） | "相同系统提示词" | 多个查询共享开头；缓存命中 |
| Ceph 层（Ceph tier） | "磁盘层" | 缓存层次结构中 DRAM 下方的持久存储 |
