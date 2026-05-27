# 实战项目 06 — Kubernetes DevOps 故障排查智能体

> AWS 的 DevOps Agent 已正式发布（GA），Resolve AI 发布了其 K8s 运维手册（Playbook），NeuBird 演示了语义监控，Metoro 将 AI SRE 与每服务 SLO 绑定。生产形态已经确定：告警 webhook 触发，智能体读取遥测数据，遍历 K8s 对象图，对根因假设进行排名，并发布带有审批按钮的 Slack 简报。默认只读。每次修复操作都由人类门控。本实战项目就是构建这个智能体，在 20 个合成事件上进行评估，并在三个共享案例上与 AWS 的 Agent 进行对比。

**类型：** 实战项目（Capstone）
**语言：** Python（智能体）、TypeScript（Slack 集成）
**前置课程：** 第 11 阶段（LLM 工程）、第 13 阶段（工具与 MCP）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及阶段：** P11 · P13 · P14 · P15 · P17 · P18
**时间：** 30 小时

## 问题

2025-2026 年的 SRE 叙事变成了："AI 智能体分类事件，人类审批修复。"AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产环境中以这种形态运行。智能体读取 Prometheus 指标、Loki 日志、Tempo 追踪、kube-state-metrics 以及 K8s 对象的知识图谱。它在五分钟内生成带有遥测引用的排名根因假设。它绝不会在没有通过 Slack 获得明确人类批准的情况下执行破坏性命令。

大部分困难工作在于范围界定和安全，而非推理。智能体需要一个默认只读的 RBAC 表面、一个加固的 MCP 工具服务器，以及记录每个被考虑的命令与被执行的命令的审计日志。它需要知道何时超出自身能力范围并升级处理。而且它必须足够便宜地运行，以免 OOM-kill 级联产生 5000 美元的智能体账单。

## 概念

该智能体在知识图谱上运行。节点是 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）加上遥测源（Prometheus 序列、Loki 流、Tempo 追踪）。边编码了所有权（Pod -> ReplicaSet -> Deployment）、调度（Pod -> Node）和观测（Pod -> Prometheus 序列）。图谱通过 kube-state-metrics 同步保持新鲜，并在每次告警时重新采样。

当告警触发时，智能体从受影响的对象开始根因分析。它遍历边，拉取相关的遥测切片（最近 15 分钟），并起草假设。假设按证据排名：有多少遥测引用支持它、多近的时间、多具体。前 3 个假设发送到 Slack，附带图谱路径可视化和修复操作的审批按钮。

修复操作是门控的。允许的默认操作是只读的。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 审批；ArgoCD 回滚钩子需要一个智能体永远不持有的认证 token。审计日志记录智能体*考虑过*的每个命令——不仅仅是已执行的——以便审查过程捕获险兆事件（near-miss）。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI 接收器
           |
           v
   LangGraph 根因分析智能体
           |
           +---- 只读 MCP 工具 ----+
           |                      |
           v                      v
   K8s 知识图谱              遥测切片
     （Neo4j / kuzu）       Prometheus、Loki、Tempo
   所有权 + 调度             最近 15 分钟，限定范围
           |
           v
   假设排名（证据权重）
           |
           v
   Slack 简报 + 审批按钮
           |
           v（已批准）
   ArgoCD 回滚钩子 / PagerDuty 升级
           |
           v
   审计日志：每个命令的"已考虑 vs 已执行"
```

## 技术栈

- 可观测性源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：Neo4j（托管）或 kuzu（嵌入式），包含 K8s 对象 + 遥测边
- 智能体：LangGraph 配合每工具允许列表，默认只读
- 工具传输：FastMCP 通过 StreamableHTTP；破坏性工具在审批门控后的独立服务器上
- 模型：Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要
- 修复：ArgoCD 回滚 webhook、PagerDuty 升级、Slack 审批卡片
- 审计：仅追加的结构化日志（已考虑、已执行、已批准、结果）
- 部署：K8s 部署，拥有自己的窄范围 RBAC 角色；独立命名空间

## 构建步骤

1. **图谱摄入。** 每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测叠加边：OBSERVED_BY（Pod 被 Prometheus 序列观测）。

2. **告警接收器。** FastAPI 端点，接受 PagerDuty 或 Alertmanager webhook。提取受影响的对象和 SLO 违规。

3. **只读工具表面。** 通过 FastMCP 包装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每个工具都有窄范围的 RBAC 动词（"get"、"list"、"describe"）。默认服务器中没有"delete"、"exec"、"scale"。

4. **根因分析智能体。** LangGraph 包含三个节点：`sample` 拉取最近 15 分钟的遥测切片，`walk` 查询图谱中的邻近对象，`hypothesize` 起草带有遥测引用的排名根因候选。

5. **证据评分。** 每个假设的得分 = 近因性 × 特异性 × 图谱路径长度倒数 × 引用数量。返回前 3 个。

6. **Slack 简报。** 发布一个附件，包含假设、图谱路径可视化（服务端渲染的子图图像）以及最多一个修复操作的审批按钮。

7. **修复门控。** 破坏性工具（缩容、回滚、删除）位于第二个 MCP 服务器上，需要审批 token。智能体只有在 Slack 卡片被人类批准后才能调用它们。

8. **审计日志。** 仅追加的 JSONL：对于每个候选命令，记录是否被考虑、是否被执行、谁批准的。每日发送到 S3。

9. **合成事件套件。** 构建 20 个场景：OOMKill 级联、DNS 抖动、HPA 震荡、PVC 填满、嘈杂邻居、故障 sidecar、错误的 ConfigMap 发布、证书轮换、镜像拉取回退等。对智能体的根因准确率和假设生成时间进行评分。

## 使用方式

```
webhook：alert.pagerduty.com -> checkout-api SLO 违规，错误率 14%
[图谱]   受影响：Deployment checkout-api（3 个 Pod，Node ip-10-2-3-4）
[遍历]   邻居：ReplicaSet checkout-api-abc、Service checkout-api、
           14 分钟前的最近发布
[采样]   prometheus 错误率 14%，上升趋势；loki 在 /api/v2/pay 上返回 500
[假设]   #1 错误发布：最新镜像 checkout-api:v2.41 导致 /healthz 失败
          引用：deploy.yaml（rev 42）、prometheus errorRate、loki 500 堆栈
[Slack]  [回滚到 v2.40]  [升级]  [忽略]
         （需要审批；智能体不会单方面回滚）
```

## 交付标准

`outputs/skill-devops-agent.md` 是交付物。给定一个 K8s 集群和告警源，智能体生成排名的根因假设和 Slack 门控的修复流程。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 场景套件上的根因分析准确率 | 20 个合成事件中 ≥80% 正确根因 |
| 20 | 安全 | 审计日志中破坏性操作守卫从未在无 Slack 审批的情况下触发 |
| 20 | 假设生成时间 | 从告警到 Slack 简报的 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设都有图谱路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端工作 |
| **100** | | |

## 练习

1. 在 AWS DevOps Agent 演示的三个相同事件上运行你的智能体。发布并排对比。报告智能体在哪些方面存在差异。
2. 添加"险兆"审计：标记智能体*考虑过*的任何在没有审批的情况下本应是破坏性的命令。测量一周内的险兆率。
3. 将假设模型从 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量根因分析准确率差异和每次事件的美元成本。
4. 构建因果过滤器：区分相关的遥测尖峰和真正的根因。在 20 个场景标签上训练一个小型分类器。
5. 添加回滚预演：在具有相同清单的 staging 集群上执行 ArgoCD 回滚。在 Slack 审批按钮之前验证实时集群中的回滚计划。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| K8s 知识图谱 | "集群图谱" | 节点 = K8s 对象 + 遥测序列；边 = 所有权、调度、观测 |
| 默认只读 | "限定范围 RBAC" | 智能体的服务账户只有 get/list/describe 动词；破坏性动词位于审批门控后的独立服务器上 |
| 审计日志 | "已考虑 vs 已执行" | 每个候选命令的仅追加记录，包括是否运行、谁批准的 |
| 假设排名 | "证据评分" | 近因性 × 特异性 × 图谱路径长度倒数 × 引用数量 |
| Slack 审批卡片 | "人机协同门控" | 带有修复按钮的交互式 Slack 消息；智能体在人类点击之前无法继续 |
| 遥测引用 | "证据指针" | 支持某个声明的 Prometheus 查询、Loki 选择器或 Tempo 追踪 URL |
| MTTR | "解决时间" | 从告警触发到 SLO 恢复的墙钟时间 |

## 扩展阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年规范参考
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞争者参考
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图谱方法
- [Metoro AI SRE](https://metoro.io) — SLO 优先的生产框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考智能体编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 门控修复目标
