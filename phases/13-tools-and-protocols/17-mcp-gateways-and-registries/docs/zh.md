# MCP 网关与注册中心 — 企业控制平面

> 企业不能让每位开发者随意安装 MCP 服务器。网关 (gateway) 集中处理认证、RBAC、审计、限流、缓存和工具投毒检测，再将合并后的工具表面作为单一 MCP 端点暴露。官方 MCP 注册中心 (Official MCP Registry)（由 Anthropic + GitHub + PulseMCP + Microsoft 共管、命名空间验证）是规范上游。本课定位网关的位置、演练一个最小实现，并概览 2026 供应商生态。

**类型：** 学习
**语言：** Python（标准库，最小网关）
**前置课程：** Phase 13 · 15（工具投毒），Phase 13 · 16（OAuth 2.1）
**时长：** 约 45 分钟

## 学习目标

- 解释 MCP 网关的位置（位于 MCP 客户端与多个后端 MCP 服务器之间）。
- 实现网关的五项职责：认证、RBAC、审计、限流、策略。
- 在网关层面强制执行固定工具哈希清单。
- 区分官方 MCP 注册中心与元注册中心（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题背景

一家世界 500 强企业有 30 个已批准的 MCP 服务器、5000 名开发者、合规与审计要求，以及一个需要集中策略的安全团队。让每位开发者在自己的 IDE 中随意安装任意服务器是不可接受的。

网关模式：

1. 网关作为单一 Streamable HTTP 端点运行，开发者连接到此。
2. 网关持有每个后端 MCP 服务器的凭证。
3. 每个开发者请求通过网关自身的 OAuth 进行认证和作用域限定。
4. 网关将调用路由到后端服务器，同时应用策略。
5. 所有调用记录用于审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway 均在 2025-2026 年发布了网关或网关功能。

与此同时，官方 MCP 注册中心作为规范上游上线：经过审核、命名空间验证、反向 DNS 命名的服务器，网关可从中拉取。元注册中心（Glama、MCPMarket、MCP.so、Smithery、LobeHub）则聚合来自多个来源的服务器。

## 核心概念

### 网关的五项职责

1. **认证 (Auth)。** OAuth 2.1 识别开发者；映射到用户角色。
2. **RBAC。** 逐用户策略：哪些服务器、哪些工具、哪些作用域。
3. **审计 (Audit)。** 每次调用记录谁、做了什么、何时、结果。
4. **限流 (Rate limit)。** 逐用户 / 逐工具 / 逐服务器的上限以防滥用。
5. **策略 (Policy)。** 拒绝被投毒的描述、执行 Rule of Two、脱敏 PII。

### 网关作为单一端点

对开发者而言，网关看起来就是一个 MCP 服务器。内部它路由到 N 个后端。会话 ID (Phase 13 · 09) 在边界处被重写。

### 凭证保管

开发者永远看不到后端令牌。网关持有它们（或代理到持有令牌的身份提供商）。一个在网关上拥有 `notes:read` 的开发者可以传递性地访问笔记 MCP 服务器，但这仅在绑定传递性访问的策略下进行。

### 网关层的工具哈希固定

网关持有一份已批准工具描述的清单（SHA256 哈希）。在发现阶段，网关获取每个后端的 `tools/list`，将哈希与清单比对，移除任何描述已变异的工具。这是 Phase 13 · 15 中 rug-pull 防御的集中化应用。

### 策略即代码

高级网关用 OPA/Rego、Kyverno 或 Styra 表达策略。如"用户 `alice` 只能对 org `acme` 中的仓库调用 `github.open_pr`"这样的规则以声明式方式编码。简单网关使用手写 Python。两种形式都有效。

### 会话感知路由

当用户的会话涉及多个服务器时，网关进行多路复用：开发者的单一 MCP 会话持有 N 个后端会话，每个服务器一个。来自任何后端的通知通过网关路由到开发者的会话。

### 命名空间合并

网关合并来自所有后端的工具命名空间，通常在冲突时加前缀。`github.open_pr`、`notes.search`。这使路由无歧义。

### 注册中心

- **官方 MCP 注册中心 (`registry.modelcontextprotocol.io`)。** 由 Anthropic、GitHub、PulseMCP、Microsoft 共同管理。命名空间验证（反向 DNS：`io.github.user/server`）。经过基本质量过滤。
- **Glama。** 以搜索为中心的元注册中心，聚合多个来源。
- **MCPMarket。** 商业化目录，列出供应商产品。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 类包管理器的安装流程。
- **LobeHub。** 集成在其 LobeChat 应用中的注册中心。

企业网关默认从官方注册中心拉取，允许管理员从元注册中心策划添加，拒绝任何未固定的来源。

### 反向 DNS 命名

官方注册中心要求公共服务器使用反向 DNS 名称：`io.github.alice/notes`。命名空间防止抢注并使信任委派更清晰。

### 供应商概览，2026 年 4 月

| 供应商 | 优势 |
|--------|------|
| Cloudflare MCP Portals | 边缘托管；OAuth 集成；免费层 |
| Kong AI Gateway | K8s 原生；细粒度策略；日志导出到 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | DevOps 导向；指标优先 |
| MintMCP | 开发者平台导向 |
| Envoy AI Gateway | 开源；可定制过滤器 |

Phase 17（生产基础设施）将深入探讨网关运维。

## 动手实践

`code/main.py` 提供约 150 行的最小网关：通过伪造 Bearer 令牌认证用户，持有逐用户 RBAC 策略，将请求路由到两个后端 MCP 服务器，将每次调用写入审计日志，执行限流，并拒绝描述哈希与固定清单不匹配的任何后端工具。

关注点：

- `RBAC` 字典以 `user_id` 为键，值为允许的 `server_tool` 条目。
- `AUDIT_LOG` 为仅追加的事件列表。
- 限流使用逐用户令牌桶。
- 固定清单为 `server::tool -> hash` 的字典。

## 交付产物

本课产出 `outputs/skill-gateway-bootstrap.md`。给定一个企业 MCP 计划（用户、后端、合规要求），该技能产出网关配置规格。

## 练习

1. 运行 `code/main.py`。以允许的用户发起调用；再以不允许的用户发起；然后进行速率超限的突发请求。验证三种流程。

2. 添加一个策略，在将结果返回客户端前脱敏 PII。对 SSN 格式的字符串使用简单正则；注意其局限性（邮箱、电话号码）。

3. 扩展审计日志以发送 OpenTelemetry GenAI span。Phase 13 · 20 涵盖了确切的属性。

4. 为一个 50 人开发团队设计 RBAC 策略，包含五个后端（notes、github、postgres、jira、slack）。谁对每个后端只读？谁有写权限？

5. 从头到尾阅读 Cloudflare 企业 MCP 博文。找出 Cloudflare 提供而本课 stdlib 网关未实现的一项功能。

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 网关 | "MCP 代理" | 位于客户端和后端之间的集中化服务器 |
| 凭证保管 | "后端令牌留在服务端" | 开发者永远看不到上游令牌 |
| 会话感知路由 | "多后端会话" | 网关为每个开发者会话多路复用 N 个后端会话 |
| 工具哈希固定 | "已批准清单" | 每个已批准工具描述的 SHA256；集中阻断 rug-pull |
| RBAC | "逐用户策略" | 针对工具和服务器的基于角色的访问控制 |
| 策略即代码 | "声明式规则" | 在网关执行的 OPA/Rego、Kyverno、Styra 策略 |
| 审计日志 | "谁、做了什么、何时" | 用于合规的仅追加事件日志 |
| 限流 | "逐用户令牌桶" | 每分钟上限以防滥用 |
| 官方 MCP 注册中心 | "规范上游" | `registry.modelcontextprotocol.io`，命名空间验证 |
| 反向 DNS 命名 | "注册中心命名空间" | `io.github.user/server` 命名约定 |

## 延伸阅读

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — 规范上游，命名空间验证
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — 带 OAuth 和策略的网关模式
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能对比文章
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — IBM 的企业网关
