# MCP 安全 II — OAuth 2.1、资源指示器、增量作用域

> 远程 MCP 服务器需要的是授权 (authorization)，而非仅仅认证 (authentication)。2025-11-25 版规范与 OAuth 2.1 + PKCE + 资源指示器 (RFC 8707) + 受保护资源元数据 (RFC 9728) 对齐。SEP-835 新增了增量作用域同意 (incremental scope consent)，通过 403 `WWW-Authenticate` 实现逐步授权 (step-up authorization)。本课将该逐步授权流程实现为状态机，以便你看清每一跳。

**类型：** 构建
**语言：** Python（标准库，OAuth 状态机模拟器）
**前置课程：** Phase 13 · 09（传输层），Phase 13 · 15（安全 I）
**时长：** 约 75 分钟

## 学习目标

- 区分资源服务器 (resource server) 和授权服务器 (authorization server) 的职责。
- 走通受 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防范混淆代理攻击 (confused-deputy attack)。
- 实现逐步授权：服务器以 403 + `WWW-Authenticate` 要求更高作用域；客户端重新提示用户同意后重试。

## 问题背景

早期 MCP（2025 年之前）以临时 API 密钥甚至无认证方式运行远程服务器。2025-11-25 版规范用完整的 OAuth 2.1 配置文件 (profile) 填补了这一缺口。

三个真实场景：

- **普通远程服务器。** 用户安装一个访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。OAuth 2.1 + PKCE 是正确的形式。
- **作用域升级。** 一个笔记服务器被授予了 `notes:read`，之后某个操作需要 `notes:write`。逐步授权 (SEP-835) 无需重做整个流程，而是仅请求附加作用域。
- **混淆代理防护。** 客户端持有面向 Server A 的令牌。恶意 Server A 试图将该令牌呈交给 Server B。资源指示器 (RFC 8707) 将令牌固定到其目标受众。

OAuth 2.1 本身并不新颖。新颖之处在于 MCP 的配置文件：限定必需的流程（仅授权码 + PKCE；无隐式流、无客户端凭证默认模式）、每次令牌请求必须携带资源指示器，以及发布受保护资源元数据以便客户端知道如何发现授权服务器。

## 核心概念

### 角色

- **客户端 (Client)。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器 (Resource server)。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **授权服务器 (Authorization server)。** 签发令牌。可以与资源服务器同一主机，也可以是独立的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的配置文件中，资源服务器与授权服务器可以是同一主机，但应当 (SHOULD) 以不同的 URL 加以区分。

### 授权码 + PKCE

流程如下：

1. 客户端生成 `code_verifier`（随机串）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向至 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向至 `redirect_uri?code=...`。
4. 客户端 POST `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器验证 verifier 的哈希与存储的 challenge 一致后签发访问令牌。
6. 客户端在每次请求资源服务器时携带 `Authorization: Bearer ...`。

PKCE 防止授权码拦截攻击。资源指示器防止令牌在其他地方被使用。

### 受保护资源元数据 (RFC 9728)

资源服务器发布 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器地址。减少配置——客户端只需要资源 URL。

### 资源指示器 (RFC 8707)

令牌请求中的 `resource` 参数将令牌的目标受众 (audience) 固定。签发的令牌包含 `aud: "https://notes.example.com"`。另一个 MCP 服务器收到此令牌时检查 `aud` 并拒绝。

### 作用域模型

作用域是空格分隔的字符串。常见 MCP 惯例：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理能力（谨慎使用）
- `profile:read` 用于身份

作用域选择应遵循最小权限原则：现在请求所需的，需要更多时再逐步升级。

### 逐步授权 (SEP-835)

用户授予 `notes:read`。随后用户要求 Agent 删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端识别到 `insufficient_scope` 错误，弹出同意对话框请求附加作用域，执行一次小型 OAuth 流程后重试请求。

### 令牌受众验证

每个请求：服务器检查 `token.aud == self.resource_url`。不匹配则 401。这阻止了跨服务器令牌重用。

### 短期令牌与轮转

访问令牌应当 (SHOULD) 是短期的（默认 1 小时）。刷新令牌在每次刷新时轮转。客户端在后台静默刷新。

### 禁止令牌传递

采样服务器 (Phase 13 · 11) 必须不能 (MUST NOT) 将客户端的令牌传递给其他服务。采样请求即为边界。

### 混淆代理防护

令牌绑定到 `aud`，客户端绑定到 `client_id`。每个请求同时验证二者。规范明确禁止了旧式"令牌传递"模式——这在 MCP 之前的远程工具生态中很常见。

### 客户端 ID 发现

每个 MCP 客户端在固定 URL 发布其元数据。授权服务器可获取客户端的元数据文档以发现重定向 URI 和联系信息。这消除了手动客户端注册。

### 网关与 OAuth

Phase 13 · 17 展示企业网关如何处理 OAuth：网关持有上游服务器的凭证，面向客户端的令牌由网关签发，上游令牌永远不出网关。这翻转了信任模型——用户向网关认证一次；网关处理 N 个服务器的授权。

## 动手实践

`code/main.py` 将完整的 OAuth 2.1 逐步授权流程模拟为状态机。它实现了：

- PKCE code-verifier / challenge 生成。
- 带资源指示器的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的令牌验证。
- `insufficient_scope` 时的逐步授权。

本课不含 HTTP 服务器；状态机在内存中运行，以便你追踪每一跳。Phase 13 · 17 的网关课将其连接到真实传输层。

## 交付产物

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带工具的远程 MCP 服务器，该技能设计作用域集合、固定规则和逐步授权策略。

## 练习

1. 运行 `code/main.py`。追踪两次作用域逐步升级流程。注意哪些步骤在 step-up 时会重复。

2. 添加刷新令牌轮转：每次刷新签发新刷新令牌并使旧的失效。模拟在轮转后使用被盗刷新令牌并确认失败。

3. 使用 stdlib `http.server` 将受保护资源元数据端点实现为真实 HTTP 响应。参照 Lesson 09 的 `/mcp` 端点。

4. 为 GitHub MCP 服务器设计作用域层级：读仓库、写 PR、审批 PR、合并 PR、管理员。在各层级之间使用逐步授权。

5. 阅读 RFC 8707 和 RFC 9728。找到 9728 中 MCP 使用方式与 RFC 示例不同的那一个字段。（提示：与 `scopes_supported` 有关。）

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| OAuth 2.1 | "现代 OAuth" | 整合后的 RFC，强制 PKCE 并禁止隐式流 |
| PKCE | "持有证明" | code verifier + challenge 防止授权码拦截 |
| 资源指示器 | "令牌受众" | RFC 8707 `resource` 参数将令牌固定到单一服务器 |
| 受保护资源元数据 | "发现文档" | RFC 9728 `.well-known/oauth-protected-resource` |
| 逐步授权 | "增量同意" | SEP-835 按需添加作用域的流程 |
| `insufficient_scope` | "403 + WWW-Authenticate" | 服务器要求重新同意更大作用域的信号 |
| 混淆代理 | "跨服务令牌重用" | 受信方不当转发令牌的攻击 |
| 短期令牌 | "访问令牌 TTL" | 快速过期的 Bearer 令牌；由刷新令牌续期 |
| 作用域层级 | "最小权限栈" | 分级作用域集合，逐层 step-up |
| 客户端 ID 元数据 | "客户端发现文档" | 客户端发布自身 OAuth 元数据的 URL |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 规范的 MCP OAuth 配置文件
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更解读
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定 RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档 RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实用 step-up 流程演练
