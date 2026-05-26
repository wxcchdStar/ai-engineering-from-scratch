# MCP 生产环境认证 — DCR、JWKS 轮转、受众固定令牌与 iii 原语

> 第 16 课在内存中搭建了 OAuth 2.1 状态机。到 2026 年，你交付给真实企业的每个 MCP 服务器都位于生产环境认证之后：动态客户端注册 (RFC 7591)、授权服务器元数据发现 (RFC 8414)、不会在凌晨 3 点破坏令牌验证的 JWKS 轮转，以及拒绝混淆代理重放的受众固定令牌。本课通过 iii 原语将一切串联起来——`iii.registerTrigger` 用于 HTTP 和 cron、`iii.registerFunction` 用于认证逻辑、`state::set/get` 用于缓存密钥——使认证表面如同引擎中其他工作负载一样，可观测、可重启、可重放。

**类型：** 构建
**语言：** Python（标准库，iii 原语在课程环境中模拟）
**前置课程：** Phase 13 · 16（OAuth 2.1 状态机），Phase 13 · 17（网关）
**时长：** 约 90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证契约。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无需管理员干预即可注册。
- 使用 cron 触发器缓存并轮转 JWKS 密钥，使签名验证在密钥轮换后仍能正常工作。
- 使用 RFC 8707 资源指示器将令牌固定到单一 MCP 资源，并拒绝混淆代理重放。
- 将每个端点和后台任务连接为 iii 原语——HTTP 触发器、cron 触发器、命名函数和 `state::*` 读取——使一次重启即可重建认证表面。
- 阅读 IdP 能力矩阵，当 IdP 无法满足 MCP 认证配置文件时拒绝部署。

## 问题背景

第 16 课模拟器在内存中运行 OAuth 2.1。生产环境有三个纯内存模拟器无法看到的运维缺口。

第一个缺口是注册。一个真实企业运行数百个 MCP 服务器和数千个 MCP 客户端。运维人员不会手动注册每一个 Cursor 用户为 OAuth 客户端。RFC 7591 动态客户端注册允许客户端向授权服务器 `POST /register` 并当场获取 `client_id`（以及可选的 `client_secret`）。服务器在其 RFC 8414 元数据中发布 `registration_endpoint`；客户端发现它而无需带外配置。

第二个缺口是密钥轮转。JWT 验证依赖于授权服务器的签名密钥，以 JSON Web Key Set (JWKS) 形式发布。授权服务器按计划轮转这些密钥（通常每小时一次，在安全事件响应中可能更频繁）。一个在启动时一次性拉取 JWKS 的 MCP 服务器在到达轮转窗口之前验证正常——之后每个请求都会失败直至重启。生产环境将 JWKS 连接为带有刷新作业的缓存值，在旧密钥过期之前覆盖缓存，外加一个缓存未命中时的补救拉取，以处理由比缓存更新的密钥签名的令牌。

第三个缺口是受众绑定。第 16 课介绍了 RFC 8707 资源指示器。在生产环境中，该指示器成为每个请求上的硬性声明检查。MCP 服务器将 `token.aud` 与自身的规范资源 URL 进行比较，不匹配时用 HTTP 401 拒绝。这是唯一能防止上游 MCP 服务器（或持有本应发给某服务器的令牌的恶意客户端）在同一信任网格中向另一个服务器重放该令牌的防御手段。

本课将每一个缺口都视为 iii 原语来处理。元数据文档是一个返回函数输出的 HTTP 触发器。JWKS 轮转是调用 `auth::rotate-jwks` 的 cron 触发器，该函数写入 `state::set("auth/jwks/<issuer>", ...)`。JWT 验证是一个函数，其他组件通过 `iii.trigger("auth::validate-jwt", token)` 调用。MCP 服务器本身只是另一个 HTTP 触发器，在分发前调用验证。重启引擎：触发器注册表重建；状态持久化；认证表面即可投入运行，无需手动对账。

## 核心概念

### RFC 8414 — OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

客户端拿到 MCP 资源 URL 后进行链式发现：RFC 9728 的 `oauth-protected-resource`（资源服务器的文档）指明 issuer，然后 `oauth-authorization-server`（本 RFC）指明每个端点。客户端从不硬编码授权 URL。

在为 MCP 信任某个 IdP 之前需要验证的契约：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 规定的 PKCE）。
- `grant_types_supported` 包含 `authorization_code`，拒绝 `password` 和 `implicit`。
- `registration_endpoint` 存在（支持 RFC 7591）。
- `response_types_supported` 正好是 `["code"]`（OAuth 2.1 要求）。

如果其中任何一项缺失，MCP 服务器拒绝部署到该 IdP。是部署清单有问题，而非代码。

### RFC 9728（回顾）— 受保护资源元数据

第 16 课涵盖过 RFC 9728。生产环境的不同之处在于：该文档是客户端查找*此* MCP 服务器所信任的授权服务器的唯一位置。单个 MCP 服务器可以接受来自多个 IdP 的令牌（一个用于员工、一个用于合作伙伴）。RFC 9728 声明该集合；RFC 8414 记录每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 — 动态客户端注册

没有 DCR 时，每个 MCP 客户端（Cursor、Claude Desktop、自定义 agent）都需要与 IdP 管理员进行带外交互。有了 DCR，客户端提交：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器返回 `client_id` 和用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是运行在用户设备上的 MCP 客户端的正确默认值。它们只获得 `client_id` ——没有可被泄露的 `client_secret`。PKCE 提供了公共客户端所需的持有证明。

三个生产环境陷阱：

- 注册端点必须按源 IP 限流。否则，恶意行为者编写脚本进行数百万次虚假注册，耗尽 `client_id` 命名空间。iii 使这变得简单：注册 HTTP 触发器在分发到注册器之前调用 `auth::rate-limit` 函数。
- 某些企业 IdP 要求 `software_statement`（一份为客户端担保的签名 JWT）。本课模拟部分跳过它；生产环境连接一个验证步骤，拒绝除 localhost 重定向 URI 之外的任何未签名注册。
- `registration_access_token` 必须以哈希形式存储，而非明文。此令牌被盗意味着攻击者可以改写客户端的重定向 URI。

### RFC 8707（回顾）— 资源指示器

第 16 课确立了基本形式。生产规则：每个令牌请求都包含 `resource=<canonical-mcp-url>`，MCP 服务器在每次调用时验证 `token.aud` 匹配自身的资源 URL。如果 MCP 服务器通过 `https://notes.example.com/mcp` 可达，规范 URL 为 `https://notes.example.com`——排除路径部分，使单个服务器在同一受众下托管多个路径。

### RFC 7636（回顾）— PKCE

PKCE 在 OAuth 2.1 中是强制性的。本课的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器拒绝任何无 verifier 或 verifier 哈希不匹配存储 challenge 的令牌请求。

### MCP 规范 2025-11-25 认证配置文件

MCP 规范 (2025-11-25) 精确规定了 MCP 服务器授权层必须做什么：

- 发布 `/.well-known/oauth-protected-resource` (RFC 9728)。
- 仅通过 `Authorization: Bearer ...` 接受令牌。
- 在每个请求中验证 `aud`、`iss`、`exp` 和所需作用域。
- 对每个 401 和 403 以携带 `Bearer error=...` 的 `WWW-Authenticate` 响应，并视情况包含 `scope=` 和 `resource=` 参数。
- 拒绝 `aud` 与规范资源不匹配的令牌。
- 拒绝 `iss` 不在受保护资源元数据的 `authorization_servers` 列表中的令牌。

OAuth 2.1 草案是基础；RFC 8414/7591/8707/9728 + RFC 7636 是表面；MCP 规范是配置文件。

### IdP 能力矩阵

并非每个 IdP 都支持完整的 MCP 配置文件。下表记录了截至 2025-11-25 规范的事实能力声明。这是*部署门槛*，而非推荐。

| IdP 类别 | RFC 8414 元数据 | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|
| 自托管 (Keycloak) | 是 | 是 | 是（自 24.x） | 是 | 本课 MCP 配置文件的参考 IdP；端到端支持每个 RFC。 |
| 企业 SSO (Microsoft Entra ID) | 是 | 是（高级层） | 是 | 是 | DCR 可用性因租户层级而异；部署前在目标租户中验证。 |
| 企业 SSO (Okta) | 是 | 是（Okta CIC / Auth0） | 是 | 是 | DCR 在 Auth0（现为 Okta CIC）上可用；经典 Okta 组织需要管理员预注册。 |
| 社交登录 IdP（通用） | 视情况 | 极少 | 极少 | 是 | 大多数社交 IdP 将客户端视为静态合作伙伴；不要依赖 DCR。仅用作身份源，在此之上叠加自己的 MCP 感知授权服务器。 |
| 自定义/自研 | 视情况 | 视情况 | 视情况 | 视情况 | 如果你自研，应交付完整配置文件。跳过上述四个 RFC 中的任何一个都会破坏 MCP 认证契约。 |

部署清单的拒绝规则：如果所选 IdP 不返回 `registration_endpoint` 且未在 `code_challenge_methods_supported` 中列出 `S256`，MCP 服务器拒绝启动。不存在降级模式。

### 基于 iii 的 JWKS 轮转模式

生产环境的故障模式是 JWKS 缓存过时。用 cron 触发器和 `state::*` 缓存解决：

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

每六小时，cron 触发器调用 `auth::rotate-jwks`，该函数拉取 `<issuer>/.well-known/jwks.json` 并写入 `state::set("auth/jwks/<issuer>", {keys, fetched_at})`。验证器从 `state::get` 读取。一个 `kid` 在缓存中缺失的令牌会触发同步 `auth::rotate-jwks` 调用作为补救。这同时处理了两种情况：定时轮转（cron）和密钥重叠窗口（同步补救）。

状态结构：

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时存在两个密钥是稳态。授权服务器通过在退役前一个密钥 (`k_2026_03`) 之前引入下一个密钥 (`k_2026_04`) 来轮转，使旧密钥签发的令牌在其过期前保持有效。缓存持有并集；验证器按 `kid` 选取。

### iii 原语连接（本课的核心内容）

五个原语构成了认证表面：

```python
# 1. RFC 8414 元数据文档
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 动态客户端注册
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT 验证作为可调用函数（资源服务器触发它）
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. 增量作用域的逐步授权签发（来自 L16 的 SEP-835）
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. cron 驱动的 JWKS 轮转
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP 服务器本身从不直接调用验证。它做的是：

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

这种间接调用是 iii 的核心价值。明天你可以将验证器替换为并行查询两个 IdP 的扇出，或添加 span emitter，或缓存正向验证结果。MCP 服务器无需更改。

### 混淆代理演练：受众绑定

Server A (`notes.example.com`) 和 Server B (`tasks.example.com`) 都注册于同一授权服务器。Server A 被攻陷。攻击者获取用户的 notes 令牌并向 Server B 重放。

Server B 的验证器：

1. 解码 JWT，按 `kid` 拉取 JWKS，验证签名。
2. 检查 `iss` 是否在受保护资源元数据的 `authorization_servers` 中。（通过——同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——令牌的 `aud` 是 `https://notes.example.com`。）
4. 返回 401 及 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`。

受众声明是协议层面抵御这一攻击的唯一防御。为性能而跳过它是最常见的生产错误；验证器必须在每个请求上运行，而不仅在会话开始时。

### 故障模式

- **JWKS 过时。** 验证器在密钥轮转后拒绝有效令牌。修复方法是上述 cron+补救模式。永远不要在没有刷新作业的情况下缓存 JWKS。
- **缺失 `aud` 声明。** 某些 IdP 默认省略 `aud` 除非令牌请求中存在 `resource`。验证器必须拒绝缺少 `aud` 的令牌，而不是将缺失视为通配符。
- **作用域升级竞态。** 两个并发的逐步授权流程可以同时成功并产生具有不同作用域的两个访问令牌。验证器必须使用请求中提供的令牌，而不是查找"用户当前作用域"——那会引入 TOCTOU 窗口。
- **注册令牌被盗。** 泄露的 `registration_access_token` 允许攻击者改写重定向 URI。在存储时对其哈希；要求客户端在每次更新时提供明文；怀疑时轮转。
- **`iss` 未固定。** 一个接受任意 `iss` 的验证器使攻击者可以搭建自己的授权服务器，为目标受众注册客户端并签发令牌。受保护资源元数据的 `authorization_servers` 列表是允许列表；强制执行它。

## 动手实践

`code/main.py` 用标准库 Python 和一个模拟 `iii.registerFunction`、`iii.registerTrigger`、`iii.trigger` 和 `state::set/get` 的小型 `iii_mock` 注册表走通完整生产流程。流程如下：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点，发现注册端点。
3. MCP 客户端提交到 `/register` (RFC 7591) 并获取 `client_id`。
4. MCP 客户端运行受 PKCE 保护、携带 `resource` 指示器 (RFC 8707) 的授权码流程 (RFC 7636)。
5. MCP 客户端以 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器触发 `auth::validate-jwt`，该函数从 `state::get` 读取 JWKS。
7. cron 触发器执行 `auth::rotate-jwks`，替换 state 中的 JWKS。
8. 下一次调用使用新密钥验证而无需重启。
9. 对另一个 MCP 资源的混淆代理尝试获得 401（受众不匹配）。

此处的模拟 JWT 使用 HS256 和共享密钥（以便课程仅在标准库上运行）。生产环境使用 RS256 或 EdDSA 配合上述 JWKS 模式；验证逻辑在其他方面一致。

## 交付产物

本课产出 `outputs/skill-mcp-auth-iii.md`。给定 MCP 服务器配置和 IdP 能力集，该技能输出需注册的 iii 原语、JWKS 轮转计划、作用域映射，以及 IdP 不支持完整 RFC 配置文件时应应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。追踪 9 步流程。注意 `state::get` 在 `auth::rotate-jwks` 覆盖之前返回过时数据的位置，以及下一个请求现在如何根据新密钥进行验证。

2. 向受保护资源元数据的 `authorization_servers` 列表添加一个新 IdP。签发一个由新 IdP 签名的令牌并确认验证器接受它。签发一个由未列出的 IdP 签名的令牌并确认验证器以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝。

3. 将 `auth::rate-limit` 实现为 iii 函数，并在注册 HTTP 触发器内部、注册器运行之前调用它。使用存储在 `state::set("auth/ratelimit/<ip>", ...)` 中的逐来源 IP 令牌桶。

4. 阅读 RFC 7591，找出本课 `/register` 处理器未验证的两处字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI scheme。）

5. 阅读 MCP 规范 2025-11-25 授权章节。找到本课验证器当前未发出的一项关于 `WWW-Authenticate` 头的规范性要求。添加它。

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| ASM | "OAuth 元数据文档" | RFC 8414 的 `/.well-known/oauth-authorization-server` JSON |
| DCR | "自助客户端注册" | RFC 7591 的 `POST /register` 流程 |
| JWKS | "JWT 验证公钥" | JSON Web Key Set，从 `jwks_uri` 拉取，由 `kid` 索引 |
| Resource indicator | "受众参数" | RFC 8707 `resource` 参数，将令牌固定到单一服务器 |
| `aud` 声明 | "受众" | 验证器与规范资源 URL 对比的 JWT 声明 |
| Confused deputy | "令牌重放" | 将针对 Server A 签发的令牌呈交给 Server B 的攻击 |
| `iss` 允许列表 | "受信任的授权服务器" | 受保护资源元数据的 `authorization_servers` 中指定的集合 |
| 密钥轮转 | "轮换 JWKS" | 带重叠窗口的签名密钥定期替换 |
| Public client | "原生或浏览器客户端" | 无 `client_secret` 的 OAuth 客户端；由 PKCE 补偿 |
| `WWW-Authenticate` | "401/403 响应头" | 携带驱动客户端恢复的 `Bearer error=...` 指令 |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 本课实现的 MCP 认证配置文件
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端持有证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 整合后的 OAuth 基础