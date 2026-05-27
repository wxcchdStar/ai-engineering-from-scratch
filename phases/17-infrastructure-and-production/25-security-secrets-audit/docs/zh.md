# 安全 — 密钥、API 密钥轮换、审计日志、Guardrail

> 通过集中式 vault（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥蔓延（secret sprawl）。永远不要将凭证存储在配置文件、VCS（版本控制系统，Version Control System）中的 env 文件或电子表格中。使用 IAM（身份与访问管理，Identity and Access Management）角色而非静态密钥；CI/CD 使用 OIDC（OpenID Connect）。AI 网关模式是 2026 年的解决方案：应用 → 网关 → 模型供应商，网关在运行时从 vault 拉取凭证。在 vault 中轮换，所有应用在几分钟内获取新密钥 — 无需重新部署，无需 Slack 上问"谁有新密钥"。轮换策略 ≤ 90 天；在每次提交时使用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任（Zero-trust）：MFA（多因素认证，Multi-Factor Authentication）、SSO（单点登录，Single Sign-On）、RBAC（基于角色的访问控制，Role-Based Access Control）/ABAC（基于属性的访问控制，Attribute-Based Access Control）、短期 token、设备姿态。PII（个人身份信息，Personally Identifiable Information）脱敏使用实体识别在转发前掩码 PHI（受保护的健康信息，Protected Health Information）/PII；一致性 tokenization（Mesh 方法）将敏感值映射到稳定的占位符，以便 LLM 保留代码/关系语义。网络出口：LLM 服务在专用 VPC（虚拟私有云，Virtual Private Cloud）/VNet 子网中，仅白名单 `api.openai.com`、`api.anthropic.com` 等；阻止所有其他出站流量。2026 年的事件驱动因素：Vercel 供应链攻击通过被入侵的 CI/CD 凭证泄露了数千个客户部署中的环境变量。

**类型：** 学习
**语言：** Python（标准库，PII 脱敏器 + 审计日志写入器）
**前置要求：** 第 17 阶段 · 19（AI 网关）、第 17 阶段 · 13（可观测性）
**时间：** 约 60 分钟

## 学习目标

- 列举四种密钥管理反模式（VCS 中的配置文件、硬编码 env、电子表格、静态密钥）并说出它们的替代方案。
- 解释 AI 网关从 vault 拉取模式作为 2026 年生产标准。
- 实现一个具有一致性 tokenization 的 PII 脱敏器（相同值 → 相同占位符），以便语义得以保留。
- 说出 2026 年 Vercel 供应链事件及其对 CI/CD 凭证卫生的教训。

## 问题

一个实习生提交了带有 API 密钥的 `.env`。他们很快删除了它。密钥已经在 git 历史中 — GitGuardian 扫描捕获了它，你的轮换过程是"在 Slack 上通知团队，更新 40 个配置文件，重新部署所有服务。"8 小时后，一半的服务在线，一半在等待部署窗口。

另外，用户提示词包含"我的 SSN 是 123-45-6789。"提示词发送到 OpenAI。你有 BAA（业务伙伴协议，Business Associate Agreement），但你的内部策略是在转发前掩码 PII。你没有。

另外，你的 EKS 集群的 LLM Pod 可以访问任何互联网主机。有人通过 DNS 查找将数据泄露到攻击者控制的域。没有任何东西阻止它。

LLM 服务的安全必须解决所有三个向量。Vault 支持的凭证。PII 脱敏。网络出口过滤。审计日志。

## 概念

### 集中式 vault + IAM 角色拉取

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。一个真实来源。

**IAM 角色**：应用/网关通过其 IAM 身份认证，而非静态密钥。Vault 在 token 的生命周期内返回密钥。

**AI 网关模式**：网关在请求时从 vault 拉取 `OPENAI_API_KEY`。在 vault 中轮换；下一个请求获取新密钥。无需重新部署。

### 轮换策略 ≤ 90 天

所有 API 密钥、vault 根 token、CI/CD 凭证。尽可能自动轮换。手动轮换记录并跟踪。

### 密钥扫描

- **TruffleHog** — 对提交进行正则 + 熵检测。
- **GitGuardian** — 商业，高准确率。
- **Gitleaks** — 开源，在 CI 中运行。

在每次提交时运行。如果检测到新密钥，阻止 PR。

### 零信任姿态

- 所有账户要求 MFA。
- 通过 SAML/OIDC 进行 SSO。
- RBAC（基于角色）或 ABAC（基于属性）用于细粒度访问。
- 短期 token（小时，而非天）。
- 设备姿态 — 仅具有磁盘加密的公司设备。

### PII / PHI 脱敏

在提示词离开你的基础设施之前：

1. 实体识别（spaCy NER（命名实体识别，Named Entity Recognition）、Presidio、商业）。
2. 掩码匹配的实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致性 tokenization（Mesh 方法）：相同值映射到相同的占位符，以便 LLM 保留关系。
4. 可选的 LLM 响应反向映射。

静态正则过滤器捕获基本模式；NER 捕获更多。两者都使用。

### 输入 + 输出 guardrail

输入：阻止已知越狱、禁止主题；按用户限流。

输出：对泄露的密钥进行正则清洗（API 密钥模式、拒绝上下文中的电子邮件模式），对策略违规进行分类器检测。

### 网络出口白名单

LLM 服务在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、vault 端点。
- 其他所有：丢弃。
- 通过仅允许列表解析器进行 DNS（避免 DNS 隧道泄露）。

### 审计日志

每次 LLM 调用的不可变日志，包含：
- 时间戳。
- 用户/租户。
- 提示词哈希（出于隐私考虑，而非原始提示词）。
- 模型 + 版本。
- Token 数量。
- 成本。
- 响应哈希。
- 任何 guardrail 触发。

根据监管要求保留（SOC 2 1 年，HIPAA 6 年）。

### 2026 年 Vercel 事件

供应链攻击：被入侵的 CI/CD 凭证泄露了数千个客户部署中的环境变量。教训：CI/CD 凭证等同于生产凭证。存储在 vault 中。窄范围授权。积极轮换。

### 你应该记住的数字

- 轮换策略：≤ 90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭证被入侵 → 数千个客户环境变量泄露。
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

## 动手实践

`code/main.py` 实现了一个具有一致性 tokenization 和仅追加审计日志的玩具 PII 脱敏器。

## 交付物

本课产出 `outputs/skill-llm-security-plan.md`。给定监管范围当前状态，规划 vault 迁移、脱敏器、出口、审计日志。

## 练习

1. 运行 `code/main.py`。发送两个引用相同 SSN 的提示词。确认两者获得相同的占位符。
2. 为调用 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS 部署设计网络出口策略。
3. 你在 git 历史中发现一个密钥（2 年前的）。正确的响应是什么 — 轮换密钥、清理历史，还是两者？说明理由。
4. 你的审计日志每天增长 10 GB。设计保留层级（热 30 天、温 12 个月、冷 6 年）。
5. 论证反向 tokenization（将真实值替换回 LLM 响应）是否值得其复杂性，还是保持占位符可见。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Vault | "密钥存储" | 集中式凭证管理服务 |
| IAM 角色 | "基于身份的认证" | 由应用承担的角色；返回短期凭证 |
| CI/CD 的 OIDC | "云颁发 token" | CI 中无静态密钥 — 通过 OIDC 获取身份 |
| TruffleHog / GitGuardian / Gitleaks | "密钥扫描器" | 提交时密钥检测 |
| RBAC / ABAC | "访问控制" | 基于角色 vs 基于属性 |
| PII 脱敏（PII scrubbing） | "数据掩码" | 移除或 tokenize 敏感实体 |
| 一致性 tokenization（Consistent tokenization） | "稳定占位符" | 相同值 → 每次相同 token |
| Mesh 方法（Mesh approach） | "Mesh tokenization" | 保留语义的 tokenization 模式 |
| 出口白名单（Egress whitelist） | "出站允许列表" | 仅允许访问的域可达 |
| 审计日志（Audit log） | "不可变历史" | 仅追加记录，用于合规 |
