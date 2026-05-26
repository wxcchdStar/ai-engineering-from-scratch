# Roots 与 Elicitation — 范围控制与执行中用户输入

> 硬编码路径在用户打开另一个项目时就会失效。预填充的工具参数在用户指定不完整时就会失效。Roots 将服务器限定在用户控制的 URI 集合中；Elicitation 在工具调用中途暂停，通过表单或 URL 向用户请求结构化输入。两个客户端原语，修复两种常见 MCP 失败模式。SEP-1036（URL 模式 elicitation，2025-11-25）在 2026 H1 之前是实验性的 — 依赖它之前请检查 SDK 版本。

**类型：** 构建  
**语言：** Python（stdlib，roots + elicitation 演示）  
**前置条件：** Phase 13 · 07（MCP 服务器）  
**时长：** ～45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 限制服务器文件操作只能在声明的根集合内的 URI 上执行。
- 使用 `elicitation/create` 在工具调用中途向用户请求确认或结构化输入。
- 在表单模式和 URL 模式 elicitation 之间选择（后者是实验性的；已注意漂移风险）。

## 问题

笔记 MCP 服务器在生产环境中遇到的两个具体失败。

**错误的路径假设。** 服务器基于 `~/notes` 编写。另一台机器上笔记在 `~/Documents/Notes` 的用户得到一个静默失败（找不到文件）的工具调用，或者更糟，写到了错误的位置。

**缺失的用户才知道的参数。** 用户问"删除旧的 TPS 报告笔记"。模型调用 `notes_delete(title: "TPS report")`，但有来自 2023、2024 和 2025 年的三条匹配笔记。工具无法猜测。返回"模糊"让人烦恼；对所有三条执行是灾难性的。

Roots 修复第一个：客户端在 `initialize` 时声明服务器可以触及的 URI 集合。Elicitation 修复第二个：服务器暂停工具调用并发送 `elicitation/create` 请求用户选择哪一个。

## 概念

### Roots

客户端在 `initialize` 时声明根列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

服务器随后可调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将 roots 视为边界：根集合之外的任何文件读写都被拒绝。这不由客户端强制执行（服务器仍然是用户信任的代码），但规范兼容的服务器会遵守。

当用户添加或移除根时，客户端发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新其边界。

### 为什么 roots 是客户端原语

Roots 由客户端声明，因为它们代表用户的同意模型。用户告诉 Claude Desktop"给这个笔记服务器访问这两个目录"。服务器不能扩大该范围。

### Elicitation：表单模式（默认）

`elicitation/create` 接收表单 schema 加自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "删除 'TPS report'？多条笔记匹配；选择一条。",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单、收集用户答案、返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的 action：`accept`（用户填写了）、`decline`（用户关闭了）、`cancel`（用户中止了整个工具调用）。

表单 schema 是扁平的 — v1 不支持嵌套对象。SDK 通常拒绝比单层更复杂的任何内容。

### Elicitation：URL 模式（SEP-1036，实验性）

2025-11-25 新增。服务器发送 URL 而非 schema：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "登录 GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开 URL、等待完成、用户返回时回复。适用于 OAuth 流程、支付授权和表单不足的文档签署场景。

漂移风险提示：SEP-1036 的响应格式仍在定型中；某些 SDK 返回回调 URL，其他返回完成令牌。在生产中使用 URL 模式前请阅读你的 SDK 发布说明。

### 何时 elicitation 是正确工具

- 破坏性动作前的用户确认（destructive hint + elicitation）。
- 消歧义（从 N 个匹配中选一个）。
- 首次运行设置（API 密钥、目录、偏好）。
- OAuth 风格的流程（URL 模式）。

### 何时 elicitation 是错误选择

- 填充模型本可以用文字询问的工具必需参数。使用正常的重新提示，而非 elicitation 对话框。
- 高频调用。Elicitation 中断对话；不要在循环内使用。
- 任何服务器事后可以验证的内容。验证、返回错误、让模型用文字问用户。

### 人在回路桥接

Elicitation 加 sampling 共同实现了 MCP 的"人在回路"模型。服务器的 Agent 循环可以暂停等待用户输入（elicitation）或模型推理（sampling）。Phase 13 · 11 涵盖了 sampling；本课涵盖 elicitation。将它们结合起来获得完整的循环中控制。

## 动手用

`code/main.py` 在笔记服务器基础上扩展了：

- `roots/list` 响应，服务器在 root-list-changed 通知后重新查询。
- 一个 `notes_delete` 工具，当多条笔记匹配时使用 `elicitation/create` 消歧义。
- 一个 `notes_setup` 工具，使用 URL 模式 elicitation 打开首次运行配置页（模拟）。
- 边界检查，拒绝在声明的 roots 之外的 URI 上的操作。

演示运行三个场景：正常路径（一个匹配）、消歧义（三个匹配，elicitation 触发）、roots 外写入（被拒绝）。

## 交付物

本课产出 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或消歧义的工具，该技能设计 elicitation 表单 schema 和消息模板。

## 练习

1. 运行 `code/main.py`。触发消歧义路径；确认模拟用户答案被路由回工具。

2. 添加一个每次都需要 elicitation 确认的 `notes_archive` 新工具（destructive hint）。检查 UX：与模型用文字重新询问相比如何？

3. 为首次运行 OAuth 流程实现 URL 模式 elicitation。注意漂移风险并添加 SDK 版本守卫。

4. 扩展 `roots/list` 处理：当通知到达时，服务器应原子性地重新读取并重新扫描可能已超出范围的打开文件句柄。

5. 阅读 GitHub 上的 SEP-1036 issue 讨论帖。找出一个影响服务器如何处理 URL 模式回调的开放问题。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| Root | "同意边界" | 客户端允许服务器触及的 URI |
| `roots/list` | "服务器请求范围" | 客户端返回当前根集合 |
| `notifications/roots/list_changed` | "用户更改了范围" | 客户端通知根集合已变更 |
| Elicitation | "调用中途问用户" | 服务器发起的结构化用户输入请求 |
| `elicitation/create` | "该方法" | elicitation 请求的 JSON-RPC 方法 |
| 表单模式 | "Schema 驱动的表单" | 在客户端 UI 中渲染为表单的扁平 JSON Schema |
| URL 模式 | "浏览器重定向" | SEP-1036 实验性；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | "用户响应结果" | 服务器处理的三个分支 |
| 消歧义 | "选一个" | 工具有 N 个候选时的常见 elicitation 用例 |
| 扁平表单 | "仅顶层属性" | Elicitation schema 不能嵌套 |

## 延伸阅读

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — 权威 roots 参考
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 权威 elicitation 参考
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 新增内容讲解
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式 elicitation 提案（实验性，有漂移风险）
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX 讲解
