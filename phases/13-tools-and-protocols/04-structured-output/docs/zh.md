# 结构化输出——JSON Schema、Pydantic、Zod、受约束解码

> "友好地让模型返回JSON"在前沿模型上有5%到15%的失败率。结构化输出通过受约束解码（constrained decoding）填补了这一差距：模型从字面上被阻止输出违反schema的token。OpenAI的strict模式、Anthropic的schema类型化tool use、Gemini的`responseSchema`、Pydantic AI的`output_type`和Zod的`.parse`是同一思想的五种表面形式。本课构建schema验证器和严格模式契约，学习者将在每个生产提取管道中使用它们。

**类型：** 构建
**语言：** Python（标准库，JSON Schema 2020-12子集）
**前置课程：** Phase 13 · 02（函数调用深入）
**时间：** 约75分钟

## 学习目标

- 使用正确的约束（enum、min/max、required、pattern）为提取目标编写JSON Schema 2020-12。
- 解释为什么严格模式和受约束解码提供了与"生成后验证"不同的保证。
- 区分三种失败模式：解析错误、schema违规、模型拒绝。
- 交付一个带有类型化修复和类型化拒绝处理的提取管道。

## 问题定义

一个读取采购订单邮件的智能体需要将自由文本转换为`{customer, line_items, total_usd}`。三种方法。

**方法一：提示输出JSON。** "以JSON格式回复，包含字段customer、line_items、total_usd。"在前沿模型上85%到95%的时间有效。以六种方式失败：缺少大括号、尾随逗号、错误类型、幻觉字段、在token限制处截断、泄漏散文如"Here is your JSON:"。

**方法二：生成后验证。** 自由生成，解析，根据schema验证，失败时重试。可靠但昂贵——你为每次重试付费，截断bug每次出现都多花一轮。

**方法三：受约束解码。** 提供商在解码时强制schema。无效token从采样分布中屏蔽。输出保证可解析且保证通过验证。失败坍缩为一种模式：拒绝（模型判定输入无法匹配schema）。

2026年每个前沿提供商都提供某种形式的方法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`加上响应中的`refusal`字段（如果模型拒绝）。
- **Anthropic。** 对`tool_use`输入进行schema强制；`stop_reason: "refusal"`不存在，但`end_turn`且无工具调用是信号。
- **Gemini。** 请求级别的`responseSchema`；2026年Gemini为选定类型发布token级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel`输出类型化为`InvoiceModel`的结构化`RunResult`。
- **Zod（TypeScript）。** 运行时解析器，验证提供商输出是否符合Zod schema；与OpenAI的`beta.chat.completions.parse`配对。

共同主线：声明一次schema，端到端强制。

## 核心概念

### JSON Schema 2020-12——通用语言

每个提供商都接受JSON Schema 2020-12。你最常使用的构造：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null`之一。
- `properties`：字段名到子schema的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子schema。
- `additionalProperties`：`false`禁止额外字段（默认因模式而异）。

OpenAI严格模式添加三个要求：每个属性都必须列在`required`中、到处`additionalProperties: false`、无未解析的`$ref`。如果你违反这些，API在请求时返回400。

### Pydantic，Python绑定

Pydantic v2通过`model_json_schema()`从数据类形状的模型生成JSON Schema。Pydantic AI将此包装起来，你只需写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

智能体框架将schema转换为OpenAI strict模式、Anthropic `input_schema`或Gemini `responseSchema`在边缘。模型的输出作为类型化的`Invoice`实例返回。验证错误引发带有类型化错误路径的`ValidationError`。

### Zod，TypeScript绑定

Zod（`z.object({customer: z.string(), ...})`）是TS等价物。OpenAI的Node SDK暴露`zodResponseFormat(Invoice)`，将其翻译为API的JSON Schema载荷。

### 拒绝

严格模式不能强制模型回答。如果输入无法匹配schema（"邮件是一首诗，不是发票"），模型输出一个`refusal`字段，包含原因。你的代码必须将此作为一等结果处理，而非失败。拒绝也可用作安全信号：被要求从受保护内容邮件中提取信用卡号的模型返回附带安全原因的拒绝。

### 开放权重中的受约束解码

开放权重实现使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从schema构建确定性有限自动机；在每一步屏蔽会违反FSM的token的logits。
2. **带JSON解析器的logit屏蔽**：与模型同步运行一个流式JSON解析器；在每一步计算有效下一token集合。
3. **带验证器的推测解码**：廉价草稿模型提议token，验证器强制schema。

商业提供商在幕后选择其中之一。2026年的技术水平：对短结构化输出比普通生成更快，对长输出大致相同速度。

### 三种失败模式

1. **解析错误。** 输出不是有效JSON。在严格模式下不可能发生。在非严格提供商上仍可发生。
2. **Schema违规。** 输出可解析但违反schema。在严格模式下不可能发生。在其外常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你在严格模式之外（Anthropic tool use、非严格OpenAI、较旧Gemini）时，恢复模式是：

```
生成 -> 解析 -> 验证 -> 如果失败，注入错误并重试，最多3次
```

一次重试通常足够。三次重试捕获弱模型的偶发问题。超过三次是schema坏了的信号：模型对某些输入无法满足它，提示或schema需要修复。

### 小模型支持

受约束解码在小模型上有效。一个3B参数的开放模型配合语法强制，在结构化任务上优于70B参数的模型配合原始提示。这是结构化输出对生产重要的主要原因：它将可靠性与模型大小解耦。

## 动手实践

`code/main.py`提供一个最小的标准库JSON Schema 2020-12验证器（types、required、enum、min/max、pattern、items、additionalProperties）。它包装一个`Invoice` schema，并运行假LLM输出通过验证器，演示解析错误、schema违规和拒绝路径。在生产中将假输出替换为任何提供商的真实响应。

需要关注的：

- 验证器返回带有路径和消息的类型化`[ValidationError]`列表。这就是你想暴露给重试提示的形状。
- 拒绝分支不重试。它记录日志并返回类型化拒绝。Phase 14 · 09使用拒绝作为安全信号。
- `additionalProperties: false`检查在对抗性测试输入上触发，展示为什么严格模式关闭了幻觉字段的大门。

## 交付产出

本课程产出`outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、支持工单、简历等），该skill产出一个兼容严格模式的JSON Schema 2020-12和一个镜像它的Pydantic模型，带有类型化拒绝和重试处理的存根。

## 练习

1. 运行`code/main.py`。添加第四个测试用例，其`total_usd`为负数。确认验证器用`minimum`约束路径拒绝它。

2. 扩展验证器以支持带有鉴别器的`oneOf`。常见情况：`line_item`要么是product要么是service，由`kind`标记。严格模式在这里有微妙规则；检查OpenAI的结构化输出指南。

3. 将相同的Invoice schema写为Pydantic BaseModel，并比较`model_json_schema()`输出与你的手写schema。找出Pydantic默认设置但手写版本遗漏的一个字段。

4. 测量拒绝率。构造十个不应该可提取的输入（一首歌词、一个数学证明、一封空邮件），并在真实提供商上以严格模式运行它们。计算拒绝vs幻觉输出。这是你拒绝感知重试的真值。

5. 从头到尾阅读OpenAI的结构化输出指南。找出一个它在严格模式中明确禁止但普通JSON Schema允许的构造。然后设计一个非本质性使用该禁止构造的schema，并将其重构为严格兼容的。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| JSON Schema 2020-12 | "schema规范" | 每个现代提供商使用的IETF草案schema方言 |
| 严格模式 | "保证的schema" | OpenAI标志，通过受约束解码强制schema |
| 受约束解码 | "logit屏蔽" | 解码时强制，屏蔽无效的下一token |
| 拒绝（Refusal） | "模型拒绝" | 输入无法匹配schema时的类型化结果 |
| 解析错误 | "无效JSON" | 输出没有解析为JSON；在严格模式下不可能 |
| Schema违规 | "错误形状" | 已解析但违反types / required / enum / range |
| `additionalProperties: false` | "不允许额外字段" | 禁止未知字段；OpenAI严格模式中必需 |
| Pydantic BaseModel | "类型化输出" | 输出和验证JSON Schema的Python类 |
| Zod schema | "TypeScript输出类型" | 用于提供商输出验证的TS运行时schema |
| 语法强制 | "开放权重受约束解码" | 基于FSM的logit屏蔽，如outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝和schema要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024年8月发布文章，解释解码保证
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 序列化到每个提供商的类型化output_type绑定
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 规范标准
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明和严格模式注意事项
