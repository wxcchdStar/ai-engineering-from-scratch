# 函数调用深入——OpenAI、Anthropic、Gemini

> 三个前沿提供商在2024年就工具调用循环达成了相同的共识，然后在其他一切上产生了分歧。OpenAI使用`tools`和`tool_calls`。Anthropic使用`tool_use`和`tool_result`块。Gemini使用`functionDeclarations`和唯一id关联。本课逐一对比三者，使在一个提供商上发布的代码在移植时不会出错。

**类型：** 构建
**语言：** Python（标准库，schema转换器）
**前置课程：** Phase 13 · 01（工具接口）
**时间：** 约75分钟

## 学习目标

- 说明OpenAI、Anthropic和Gemini函数调用载荷之间的三个形状差异（声明、调用、结果）。
- 将一个工具声明翻译为所有三个提供商格式，并预测严格模式约束在哪里不同。
- 在每个提供商中使用`tool_choice`来强制、禁止或自动选择工具调用。
- 了解每个提供商的硬限制（工具数量、schema深度、参数长度）以及违反限制时各自发出的错误签名。

## 问题定义

函数调用请求的形状因提供商而异。以下是2026年生产栈的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传递`tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含`choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中`arguments`是你必须解析的JSON字符串。严格模式（`strict: true`）通过受约束解码强制schema合规。

**Anthropic Messages API。** 你传递`tools: [{name, description, input_schema}]`。响应以`content: [{type: "text"}, {type: "tool_use", id, name, input}]`的形式返回。`input`已经是解析后的对象（不是字符串）。你用一个包含`{type: "tool_result", tool_use_id, content}`块的新`user`消息回复。

**Google Gemini API。** 你传递`tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在`functionDeclarations`下）。响应以`candidates[0].content.parts: [{functionCall: {name, args, id}}]`到达，其中`id`在Gemini 3及以上版本中是唯一的，用于并行调用关联。你用`{functionResponse: {name, id, response}}`回复。

相同的循环。不同的字段名、不同的嵌套、不同的字符串vs对象约定、不同的关联机制。一个在OpenAI上编写天气智能体的团队，移植到Anthropic需要两天的管道工作，再移植到Gemini又要一天。

本课构建一个转换器，将三种格式统一为一个规范化的工具声明，并在边缘进行路由。Phase 13 · 17将相同的模式泛化为LLM网关。

## 核心概念

### 公共结构

每个提供商需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入schema。
2. **工具选择。** 强制特定工具、禁止工具或让模型决定。
3. **调用输出。** 结构化输出，命名工具和参数。
4. **调用id。** 将响应与正确的调用关联（对并行很重要）。
5. **结果注入。** 将结果绑定回调用的消息或块。

### 形状差异，逐字段对比

| 方面 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | assistant消息上的`tool_calls[]` | 类型为`tool_use`的`content[]` | 类型为`functionCall`的`parts[]` |
| 参数类型 | 字符串化JSON | 已解析对象 | 已解析对象 |
| Id格式 | `call_...`（OpenAI生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色`tool`，`tool_call_id` | `user`中带`tool_result`，`tool_use_id` | `functionResponse`带匹配`id` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格schema | `strict: true` | schema即schema（始终强制） | 请求级别的`responseSchema` |

### 你会实际遇到的限制

- **OpenAI。** 每个请求128个工具。Schema深度5层。参数字符串 <= 8192字节。严格模式要求：无`$ref`、无重叠的`oneOf`/`anyOf`/`allOf`、每个属性都列在`required`中。
- **Anthropic。** 每个请求64个工具。Schema深度实际上无界但实用限制为10。无严格模式标志；schema是契约，模型倾向于遵守。
- **Gemini。** 每个请求64个函数。Schema类型是OpenAPI 3.0子集（与JSON Schema 2020-12有轻微分歧）。Gemini 3起并行调用有唯一id。

### `tool_choice`行为

每个人都支持三种模式，命名不同。

- **Auto。** 模型选择工具或文本。默认。
- **Required / Any。** 模型必须调用至少一个工具。
- **None。** 模型不得调用工具。

加上每个提供商特有的一种模式：

- **OpenAI。** 按名称强制特定工具。
- **Anthropic。** 按名称强制特定工具；`disable_parallel_tool_use`标志分离单个vs多个。
- **Gemini。** `mode: "VALIDATED"`将每个响应路由通过schema验证器，无论模型意图如何。

### 并行调用

OpenAI的`parallel_tool_calls: true`（默认）在一个assistant消息中输出多个调用。你运行所有调用并用批量tool角色消息回复，每个`tool_call_id`一个条目。Anthropic历史上是单调用；`disable_parallel_tool_use: false`（从Claude 3.5起默认）启用多调用。Gemini 2允许并行调用但没有稳定id；Gemini 3添加UUID使乱序响应能干净关联。

### 流式传输

三者都支持流式工具调用。线路格式不同：

- **OpenAI。** `tool_calls[i].function.arguments`的delta块增量到达。你累积直到`finish_reason: "tool_calls"`。
- **Anthropic。** block-start / block-delta / block-stop事件。`input_json_delta`块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3新增）输出带`functionCallId`的块，使多个并行调用可以干净地交错。

Phase 13 · 03深入讲解并行+流式重组。本课聚焦于声明和单调用形状。

### 错误与修复

无效参数错误看起来也不同。

- **OpenAI（非严格）。** 模型返回`arguments: "{bad json}"`，你的JSON解析失败，注入错误消息并重新调用。
- **OpenAI（严格）。** 验证在解码时发生；无效JSON不可能出现，但`refusal`可以出现。
- **Anthropic。** `input`可能包含意外字段；schema是建议性的。服务端验证。
- **Gemini。** OpenAPI 3.0怪癖：对象字段上的`enum`被静默忽略；自己验证。

### 转换器模式

你代码中的规范化工具声明看起来像这样（你选择形状）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其翻译为三种提供商形状。`code/main.py`中的工具正是这样做的，然后通过每个提供商的响应形状往返一个假工具调用。不需要网络——本课教的是形状，不是HTTP。

生产团队将此转换器包装在`AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或`BaseTool`（LlamaIndex）中。Phase 13 · 17发布一个网关，在三者中任何一个前面暴露OpenAI形状的API。

## 动手实践

`code/main.py`定义一个规范化的`Tool`数据类和三个转换器，分别输出OpenAI、Anthropic和Gemini的声明JSON。然后它将每种形状的手工制作的提供商响应解析为相同的规范化调用对象，证明语义在表象之下是相同的。运行它并逐一对比三个声明。

需要关注的：

- 三个声明块仅在信封和字段名上不同。
- 三个响应块在调用所在位置上不同（顶层`tool_calls`、`content[]`块、`parts[]`条目）。
- 一个`canonical_call()`函数从所有三种响应形状中提取`{id, name, args}`。

## 交付产出

本课程产出`outputs/skill-provider-portability-audit.md`。给定一个针对某个提供商的函数调用集成，该skill产出可移植性审计：它依赖了哪些提供商限制、哪些字段需要重命名、以及移植到其他每个提供商时什么会崩溃。

## 练习

1. 运行`code/main.py`并验证三个提供商的声明JSON都序列化了相同的底层`Tool`对象。修改规范化工具以添加一个enum参数，确认只有Gemini转换器需要处理OpenAPI怪癖。

2. 为每个提供商添加一个`ListToolsResponse`解析器，提取模型在`list_tools`或发现调用后返回的工具列表。OpenAI原生没有这个；注意这种不对称性。

3. 实现`tool_choice`转换：将规范化的`ToolChoice(mode="force", tool_name="x")`映射到所有三种提供商形状。然后映射`mode="any"`和`mode="none"`。检查课程的差异表。

4. 选择三个提供商之一，从头到尾阅读其函数调用指南。找到一个它的schema规格中其他两个不支持的字段。候选项：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明schema的工具调用。通过每个提供商的验证器（第01课中的标准库验证器可以作为代理）运行它，记录哪些错误触发。记录你在生产中会使用哪个提供商来保证严格性。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 函数调用 | "工具使用" | 用于结构化工具调用输出的提供商级API |
| 工具声明 | "工具规格" | 名称 + 描述 + JSON Schema输入载荷 |
| `tool_choice` | "强制/禁止" | Auto / required / none / 特定名称模式 |
| 严格模式 | "Schema强制" | OpenAI标志，通过受约束解码匹配schema |
| `tool_use`块 | "Anthropic的调用形状" | 带有id、name、input的内联内容块 |
| `functionCall` part | "Gemini的调用形状" | 包含name、args和id的`parts[]`条目 |
| 参数即字符串 | "字符串化JSON" | OpenAI将args作为JSON字符串返回，而非对象 |
| 并行工具调用 | "一轮扇出" | 一个assistant消息中的多个工具调用 |
| 拒绝（Refusal） | "模型拒绝" | 仅限严格模式的拒绝块代替调用 |
| OpenAPI 3.0子集 | "Gemini schema怪癖" | Gemini使用与JSON-Schema类似但有细微差异的方言 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 包括严格模式和并行调用的权威参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use`和`tool_result`块语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一id和OpenAPI子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini的企业接口
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式schema强制细节
