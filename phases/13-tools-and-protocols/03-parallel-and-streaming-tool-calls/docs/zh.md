# 并行工具调用与流式工具使用

> 三个独立的天气查询串行执行是三次往返。并行运行后总时间坍缩为最慢的单次调用。现在每个前沿提供商都在单轮中输出多个工具调用。收益是真实的；管道是微妙的。本课讲解两个部分：并行扇出和流式参数重组，重点关注id关联陷阱。

**类型：** 构建
**语言：** Python（标准库，线程池 + 流式工具）
**前置课程：** Phase 13 · 02（函数调用深入）
**时间：** 约75分钟

## 学习目标

- 解释`parallel_tool_calls: true`为何存在以及何时禁用它。
- 在并行扇出期间将流式参数块关联到正确的工具调用id。
- 将部分`arguments`字符串重组为完整JSON而不提前解析。
- 运行三城市天气基准测试，演示顺序vs并行延迟。

## 问题定义

没有并行调用时，一个回答"班加罗尔、东京和苏黎世的天气怎么样"的智能体会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> 运行执行器，返回结果
LLM -> call get_weather(Tokyo)
host -> 运行执行器，返回结果
LLM -> call get_weather(Zurich)
host -> 运行执行器，返回结果
LLM -> 最终文本回答
```

三次LLM往返，每次还要加上执行器延迟。大约是理想挂钟时间的4倍。

使用并行调用：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> 并发运行三个执行器，返回三个结果
LLM -> 最终文本回答
```

一次LLM往返。执行器时间是三者的最大值，而非总和。在OpenAI、Anthropic和Gemini上的生产基准测试显示，扇出工作负载的挂钟时间减少60%到70%。

代价是关联复杂性。当三个调用以乱序完成时，你的结果必须携带匹配的`tool_call_id`，以便模型对齐它们。当结果流式传输时，你必须将部分参数片段组装成完整的JSON后再执行。Gemini 3添加唯一id部分是为了解决一个实际问题：两个对同一工具的并行调用无法区分。

## 核心概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true`默认开启。设为`false`强制串行。
- **Anthropic。** 通过`disable_parallel_tool_use: false`（Claude 3.5及以上默认）启用并行。设为`true`改为串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"`让模型决定。

在以下情况下禁用并行：工具有排序依赖（`create_file`然后`write_file`）、一个调用的输出是另一个调用的输入、或速率限制器无法处理扇出。

### Id关联

模型输出的每个调用都有一个`id`。宿主返回的每个结果必须包含相同的id。没有这个，结果就是模糊的。

- **OpenAI。** 每个tool角色消息上的`tool_call_id`。
- **Anthropic。** 每个`tool_result`块上的`tool_use_id`。
- **Gemini。** 每个`functionResponse`上的`id`（Gemini 3及以上；Gemini 2按名称匹配，这对同名并行调用会出错）。

### 并发运行调用

宿主在各自的线程、协程或远程worker上运行每个调用的执行器。最简单的工具使用线程池；生产环境使用asyncio配合`asyncio.gather`或结构化并发。完成顺序不可预测——id是标识符。

一个常见bug：以调用列表顺序而非完成顺序回复结果。这通常有效，因为模型只关心`tool_call_id`，但如果一个结果被丢弃或重复，乱序提交使调试更困难。优先以完成顺序带显式id回复。

### 流式工具调用

当模型流式输出时，`arguments`分片到达。三个并行调用的三个独立块流在线路上交错。你需要每个id一个累加器。

各提供商的形状：

- **OpenAI。** 每个块是`choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带`index`（调用列表中的位置）。你按index累积，在id首次出现时读取，在`finish_reason = "tool_calls"`时解析JSON。
- **Anthropic。** 流事件为`message_start`，然后每个块一个`content_block_start`，类型为`tool_use`（包含id、name、空input）。`content_block_delta`事件携带`input_json_delta`块。`content_block_stop`关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3新增）输出带`functionCallId`的块，使调用可以干净地交错。在Gemini 3之前，流式一次返回一个完整调用。

### 部分JSON与提前解析陷阱

在参数完成之前你不能解析`arguments`。部分JSON如`{"city": "Beng`是无效的，会抛出异常。正确的门控是提供商的调用结束信号：OpenAI的`finish_reason = "tool_calls"`、Anthropic的`content_block_stop`或Gemini的流结束事件。只有那时才尝试`json.loads`。更健壮的方法使用增量JSON解析器，在结构完成时产出事件；OpenAI的流式指南为显示实时"思考中"指示器的UX推荐这种方法。大括号计数作为完整性测试不可靠（引号字符串或转义内容中的大括号导致误报），应仅作为非正式调试启发式使用。

### 乱序完成

```
call_A: 快速API，最先返回
call_B: 慢速API，第二返回
call_C: 中等API，第三返回
```

宿主回复仍必须引用id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在OpenAI或Anthropic上，回复中的顺序对正确性无影响。Gemini接受任何顺序，只要id匹配。

### 基准测试：顺序vs并行

`code/main.py`中的工具模拟三个具有400、600和800毫秒延迟的执行器。顺序运行总计1800毫秒。并行运行为max(400, 600, 800) = 800毫秒。差异是恒定的而非成比例的，因此节省随工具数量增长。

实际注意事项：并行调用会对下游API产生压力。对限速服务的10路扇出会失败。Phase 13 · 17覆盖网关级背压；重试语义计划在未来阶段。

### 流式扇出挂钟时间

如果模型本身是流式的，你可以在一个调用的参数完成后立即开始执行，而不是等待所有调用完成。这是OpenAI文档记录的优化，但并非所有SDK都暴露。本课的工具实现了它：一旦模拟流产出一个完整的参数对象，宿主就启动该调用。

## 动手实践

`code/main.py`有两个部分。第一部分使用`concurrent.futures.ThreadPoolExecutor`顺序和并行运行三个模拟天气调用，并打印挂钟时间。第二部分重放一个假流式响应——三个并行调用的`arguments`块交错在一个流上——并使用`StreamAccumulator`按id重组它们。没有LLM，没有网络，只有重组逻辑。

需要关注的：

- 顺序计时器达到1.8秒。并行计时器在相同的假延迟下达到0.8秒。
- 累加器通过按id缓冲并仅在每个调用的JSON完成时解析来处理乱序到达的块。
- 执行器在一个id的参数完成后立即启动，而不是在所有流结束后。

## 交付产出

本课程产出`outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该skill审计哪些工具可以安全并行化、哪些有排序依赖、以及哪些会淹没下游速率限制——返回带有每个工具`parallel_safe`标志的修订注册表。

## 练习

1. 运行`code/main.py`并改变模拟延迟。确认并行与顺序的比率大约为`max/sum`（实际运行与理想有轻微偏差，因为线程调度、序列化和工具开销）。在什么延迟分布下并行不再重要？

2. 扩展累加器以处理"调用在流中途被取消"的情况，丢弃其缓冲区并输出`cancelled`事件。哪个提供商明确记录了这种情况？检查Anthropic的`content_block_stop`语义和OpenAI的`finish_reason: "length"`行为。

3. 将线程池替换为`asyncio.gather`。对两者做基准测试。你应该看到async的小幅胜利，因为上下文切换成本更低，但仅在执行器做真实I/O时。

4. 选择两个不应并行化的工具（例如`create_file`然后`write_file`）。在注册表中添加`ordering_dependency`图，并根据该图门控并行扇出。这是依赖感知调度的最小机制，未来的智能体工程阶段会正式化。

5. 阅读OpenAI的并行函数调用部分和Anthropic的`disable_parallel_tool_use`文档。找出Anthropic建议禁用并行的一种真实世界工具类型。（提示：对同一资源的有后果突变。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 并行工具调用 | "一轮扇出" | 模型在单个assistant消息中输出多个工具调用 |
| `parallel_tool_calls` | "OpenAI的标志" | 启用或禁用多调用输出 |
| `disable_parallel_tool_use` | "Anthropic的反向标志" | 退出标志；默认启用并行 |
| 工具调用id | "关联句柄" | 结果消息必须回显的每调用标识符 |
| 累加器 | "流缓冲区" | 用于部分`arguments`块的每id字符串缓冲区 |
| 乱序完成 | "最快先到" | 并行调用以不可预测的顺序完成；id是粘合剂 |
| 依赖图 | "排序约束" | 输出为其他工具输入的工具；不能并行化 |
| 提前解析陷阱 | "JSON.parse爆炸" | 尝试解析不完整的`arguments`字符串 |
| `streamFunctionCallArguments` | "Gemini 3特性" | 带有每调用唯一id的流式参数块 |
| 完成顺序回复 | "不等全部" | 结果到达时就按id回复 |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和退出标志
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use`和结果批处理
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3的id关联并行调用
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI流的分块参数重组
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — 带`input_json_delta`的`content_block_delta`
