# 结构化输出 (Structured Outputs)：JSON、Schema 校验、约束解码 (Constrained Decoding)

> 你的 LLM 返回一个字符串。你的应用需要 JSON。这种落差导致的线上事故比任何模型幻觉都多。结构化输出 (Structured Output) 是自然语言与类型化数据之间的桥梁。做对了，你的 LLM 就变成了一个可靠的 API。做错了，你就得凌晨三点用正则表达式解析自由文本。

**类型：** 构建 (Build)
**语言：** Python
**前置要求：** Phase 10, Lessons 01-05（从零构建 LLM）
**时间：** 约 90 分钟
**相关：** Phase 5 · 20（结构化输出与约束解码）涵盖了解码器层面的理论（FSM/CFG logit 处理器、Outlines、XGrammar）。本课聚焦于生产环境 SDK 层面（OpenAI `response_format`、Anthropic tool use、Instructor）——如果你想了解 API 之下的原理，请先阅读 Phase 5 · 20。

## 学习目标

- 使用 OpenAI 和 Anthropic API 参数实现 JSON 模式 (JSON Mode) 和 Schema 约束输出
- 构建一个 Pydantic 校验层，拒绝格式错误的 LLM 输出并以错误反馈进行重试
- 解释约束解码 (Constrained Decoding) 如何在 token 层面强制生成合法 JSON，而无需后处理
- 设计健壮的提取提示词 (Extraction Prompt)，可靠地将非结构化文本转换为类型化数据结构

## 问题所在

你让 LLM："从这段文本中提取产品名称、价格和库存状态。"它回复：

```
The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.
```

这是一个完全正确的回答。但它对你的应用完全没用。你的库存系统需要的是 `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要一个具有特定键 (Key)、特定类型 (Type) 和特定值约束的 JSON 对象。你不需要一个句子。

天真的解决方案：在你的提示词 (Prompt) 中添加"用 JSON 回复"。这在 90% 的情况下有效。另外 10% 的情况下，模型会把 JSON 包裹在 Markdown 代码块中，或者加上"以下是 JSON："这样的前言，或者因为提前闭合了大括号而产生语法错误的 JSON。你的 JSON 解析器崩溃了。你的流水线中断了。你加了 try/except 和重试循环。重试有时会产生不同的数据。现在你在解析问题之上又多了一个一致性问题。

这不是一个提示工程 (Prompt Engineering) 问题。这是一个解码 (Decoding) 问题。模型从左到右生成 token。在每个位置上，它从 10 万多个选项的词汇表中选出最可能的下一个 token。其中大多数选项在给定位置都会产生无效的 JSON。如果模型刚输出了 `{"price":`，下一个 token 必须是数字、引号（用于字符串）、`null`、`true`、`false` 或负号。任何其他 token 都会产生无效的 JSON。没有约束的情况下，模型可能会选择一个完全合理的英文单词，但在语法上却是灾难性的错误。

## 核心概念

### 结构化输出层次谱系

结构化输出控制有四个层次，每个层次都比前一个更可靠。

```mermaid
graph LR
    subgraph Spectrum["Structured Output Spectrum"]
        direction LR
        A["基于提示词\n'返回 JSON'\n~90% 合法"] --> B["JSON 模式\n保证合法 JSON\n不保证 Schema"]
        B --> C["Schema 模式\nJSON + 匹配 Schema\n保证合规"]
        C --> D["约束解码\nToken 级强制\n100% 合规"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**基于提示词**（"用合法 JSON 回复"）：无强制执行。模型通常会遵守，但有时不会。可靠性：约 90%。失败模式：Markdown 代码块、前言文本、输出截断、错误结构。

**JSON 模式 (JSON Mode)**：API 保证输出是合法的 JSON。OpenAI 的 `response_format: { type: "json_object" }` 启用此功能。输出解析不会出错。但可能不符合你期望的 Schema——额外的键、错误的类型、缺失的字段。

**Schema 模式 (Schema Mode)**：API 接受一个 JSON Schema 并保证输出与之匹配。到 2026 年，每个主流提供商都原生支持此功能：OpenAI 的 `response_format: { type: "json_schema", json_schema: {...} }`（也可作为 `tool_choice="required"`），Anthropic 的带 `input_schema` 的 tool use，以及 Gemini 的 `response_schema` + `response_mime_type: "application/json"`。输出会精确地包含你指定的键、类型和约束。

**约束解码 (Constrained Decoding)**：在生成过程中的每个 token 位置，解码器会屏蔽所有会产出无效输出的 token。如果 Schema 要求一个数字而模型即将发出一个字母，该 token 的概率会被设为零。模型只能生成能导向合法输出的 token。这就是 OpenAI 的结构化输出模式以及 Outlines、Guidance 等库底层实现的方式。

### JSON Schema：契约语言

JSON Schema 是你告诉模型（或校验层）输出必须具有什么形状的方式。每个主流的结构化输出系统都使用它。

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

这个 Schema 表示：输出必须是一个对象，包含一个字符串 `product`、一个非负数 `price`、一个布尔值 `in_stock`，以及一个可选的字符串数组 `categories`。任何不匹配的输出都会被拒绝。

Schema 可以处理各种复杂情况：嵌套对象、带类型元素的数组、枚举（将字符串约束到特定值）、模式匹配（字符串的正则表达式）以及组合器（oneOf、anyOf、allOf 用于多态输出）。

### Pydantic 模式

在 Python 中，你不需要手写 JSON Schema。你定义一个 Pydantic 模型，它会自动为你生成 Schema。

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

这会生成与上面相同的 JSON Schema。Instructor 库（以及 OpenAI 的 SDK）可以直接接受 Pydantic 模型：传入模型类，返回一个经过校验的实例。如果 LLM 输出不匹配，Instructor 会自动重试。

### 函数调用 / 工具使用 (Function Calling / Tool Use)

这是解决同一问题的另一种接口。你不是要求模型直接输出 JSON，而是定义带有类型化参数的"工具"（函数）。模型输出一个带有结构化参数的函数调用。OpenAI 称之为"函数调用 (Function Calling)"。Anthropic 称之为"工具使用 (Tool Use)"。结果是相同的：结构化数据。

```mermaid
graph TD
    subgraph ToolUse["Tool Use Flow"]
        U["用户: 从这段评论文本中\n提取产品信息"] --> M["模型处理输入"]
        M --> TC["工具调用:\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["根据函数 Schema\n进行校验"]
        V --> R["结构化结果:\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

当模型需要选择调用哪个函数而不只是填充参数时，tool use 是更好的选择。如果你有 10 个不同的提取 Schema，模型需要根据输入选择正确的一个，tool use 会同时给你 Schema 选择和结构化输出。

### 常见失败模式

即使有 Schema 强制执行，结构化输出也可能以微妙的方式失败。

**幻觉值 (Hallucinated Values)**：输出符合 Schema 但包含编造的数据。当文本说 $348 时，模型输出 `{"price": 299.99}`。Schema 校验无法发现这一点——类型是正确的，值是错误的。

**枚举混淆 (Enum Confusion)**：你将一个字段约束为 `["in_stock", "out_of_stock", "preorder"]`。模型输出 `"available"`——语义上正确，但不在允许的集合中。良好的约束解码可以防止这种情况。基于提示词的方法则不行。

**嵌套对象深度 (Nested Object Depth)**：深度嵌套的 Schema（4 层以上）会产生更多错误。每多一层嵌套，模型就多一个可能丢失结构跟踪的地方。

**数组长度 (Array Length)**：模型可能生成过多或过少的数组元素。Schema 支持 `minItems` 和 `maxItems`，但并非所有提供商都在解码层面强制执行它们。

**可选字段遗漏 (Optional Field Omission)**：模型遗漏了在技术上可选但对你用例而言语义上重要的字段。即使在数据有时缺失的情况下，也应在 Schema 中将它们设为必需字段——强制模型显式地输出 `null`。

## 动手构建

### 步骤 1：JSON Schema 校验器

从头构建一个校验器，检查一个 Python 对象是否匹配 JSON Schema。这是在输出端运行以确保合规性的组件。

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### 步骤 2：Pydantic 风格模型转 Schema

构建一个最小的类到 Schema 的转换器。定义一个 Python 类并自动生成其 JSON Schema。

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### 步骤 3：约束 Token 过滤器

模拟约束解码。给定一个部分 JSON 字符串和一个 Schema，确定当前位置哪些 token 类别是合法的。

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### 步骤 4：提取流水线

将所有组件组合成一个提取流水线：定义 Schema，模拟 LLM 产出结构化输出，校验输出，并处理重试。

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### 步骤 5：运行完整流水线

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## 使用方法

### OpenAI 结构化输出 (Structured Outputs)

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

OpenAI 的结构化输出模式在内部使用了约束解码 (constrained decoding)。模型生成的每一个 token 都保证能产生与 Pydantic schema 匹配的输出。无需重试，无需验证。约束已经内嵌到解码过程中。

### Anthropic 工具调用 (Tool Use)

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

Anthropic 通过工具调用实现结构化输出。模型发出一个工具调用，其中包含与 input_schema 匹配的结构化参数。相同的结果，不同的 API 接口。

### Instructor 库

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

Instructor 封装了任意 LLM 客户端，并添加了自动重试和验证功能。如果第一次尝试未通过验证，它会将错误信息作为上下文发送回模型，要求其修正输出。这适用于任何服务商，而不仅仅是 OpenAI。

## 产出物

本课产出 `outputs/prompt-structured-extractor.md`——一个可复用的提示词模板，能够根据给定的 schema 定义从任意文本中提取结构化数据。向其输入一个 JSON Schema 和非结构化文本，它就能返回经过验证的 JSON。

同时产出 `outputs/skill-structured-outputs.md`——一个决策框架，用于根据你所使用的服务商、可靠性需求和 schema 复杂度，选择正确的结构化输出策略。

## 练习

1. 扩展 schema 验证器以支持 `oneOf`（数据必须精确匹配多个 schema 中的一个）。这可以处理多态输出——例如，一个字段既可以是 `Product` 对象，也可以是 `Service` 对象，两者具有不同的结构。

2. 构建一个"schema 差异对比"工具，用于比较两个 schema 并识别破坏性变更（移除了必填字段、更改了类型）和非破坏性变更（新增了可选字段、放宽了约束）。这对于在生产环境中对提取 schema 进行版本管理至关重要。

3. 实现一个更贴近实际的约束解码模拟器。给定一个 JSON Schema 和一个包含 100 个 token 的词汇表（字母、数字、标点符号、关键字），逐步遍历生成过程，在每一步屏蔽无效的 token。测量每一步中有效 token 占词汇表的百分比。

4. 构建一个提取评估套件。创建 50 条带有手工标注 JSON 输出的产品描述。在这 50 条数据上运行你的提取流水线，衡量精确匹配率、字段级准确率和类型合规性。找出哪些字段最难被正确提取。

5. 为你的提取流水线添加"置信度分数"。对于每个提取出的字段，估计模型的置信度（基于 token 概率，或者通过运行提取 3 次并衡量一致性）。标记低置信度字段以供人工审核。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| JSON 模式 (JSON mode) | "返回 JSON" | API 标志，保证输出为语法上有效的 JSON，但不强制执行任何特定的 schema |
| 结构化输出 (Structured output) | "带类型的 JSON" | 匹配特定 JSON Schema 的输出，具有正确的键名、类型和约束 |
| 约束解码 (Constrained decoding) | "引导式生成" | 在每个 token 位置，屏蔽会生成无效输出的 token——保证 100% 的 schema 合规性 |
| JSON Schema | "一个 JSON 模板" | 一种声明式语言，用于描述 JSON 数据的结构、类型和约束（被 OpenAPI、JSON Forms 等广泛使用） |
| Pydantic | "Python dataclass 的增强版" | 一个 Python 库，通过类型验证定义数据模型，被 FastAPI 和 Instructor 用于生成 JSON Schema |
| 函数调用 (Function calling) | "工具调用 (Tool use)" | LLM 输出一个结构化的函数调用（函数名 + 带类型的参数），而不是自由文本——OpenAI 和 Anthropic 都支持这一点 |
| Instructor | "面向 LLM 的 Pydantic" | 一个 Python 库，封装 LLM 客户端以返回经过验证的 Pydantic 实例，并在验证失败时自动重试 |
| Token 屏蔽 (Token masking) | "过滤词汇表" | 在生成过程中将特定 token 的概率设为零，使模型无法生成这些 token |
| Schema 合规性 (Schema compliance) | "匹配结构" | 输出包含所有必填字段、正确的类型、值在约束范围内，且没有额外的非法字段 |
| 重试循环 (Retry loop) | "反复尝试直到成功" | 将验证错误发送回模型，要求其修正输出——Instructor 会自动执行此操作，最多可配置重试次数 |

## 延伸阅读

- [OpenAI 结构化输出指南](https://platform.openai.com/docs/guides/structured-outputs)——OpenAI API 中基于 JSON Schema 的约束解码官方文档
- [Willard & Louf, 2023——"Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702)——Outlines 论文，描述了如何将 JSON Schema 编译为有限状态机以实现 token 级别的约束
- [Instructor 文档](https://python.useinstructor.com/)——从任意 LLM 获取结构化输出的标准库，支持 Pydantic 验证和重试
- [Anthropic 工具调用指南](https://docs.anthropic.com/en/docs/tool-use)——Claude 如何通过带有 JSON Schema input_schema 的工具调用来实现结构化输出
- [JSON Schema 规范](https://json-schema.org/)——所有主流结构化输出系统所使用的 schema 语言的完整规范
- [Outlines 库](https://github.com/outlines-dev/outlines)——使用正则表达式和 JSON Schema 编译为有限状态机的开源约束生成工具
- [Dong 等，"XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100)——当前最先进的语法引擎；基于下推自动机编译，以约 100 纳秒/token 的速度屏蔽 token。
- [Beurer-Kellner 等，"Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094)——LMQL 论文，将约束解码构建为一种具有类型和值约束的查询语言。
- [Microsoft Guidance（框架文档）](https://github.com/guidance-ai/guidance)——模板驱动的约束生成；与 Outlines 和 XGrammar 互补的、不绑定特定服务商的方案。
