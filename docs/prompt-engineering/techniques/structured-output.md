# 结构化输出

结构化输出（Structured Output）是指让 LLM 按照预定义的数据格式（JSON、XML、Markdown 表格等）返回结果，使程序能够可靠地解析和使用 LLM 的输出。在构建 AI 应用时，结构化输出是连接 LLM 与业务逻辑的关键技术。

## 为何需要结构化输出

纯自然语言输出对人类友好，但对程序不友好。假设需要从用户描述中提取联系信息：

**自然语言输出**（难以解析）：
```
根据您的描述，这位联系人名叫张伟，他的电话是 138-0000-1234，
邮件地址是 zhangwei@example.com，他在北京工作。
```

**JSON 输出**（程序可直接使用）：
```json
{
  "name": "张伟",
  "phone": "138-0000-1234",
  "email": "zhangwei@example.com",
  "city": "北京"
}
```

结构化输出使 LLM 的结果可以直接传递给数据库、API 或其他下游服务，无需人工解析。

## JSON 输出指令设计

### 基础方法：在 Prompt 中指定格式

最简单的方法是在 Prompt 中明确要求 JSON 输出并给出 Schema：

```
从以下客服对话中提取关键信息，以 JSON 格式返回。

Schema：
{
  "customer_name": "string，客户姓名",
  "issue_type": "billing|technical|shipping|other",
  "priority": "high|medium|low",
  "resolution": "string，解决方案描述",
  "follow_up_required": "boolean"
}

要求：
- 只返回 JSON，不要任何其他文字
- 所有字段必须填写，未知值使用 null
- issue_type 和 priority 必须使用指定的枚举值

对话内容：
[客服对话文本]
```

### 使用示例强化格式

结合 Few-shot 提供输入-输出示例：

```
示例输入："客户李明打来说他的订单三天前发货但一直没收到，很着急。"
示例输出：
{
  "customer_name": "李明",
  "issue_type": "shipping",
  "priority": "high",
  "resolution": null,
  "follow_up_required": true
}

实际输入：[...]
输出：
```

## OpenAI/Anthropic 原生 JSON 模式

### OpenAI JSON Mode

OpenAI API 提供原生 JSON 模式，保证输出是有效的 JSON：

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是数据提取助手，始终以JSON格式输出"},
        {"role": "user", "content": "提取：张伟，138-0000-1234，zhangwei@example.com"}
    ],
    response_format={"type": "json_object"}  # 开启 JSON 模式
)
# 保证输出是可解析的 JSON
import json
data = json.loads(response.choices[0].message.content)
```

### OpenAI Structured Outputs（2024）

更进一步，OpenAI 的 Structured Outputs 功能支持传入 JSON Schema，**保证**输出完全符合 Schema：

```python
from pydantic import BaseModel
from openai import OpenAI

class ContactInfo(BaseModel):
    name: str
    phone: str | None
    email: str | None
    city: str | None

client = OpenAI()
response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "提取联系人信息：张伟..."}],
    response_format=ContactInfo  # 直接传入 Pydantic 模型
)
contact = response.choices[0].message.parsed  # 直接得到 ContactInfo 实例
```

## Pydantic + LLM：Instructor 库

**Instructor** 是在 LLM 之上构建类型安全输出的优秀库，支持 OpenAI、Anthropic、Google 等多个提供商：

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field
from typing import List

# 定义输出数据结构
class ProductReview(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    score: int = Field(ge=1, le=5, description="Score from 1-5")
    key_points: List[str] = Field(description="Main points from review")
    summary: str = Field(max_length=100)

# 创建 Instructor 客户端
client = instructor.from_anthropic(Anthropic())

# 调用时直接得到 Pydantic 对象
review = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "分析这条评论：这款手机摄像头不错，但电池续航太差了，给3分。"
    }],
    response_model=ProductReview,  # 指定期望的输出类型
)

print(review.sentiment)    # "negative"
print(review.score)        # 3
print(review.key_points)   # ["摄像头不错", "电池续航差"]
```

Instructor 内部会自动处理重试逻辑：当 LLM 输出不符合 Schema 时，自动将错误信息反馈给模型并重新生成。

## 输出验证与重试

生产环境中 LLM 输出可能不符合预期格式，需要验证和重试机制：

```python
import json
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def extract_with_retry(text: str) -> dict:
    """带重试的结构化提取"""
    response = call_llm(prompt=f"提取JSON：{text}")

    try:
        data = json.loads(response)
        # 验证必要字段
        assert "name" in data, "缺少 name 字段"
        assert data.get("priority") in ["high", "medium", "low"], "priority 值无效"
        return data
    except (json.JSONDecodeError, AssertionError) as e:
        # 将错误信息附加到重试请求中
        raise ValueError(f"输出格式错误：{e}，原始输出：{response}")
```

**重试策略**：
- 将上次失败的输出和错误信息一同发回给模型："你上次的输出是 `{bad_output}`，错误是 `{error}`，请重新生成符合格式的 JSON"
- 指数退避重试，避免 API 限流
- 设置最大重试次数（通常 3 次），超过后人工介入或使用默认值

## Outlines / lm-format-enforcer：强制格式

对于本地部署的开源模型，可以使用**受约束解码（Constrained Decoding）**技术，在 token 生成阶段就强制保证输出符合格式：

```python
# 使用 Outlines 库（支持本地 HuggingFace 模型）
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

# 定义 Pydantic Schema
from pydantic import BaseModel
class Person(BaseModel):
    name: str
    age: int

# 保证输出严格符合 JSON Schema
generator = outlines.generate.json(model, Person)
result = generator("提取信息：张伟，35岁")
# result 保证是有效的 Person 实例，不会有格式错误
```

受约束解码在生成每个 token 时，只允许那些能构成有效 JSON（或其他格式）的 token，从根本上消除了格式错误。

## 常见问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 输出截断 | max_tokens 不足 | 增加 max_tokens，或简化 Schema |
| 字段遗漏 | 指令不够明确 | 明确说明"所有字段必须填写，未知用 null" |
| 格式前有解释文字 | 模型习惯加前缀 | 使用助手预填 `{` 或 JSON 模式 |
| 枚举值不规范 | 模型创造新值 | 在 Schema 中明确列出所有合法值，加重申 |
| 嵌套结构错误 | 复杂 Schema 难以遵循 | 简化 Schema，或分多步提取 |
| 数字类型错误 | 模型输出字符串而非整数 | 在 Schema 中加类型说明，或后处理转换 |
