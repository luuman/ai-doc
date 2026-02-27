# Anthropic API 使用指南

Anthropic 提供的 Claude 系列模型以安全性和长上下文能力见长。本指南覆盖从账号配置到高级功能的完整使用流程。

## 账号与密钥管理

前往 [console.anthropic.com](https://console.anthropic.com) 注册账号并创建 API Key。密钥格式为 `sk-ant-api03-...`。

最佳实践：
- 将密钥存入环境变量 `ANTHROPIC_API_KEY`，不要硬编码在代码中
- 为不同环境（开发/生产）使用不同密钥
- 定期轮换密钥，删除不再使用的旧密钥
- 通过 Workspaces 功能按团队或项目隔离用量和密钥

## Messages API 请求结构

核心参数：

```python
import anthropic

client = anthropic.Anthropic()  # 自动读取 ANTHROPIC_API_KEY

message = client.messages.create(
    model="claude-opus-4-5",          # 模型名称
    max_tokens=1024,                  # 最大输出 Token 数（必填）
    system="你是一个资深 Python 工程师。",  # 系统提示
    messages=[
        {"role": "user", "content": "如何优化 Python 的内存使用？"}
    ]
)

print(message.content[0].text)
```

关键参数说明：
- `model`：指定模型版本，建议使用完整版本号（如 `claude-opus-4-5`）确保行为稳定
- `max_tokens`：必填，设置输出上限；不会影响实际收费，但会截断超长输出
- `temperature`：默认 1.0，创意写作可适当提高，精确任务建议 0.0-0.3
- `system`：系统提示，与 `messages` 分开传递（区别于 OpenAI 的 system role）
- `stop_sequences`：遇到指定字符串时停止生成

## 流式输出

对于长文本生成，流式输出可以显著降低用户感知延迟：

```python
with client.messages.stream(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "写一篇关于量子计算的文章"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# 获取完整的最终 Message 对象
final_message = stream.get_final_message()
print(f"\n总 Token 用量：{final_message.usage}")
```

## 工具调用（Tool Use）

工具调用允许模型决定何时调用哪个函数：

```python
tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的当前天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
]

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}]
)

# 检查是否需要调用工具
if response.stop_reason == "tool_use":
    tool_use = next(b for b in response.content if b.type == "tool_use")
    tool_result = execute_tool(tool_use.name, tool_use.input)  # 自行实现

    # 将工具结果回传
    messages = [
        {"role": "user", "content": "北京今天天气怎么样？"},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_use.id, "content": tool_result}
        ]}
    ]
    final = client.messages.create(model="claude-opus-4-5", max_tokens=1024,
                                    tools=tools, messages=messages)
```

## 多轮对话管理

Anthropic API 是无状态的，每次请求需要手动携带完整对话历史：

```python
conversation_history = []

def chat(user_message: str) -> str:
    conversation_history.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system="你是一个友好的助手。",
        messages=conversation_history
    )

    assistant_message = response.content[0].text
    conversation_history.append({"role": "assistant", "content": assistant_message})
    return assistant_message
```

注意事项：
- 对话历史过长时需手动截断或压缩，保持在模型上下文窗口内
- Claude 3.5 系列上下文窗口为 200K Token

## Vision 图像输入

支持 JPEG、PNG、GIF、WebP 格式，可通过 Base64 或 URL 传入：

```python
import base64

with open("diagram.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_data
            }},
            {"type": "text", "text": "请描述这张架构图的主要组件"}
        ]
    }]
)
```

## Computer Use

Computer Use 是 Claude 3.5 Sonnet 的实验性功能，允许模型控制计算机桌面：

- 提供 `computer`、`text_editor`、`bash` 三种内置工具
- 模型可截图、点击、键盘输入、运行命令
- 适用场景：自动化测试、RPA、复杂界面交互
- 生产使用时需在沙箱环境运行，避免安全风险

## 计费方式

定价基于 Token 用量（Input + Output 分别计费）：

| 模型 | Input（每百万 Token） | Output（每百万 Token） |
|------|----------------------|----------------------|
| claude-opus-4-5 | $15 | $75 |
| claude-sonnet-4-5 | $3 | $15 |
| claude-haiku-3-5 | $0.8 | $4 |

成本优化策略：
- **Prompt Caching**：重复使用的长系统提示可缓存，缓存命中价格为原价的 10%
- **Batch API**：异步批处理，价格打五折，适合大量非实时请求
- 优先用 Haiku 做简单分类、路由任务

## 速率限制与错误处理

```python
import anthropic
import time

client = anthropic.Anthropic()

def create_with_retry(max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                time.sleep(wait_time)
            else:
                raise
        except anthropic.APIStatusError as e:
            print(f"API 错误 {e.status_code}: {e.message}")
            raise
```

常见错误码：
- `429`：超出速率限制，实现指数退避重试
- `529`：服务过载，稍后重试
- `400`：请求格式错误，检查参数

## 与 OpenAI SDK 的兼容层

Anthropic 提供兼容 OpenAI 接口的 endpoint，可以用 OpenAI SDK 直接调用 Claude：

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-anthropic-key",
    base_url="https://api.anthropic.com/v1/",
)

response = client.chat.completions.create(
    model="claude-opus-4-5",
    messages=[{"role": "user", "content": "Hello"}]
)
```

这在迁移已有 OpenAI 项目时非常方便，但部分 Anthropic 特有功能（如 Computer Use）不在兼容层内。
