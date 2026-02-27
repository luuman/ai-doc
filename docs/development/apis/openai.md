# OpenAI API 使用指南

OpenAI 是目前生态最成熟的 LLM 平台，提供从文本到图像、语音的全系列 API。本指南覆盖主要 API 的核心用法。

## Chat Completions API

这是最核心的接口，适用于绝大多数对话和文本生成场景：

```python
from openai import OpenAI

client = OpenAI()  # 自动读取 OPENAI_API_KEY 环境变量

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一个资深软件工程师。"},
        {"role": "user", "content": "解释一下 Python 的 GIL 机制"},
    ],
    temperature=0.7,
    max_tokens=1024,
)

print(response.choices[0].message.content)
print(f"Token 用量：{response.usage}")
```

关键参数：
- `model`：`gpt-4o`（旗舰）、`gpt-4o-mini`（轻量低价）、`o1`（推理模型）
- `temperature`：0.0-2.0，控制随机性，精确任务设 0
- `top_p`：核采样，与 temperature 二选一使用
- `presence_penalty` / `frequency_penalty`：减少重复内容，范围 -2 到 2

## 流式响应

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "详细解释 Transformer 架构"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Function Calling / Tool Use

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "在产品数据库中搜索商品",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "category": {"type": "string", "enum": ["electronics", "clothing", "food"]},
                    "max_price": {"type": "number", "description": "最高价格（元）"}
                },
                "required": ["query"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "帮我找价格在500元以内的蓝牙耳机"}],
    tools=tools,
    tool_choice="auto",  # auto/none/required 或指定工具
)

# 处理工具调用
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = search_database(**args)  # 自行实现

        # 将结果回传
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
```

## JSON Mode

强制模型输出合法 JSON，适合结构化数据提取：

```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "始终以 JSON 格式回复。"},
        {"role": "user", "content": "从以下文本提取姓名和邮箱：张三，zhangsan@example.com"}
    ]
)

data = json.loads(response.choices[0].message.content)
```

更推荐使用 Structured Outputs（`response_format={"type": "json_schema", ...}`）配合 JSON Schema 强制约束输出结构。

## Assistants API

Assistants API 提供有状态的对话管理，内置线程、文件和工具：

```python
# 1. 创建 Assistant
assistant = client.beta.assistants.create(
    name="代码助手",
    instructions="你是一个 Python 专家，帮助用户解决编程问题。",
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}, {"type": "file_search"}]
)

# 2. 创建 Thread（会话）
thread = client.beta.threads.create()

# 3. 添加消息
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="帮我分析这个 CSV 文件的数据分布"
)

# 4. 运行并轮询状态
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id
)

if run.status == "completed":
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    print(messages.data[0].content[0].text.value)
```

Assistants API 特性：
- 自动管理对话历史（Thread）
- 内置 File Search（文件检索，自动 Embedding）
- Code Interpreter（在沙箱中执行 Python）
- 支持上传文件（PDF、Excel、图片等）

## Batch API

异步批量请求，价格降低 50%，适合大规模数据处理：

```python
import jsonl  # pip install jsonlines

# 准备批量请求文件（JSONL 格式）
requests = [
    {
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 100
        }
    }
    for i, text in enumerate(texts)
]

# 上传文件并提交批次
batch_file = client.files.create(
    file=open("requests.jsonl", "rb"),
    purpose="batch"
)

batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"  # 24 小时内完成
)

# 轮询状态（通常几分钟到数小时）
batch = client.batches.retrieve(batch.id)
print(batch.status)  # validating/in_progress/completed/failed
```

## Embeddings API

```python
response = client.embeddings.create(
    model="text-embedding-3-small",  # 或 text-embedding-3-large
    input=["OpenAI 的 Embedding API 非常易用", "向量数据库存储稠密向量"],
    encoding_format="float"
)

vectors = [item.embedding for item in response.data]
print(f"向量维度：{len(vectors[0])}")  # small=1536, large=3072
```

模型对比：
- `text-embedding-3-small`：1536 维，性价比最高
- `text-embedding-3-large`：3072 维，支持 Matryoshka 降维，性能更好
- 支持 `dimensions` 参数缩短向量，降低存储成本

## DALL-E 3 图像生成

```python
response = client.images.generate(
    model="dall-e-3",
    prompt="一只在太空中穿宇航服的柴犬，科幻风格，4K 超清",
    size="1024x1024",       # 1024x1024/1024x1792/1792x1024
    quality="hd",           # standard/hd
    style="vivid",          # vivid/natural
    n=1
)

image_url = response.data[0].url
revised_prompt = response.data[0].revised_prompt  # DALL-E 改写后的提示
```

## Whisper 语音转文字

```python
with open("audio.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="zh",           # 指定语言提高准确率
        response_format="text",  # text/json/srt/vtt
        timestamp_granularities=["word"]  # 词级时间戳
    )

print(transcript.text)
```

## TTS API

```python
response = client.audio.speech.create(
    model="tts-1",      # tts-1（快速）或 tts-1-hd（高质量）
    voice="alloy",      # alloy/echo/fable/onyx/nova/shimmer
    input="你好，这是一段语音合成测试。",
    speed=1.0           # 0.25-4.0
)

response.stream_to_file("output.mp3")
```

## Prompt Caching

对于重复使用的长系统提示，OpenAI 会自动缓存（超过 1024 Token 的提示前缀）：
- 缓存命中时 Input Token 价格降低 50%
- 缓存有效期约 5-10 分钟（活跃使用时自动延续）
- 响应中 `usage.prompt_tokens_details.cached_tokens` 显示缓存命中数量

## 计费策略

| 模型 | Input | Output | 缓存 Input |
|------|-------|--------|-----------|
| GPT-4o | $2.5/M | $10/M | $1.25/M |
| GPT-4o mini | $0.15/M | $0.6/M | $0.075/M |
| o1 | $15/M | $60/M | $7.5/M |

成本控制建议：
- 用 GPT-4o mini 做路由、分类、摘要等简单任务
- 开启 Prompt Caching 减少重复提示费用
- 非实时任务使用 Batch API 节省 50%
