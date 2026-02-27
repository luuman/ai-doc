# 流式生成

流式生成（Streaming Generation）是现代 AI 应用的基础交互范式。当模型需要数秒甚至数十秒才能完成完整输出时，等待所有内容生成完毕再显示会造成极差的用户体验。流式传输使得用户能够实时看到模型的输出，就像人类打字一样自然。

## 为何需要流式（TTFT vs 完整等待）

考虑生成一篇 500 字文章的场景：
- 假设模型每秒生成 40 个 Token（约 30 个中文字）
- 完整等待：用户等待约 17 秒，然后一次性看到全文
- 流式传输：用户约 500ms 后看到第一个字，持续看到文字出现，总时长相同

**TTFT（Time To First Token）**是用户感知的关键指标：即使总生成时间相同，流式传输通过极低的 TTFT（通常 < 1 秒）大幅提升了用户体验。心理学研究表明，当系统响应时间超过 1 秒时，用户会感到明显等待；超过 10 秒则会严重影响专注度。

流式传输带来的工程挑战：
- 服务端需要支持长连接和逐步发送数据
- 客户端需要处理不完整的响应（Partial Response）
- 中间层（代理、CDN）不能缓冲完整响应
- 错误处理（连接中断如何恢复）更复杂

## SSE（Server-Sent Events）协议

SSE 是 HTML5 标准中定义的单向推送协议，是 LLM 流式传输的标准实现方式。

SSE 的 HTTP 特征：
- **Content-Type**: `text/event-stream`
- **Connection**: `keep-alive`
- **Cache-Control**: `no-cache`
- 响应保持连接，服务端持续发送数据

SSE 数据格式（每个事件由空行分隔）：

```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"你"},"index":0}]}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"好"},"index":0}]}

data: [DONE]

```

每行以 `data: ` 开头（注意有空格），最后发送 `data: [DONE]` 标记流结束。OpenAI 的流式 API 完全采用此格式，所有 OpenAI 兼容的服务也遵循此标准。

SSE 相比 WebSocket 的优势：
- 基于普通 HTTP，无需握手，防火墙/代理友好
- 自动重连（浏览器原生支持）
- 单向推送，对 LLM 场景足够（无需双向实时通信）

## 流式 Token 生成实现

### Python 同步生成器

```python
from openai import OpenAI

client = OpenAI()

def stream_response(prompt: str):
    """同步生成器，逐个 yield Token"""
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

# 使用
for token in stream_response("讲一个故事"):
    print(token, end="", flush=True)
```

### Python 异步生成器（AsyncGenerator）

在异步 Web 框架（FastAPI、aiohttp）中，使用 async 生成器处理流式输出：

```python
from openai import AsyncOpenAI
from typing import AsyncGenerator

client = AsyncOpenAI()

async def stream_tokens(prompt: str) -> AsyncGenerator[str, None]:
    """异步生成器，适用于 FastAPI 等异步框架"""
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

# FastAPI 流式端点
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for token in stream_tokens(request.message):
            # SSE 格式
            yield f"data: {json.dumps({'content': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

### 服务端 vLLM 流式输出

```python
from vllm import AsyncLLMEngine, SamplingParams

engine = AsyncLLMEngine.from_engine_args(engine_args)

async def generate_stream(prompt: str):
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
    request_id = str(uuid.uuid4())

    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            token_text = output.outputs[0].text
            yield token_text
```

## 前端流式显示

### React + ReadableStream

```jsx
import { useState } from "react";

function StreamingChat() {
    const [response, setResponse] = useState("");

    const sendMessage = async (message) => {
        const res = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message })
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            // 解析 SSE 格式
            const lines = text.split("\n");
            for (const line of lines) {
                if (line.startsWith("data: ") && line !== "data: [DONE]") {
                    const data = JSON.parse(line.slice(6));
                    setResponse(prev => prev + (data.content || ""));
                }
            }
        }
    };

    return (
        <div>
            <div className="response">{response}</div>
            <button onClick={() => sendMessage("你好")}>发送</button>
        </div>
    );
}
```

### 使用 EventSource API

浏览器原生 EventSource API 专为 SSE 设计，但不支持 POST 请求：

```javascript
const source = new EventSource("/api/chat/stream?q=你好");

source.onmessage = (event) => {
    if (event.data === "[DONE]") {
        source.close();
        return;
    }
    const data = JSON.parse(event.data);
    document.getElementById("output").textContent += data.content;
};

source.onerror = () => source.close();
```

对于需要 POST Body 的场景，使用 `fetch` + `ReadableStream` 替代。

## 流式 JSON 解析挑战

当要求模型输出 JSON 格式时，流式传输会产生不完整的 JSON（如只收到了 `{"name": "张`），前端无法直接解析。

解决方案：

1. **增量 JSON 解析**：使用 `jsonstream`、`streaming-json-parser` 等库处理不完整 JSON
2. **延迟解析**：累积完整 JSON 后再解析（放弃流式显示中间状态）
3. **结构化流**：如 Vercel AI SDK 的 `streamObject`，服务端保证每次 flush 都是有效的部分 JSON

```javascript
// 使用 Vercel AI SDK 流式 JSON
import { streamObject } from "ai";
import { z } from "zod";

const stream = streamObject({
    model: openai("gpt-4o"),
    schema: z.object({
        title: z.string(),
        items: z.array(z.string())
    }),
    prompt: "生成一个待办清单"
});

for await (const partial of stream.partialObjectStream) {
    console.log(partial);  // 逐步填充的对象，始终是合法状态
}
```

## 中断与取消（Abort Signal）

用户可能在生成过程中点击"停止"按钮，需要优雅地中断流式生成：

```javascript
// 前端取消
const controller = new AbortController();

const res = await fetch("/api/chat/stream", {
    signal: controller.signal,
    // ...
});

// 用户点击停止
cancelButton.onclick = () => controller.abort();
```

```python
# 后端处理取消（FastAPI）
from fastapi import Request

@app.post("/chat/stream")
async def chat_stream(request: Request, query: str):
    async def generate():
        async for token in stream_tokens(query):
            # 检查客户端是否已断开
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps({'content': token})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

## 流式工具调用

当模型调用工具（Function Calling）时，工具调用参数也以流式方式传输。工具调用不应中途显示给用户，需要在接收完整工具调用后再执行：

```python
# OpenAI 流式工具调用处理
tool_calls = []
current_tool = None

async for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        for tc in delta.tool_calls:
            if tc.index not in tool_calls:
                tool_calls[tc.index] = {"id": "", "function": {"name": "", "arguments": ""}}
            if tc.id:
                tool_calls[tc.index]["id"] = tc.id
            if tc.function.name:
                tool_calls[tc.index]["function"]["name"] = tc.function.name
            if tc.function.arguments:
                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

# 工具调用完整后执行
for tc in tool_calls.values():
    args = json.loads(tc["function"]["arguments"])
    result = execute_tool(tc["function"]["name"], args)
```

## Nginx/CDN 流式代理配置

Nginx 默认会缓冲上游响应，导致流式输出无法实时传递给客户端：

```nginx
location /api/chat/ {
    proxy_pass http://backend:8000;

    # 禁用响应缓冲（流式必须）
    proxy_buffering off;
    proxy_cache off;

    # 增大超时时间（长文本生成可能超过默认 60s）
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    # SSE 必要头部
    proxy_set_header Connection "";
    proxy_http_version 1.1;

    # 告知 Nginx 不缓存此响应
    add_header X-Accel-Buffering no;
}
```

CDN 配置（Cloudflare 示例）：
- 在 Cache Rules 中对流式 API 路径设置 `Cache Level: Bypass`
- Cloudflare 的 Workers 支持原生流式响应（不缓冲）
- AWS CloudFront 需要将 `TTL` 设为 0 并禁用压缩（gzip 会导致缓冲）
