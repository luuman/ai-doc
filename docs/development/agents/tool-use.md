# 工具调用（Tool Use）

工具调用（Tool Use / Function Calling）是 AI Agent 与外部世界交互的核心机制。通过定义工具，LLM 可以获取实时信息、执行代码、操作文件和调用 API，大幅扩展其能力边界。

## Function Calling 机制

Function Calling 的工作流程：

1. 开发者定义工具（函数名、描述、参数 JSON Schema）
2. 将工具列表随消息一起发送给 LLM
3. LLM 决定是否需要调用工具，若需要则返回工具名和参数
4. 开发者执行实际的工具函数
5. 将执行结果回传给 LLM
6. LLM 基于结果继续推理或生成最终回答

关键点：LLM 本身不执行工具，只是"建议"调用哪个工具、传什么参数，实际执行由开发者代码完成。

## OpenAI 工具定义格式

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "在产品数据库中查询商品信息。返回商品名称、价格和库存数量。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "商品的唯一标识符"
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要返回的字段，可选：name/price/stock/description",
                        "default": ["name", "price", "stock"]
                    }
                },
                "required": ["product_id"]
            }
        }
    }
]
```

## Anthropic 工具定义格式

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "query_database",
        "description": "在产品数据库中查询商品信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "商品唯一标识符"
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "需要返回的字段列表"
                }
            },
            "required": ["product_id"]
        }
    }
]
```

两者的 JSON Schema 格式基本一致，差异在于 OpenAI 外层有 `type: "function"` 包装，而 Anthropic 直接是工具对象。

## 并行工具调用

现代 LLM（GPT-4o、Claude 3）支持在一次响应中同时调用多个工具，显著提高效率：

```python
def run_agent_with_parallel_tools(query: str):
    messages = [{"role": "user", "content": query}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message
        messages.append(message)

        if response.choices[0].finish_reason == "stop":
            return message.content

        # 处理所有工具调用（可能是并行的）
        if message.tool_calls:
            tool_results = []
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = execute_tool(tool_call.function.name, args)

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })

            messages.extend(tool_results)
```

## 工具执行与结果回传

```python
# 工具注册表：将工具名映射到实际函数
TOOL_REGISTRY = {
    "search_web": search_web,
    "query_database": query_database,
    "execute_python": execute_python,
    "send_email": send_email,
}

def execute_tool(tool_name: str, args: dict) -> str:
    if tool_name not in TOOL_REGISTRY:
        return f"错误：工具 {tool_name} 不存在"

    try:
        result = TOOL_REGISTRY[tool_name](**args)
        return str(result)
    except TypeError as e:
        return f"参数错误：{e}"
    except Exception as e:
        return f"执行失败：{type(e).__name__}: {e}"
```

## 错误处理与重试

工具调用失败时，将错误信息回传给 LLM，让其自行调整：

```python
def execute_with_retry(tool_name: str, args: dict, max_retries: int = 2) -> str:
    for attempt in range(max_retries + 1):
        try:
            return TOOL_REGISTRY[tool_name](**args)
        except Exception as e:
            if attempt < max_retries:
                # 返回错误，让 LLM 决定如何重试
                return f"第 {attempt + 1} 次尝试失败：{e}。请检查参数并重试。"
            return f"工具执行失败（已重试 {max_retries} 次）：{e}"
```

## 工具设计原则

### 单一职责

每个工具只做一件事，职责明确：

- 好的工具命名：`search_knowledge_base`、`create_calendar_event`、`get_stock_price`
- 避免：`do_everything_tool`（功能过于宽泛，LLM 难以准确选择）

### 清晰的描述

工具描述是 LLM 决定是否使用该工具的关键依据：

```python
# 差的描述
"description": "查询数据"

# 好的描述
"description": """
在企业产品数据库中查询商品信息。
- 适用场景：用户询问特定商品的价格、库存、规格
- 不适用：需要实时价格时请用 search_web 工具
- 返回格式：JSON 对象，包含 name/price/stock/description 字段
- 注意：product_id 格式为 "SKU-XXXXX"，共 10 个字符
"""
```

### 参数描述详细

每个参数都应有明确的类型、含义和取值范围说明，减少 LLM 传错参数。

## 内置工具类型

### 代码执行（Code Interpreter）

在沙箱环境中执行 Python 代码，可进行数据分析、计算、绘图：

```python
# OpenAI Assistants API 内置
assistant = client.beta.assistants.create(
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}]
)
```

### 网络搜索

```python
def search_web(query: str, num_results: int = 5) -> list[dict]:
    """使用搜索引擎获取实时信息"""
    from googleapiclient.discovery import build
    service = build("customsearch", "v1", developerKey=API_KEY)
    result = service.cse().list(q=query, cx=CSE_ID, num=num_results).execute()
    return [{"title": item["title"], "snippet": item["snippet"], "url": item["link"]}
            for item in result.get("items", [])]
```

### 文件操作

```python
def read_file(path: str) -> str:
    """读取指定路径的文本文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path: str, content: str) -> str:
    """将内容写入指定路径的文件"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"已成功写入 {path}（{len(content)} 个字符）"
```

## 工具安全边界

工具调用引入了严重的安全风险，需要明确边界：

- **读写分离**：区分只读工具和会产生副作用的写操作工具
- **操作确认**：发送邮件、删除文件等不可逆操作，建议先返回预览，等待用户确认
- **沙箱隔离**：代码执行类工具必须在隔离环境运行，防止访问宿主系统
- **权限最小化**：数据库工具只授予 SELECT 权限，不给 DROP/DELETE
- **速率限制**：防止 Agent 在循环中大量调用付费 API
- **路径白名单**：文件操作工具限制在指定目录内

## MCP（Model Context Protocol）

MCP 是 Anthropic 于 2024 年底推出的开放标准，用于统一 LLM 应用与外部工具/数据源的集成方式：

- **标准化接口**：任何工具只需实现 MCP 协议，即可被所有支持 MCP 的 LLM 客户端使用
- **三种能力**：Resources（数据访问）、Tools（工具调用）、Prompts（提示模板）
- **双向通信**：通过 stdio 或 HTTP/SSE 实现 LLM 客户端与工具服务器的通信
- **生态系统**：Claude Desktop、Cursor、各开源框架已支持 MCP

```python
# MCP 服务器示例（使用 mcp Python SDK）
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("my-tools")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="query_database",
            description="查询企业数据库",
            inputSchema={"type": "object", "properties": {"sql": {"type": "string"}}}
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "query_database":
        result = execute_sql(arguments["sql"])
        return [TextContent(type="text", text=str(result))]

async def main():
    async with stdio_server() as streams:
        await server.run(*streams)
```
