# 成本优化

LLM API 的费用随使用量快速增长，在产品规模化后往往成为可观的运营支出。系统性的成本优化可以在不牺牲用户体验的前提下，将 LLM 成本降低 50-90%。理解成本结构是优化的起点。

## LLM 成本结构

LLM API 通常按 Token 计费，分为三个部分：

**输入 Token（Input / Prompt Tokens）**：
- 发送给模型的所有内容：System Prompt + 历史对话 + 用户消息 + 上下文
- 通常比输出 Token 便宜（约 1/3 到 1/5 价格）
- 是多轮对话和 RAG 场景中最大的成本来源

**输出 Token（Output / Completion Tokens）**：
- 模型生成的内容
- 通常比输入贵，因为自回归生成的计算开销更大
- 受 `max_tokens` 参数控制上限

**请求次数**：
- 部分服务按请求收费（如 Perplexity API），大多数按 Token 计费不额外收请求费

以 Claude Opus 4.5 为例：
- 输入：$15 / 1M tokens
- 输出：$75 / 1M tokens
- 月均 100 万次请求，每次平均 500 input + 200 output tokens：
  - 月成本 = (1M × 500 × $15/1M) + (1M × 200 × $75/1M) = $7,500 + $15,000 = $22,500/月

## Prompt 压缩技术

### LLMLingua

LLMLingua（Microsoft）是一种基于模型的 Prompt 压缩方法：

**工作原理**：
1. 使用小型语言模型（如 GPT-2）计算 Prompt 中每个 Token 的条件概率
2. 低概率 Token（信息量大，难以预测）保留；高概率 Token（冗余、可预测）压缩或删除
3. 在保持语义的前提下，压缩率可达 2-5 倍

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    device_map="cpu"
)

compressed = compressor.compress_prompt(
    context=long_document,
    question=user_query,
    rate=0.5,           # 压缩到原长度的 50%
    force_tokens=["。", "！", "\n"]  # 强制保留的 Token
)

print(f"原始长度: {len(original.split())} tokens")
print(f"压缩后: {len(compressed['compressed_prompt'].split())} tokens")
print(f"压缩率: {compressed['ratio']:.1%}")
```

LLMLingua 在 RAG 场景特别有效：将检索到的文档上下文压缩后再喂给大模型，可在保持 95% 以上准确率的前提下节省 50% 的输入 Token。

### 手动 Prompt 精简

在专业场景下，精心设计的简洁 Prompt 往往比自动压缩效果更好：

```
# 冗余的写法（约 30 tokens）
你是一个非常有帮助、友善、耐心的 AI 助手，你总是尽力给用户提供最好的帮助和支持，
请根据用户的问题给出详细、准确、有用的回答。

# 精简写法（约 10 tokens）
你是 AI 助手，请简洁准确地回答问题。
```

精简 System Prompt 的原则：
- 去除语气词和修饰语（"非常"、"尽力"）
- 删除隐含的默认行为描述
- 用示例代替文字描述（少量 Few-shot 比长篇 instruction 更高效）

## Prompt Caching（前缀缓存）

### Claude Prompt Caching

Anthropic 提供的 Prompt Caching 功能可以缓存 Prompt 的固定前缀，后续请求复用已缓存内容，**缓存命中的 Token 价格降低 90%**：

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "你是一个专业的 Python 代码审查助手...",  # 固定系统提示
        },
        {
            "type": "text",
            "text": long_codebase_context,  # 大型代码库上下文
            "cache_control": {"type": "ephemeral"}  # 标记为可缓存（5分钟TTL）
        }
    ],
    messages=[{"role": "user", "content": "请审查这段代码的安全性"}]
)

# 首次请求：正常计费（写入缓存）
# 后续请求（5分钟内）：缓存命中，节省 90% 的输入成本
```

适用场景：
- 固定的长 System Prompt（角色设定、规则手册）
- RAG 检索文档（同一文档被多次查询）
- Long Context 分析（同一份报告被多个问题查询）

### OpenAI Prompt Caching

OpenAI 对于超过 1024 tokens 的 Prompt，自动缓存前缀（无需显式标记），缓存命中价格降低 50%：

- 缓存粒度：1024 tokens 的倍数（Prompt 最大公共前缀）
- TTL：约 5-10 分钟（不固定）
- 适用模型：GPT-4o、GPT-4o-mini

## Batch API（批量处理）

### OpenAI Batch API

对于不需要实时响应的任务（数据分析、批量标注、离线评估），OpenAI Batch API 提供 50% 的价格折扣：

```python
from openai import OpenAI
import json

client = OpenAI()

# 准备批量请求（JSONL 格式）
requests = []
for i, item in enumerate(dataset):
    requests.append({
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": f"分析以下文本的情感: {item['text']}"}
            ],
            "max_tokens": 50
        }
    })

# 上传批量请求文件
with open("batch_requests.jsonl", "w") as f:
    for req in requests:
        f.write(json.dumps(req, ensure_ascii=False) + "\n")

batch_file = client.files.create(
    file=open("batch_requests.jsonl", "rb"),
    purpose="batch"
)

# 创建批量任务（24小时内完成）
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# 轮询检查状态
import time
while True:
    batch = client.batches.retrieve(batch.id)
    if batch.status == "completed":
        break
    time.sleep(60)

# 下载结果
results = client.files.content(batch.output_file_id)
```

Batch API 的代价：延迟（最长 24 小时），不适合实时交互场景。

## 模型降级

根据任务复杂度智能选择模型，是最直接有效的成本优化手段：

```python
TASK_COMPLEXITY_MAP = {
    # 简单任务 → 使用最经济的模型
    "intent_classification": "claude-haiku-3-5",    # $0.25/M input
    "translation": "claude-haiku-3-5",
    "keyword_extraction": "claude-haiku-3-5",
    "simple_summarization": "claude-haiku-3-5",

    # 中等复杂度 → 平衡模型
    "structured_extraction": "claude-sonnet-4-5",   # $3/M input
    "document_analysis": "claude-sonnet-4-5",
    "multi_turn_chat": "claude-sonnet-4-5",

    # 复杂任务 → 最强模型
    "code_generation": "claude-opus-4-5",           # $15/M input
    "complex_reasoning": "claude-opus-4-5",
    "creative_long_form": "claude-opus-4-5"
}
```

## Output 长度控制

输出 Token 通常比输入贵 3-5 倍，控制输出长度是直接的成本优化手段：

```python
# 在 Prompt 中明确要求简洁
system_prompt = """
回答要求：
- 直接给出答案，不需要重复问题
- 不需要总结段落
- 代码示例控制在 20 行以内
"""

# 设置合理的 max_tokens 上限
response = client.chat.completions.create(
    model="claude-opus-4-5",
    messages=[...],
    max_tokens=500  # 根据任务场景设置，不要使用过大默认值
)

# 监控实际输出长度分布
# 如果 P90 输出长度只有 200 tokens，但 max_tokens=2000，则过度预留
```

## 语义缓存

对于相似的用户查询，不需要每次都调用 LLM，可以直接返回缓存的答案：

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import redis
import json

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
cache = redis.Redis()

SIMILARITY_THRESHOLD = 0.95  # 相似度阈值

def get_cached_response(query: str) -> str | None:
    query_embed = embed_model.encode([query])[0]

    # 从缓存中获取所有已缓存的问题
    cached_keys = cache.keys("qa:embed:*")
    for key in cached_keys:
        cached_data = json.loads(cache.get(key))
        cached_embed = np.array(cached_data["embedding"])
        similarity = np.dot(query_embed, cached_embed) / (
            np.linalg.norm(query_embed) * np.linalg.norm(cached_embed)
        )
        if similarity > SIMILARITY_THRESHOLD:
            return cached_data["response"]
    return None

def call_llm_with_cache(query: str) -> str:
    # 1. 检查缓存
    cached = get_cached_response(query)
    if cached:
        return cached

    # 2. 调用 LLM
    response = call_llm(query)

    # 3. 存入缓存（TTL 24 小时）
    cache_data = {
        "query": query,
        "embedding": embed_model.encode([query])[0].tolist(),
        "response": response
    }
    cache.setex(
        f"qa:embed:{hash(query)}",
        86400,  # 24 小时
        json.dumps(cache_data)
    )

    return response
```

注意：语义缓存不适合对实时性要求高的场景（天气、股价、最新新闻），这类场景即使查询相似也应该重新调用。

## 成本监控告警

系统性的成本监控是防止意外超支的基础：

```python
import anthropic
from datetime import datetime
import smtplib

class CostTracker:
    def __init__(self, daily_budget: float, alert_threshold: float = 0.8):
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold
        self.daily_costs = {}  # date -> total_cost

    def track_usage(self, response):
        """追踪每次调用的成本"""
        usage = response.usage
        date = datetime.today().strftime("%Y-%m-%d")

        # Claude 定价（per 1M tokens）
        input_cost = (usage.input_tokens / 1_000_000) * 15.0
        output_cost = (usage.output_tokens / 1_000_000) * 75.0
        total_cost = input_cost + output_cost

        self.daily_costs[date] = self.daily_costs.get(date, 0) + total_cost

        # 检查是否超过告警阈值
        if self.daily_costs[date] >= self.daily_budget * self.alert_threshold:
            self.send_alert(self.daily_costs[date])

        return total_cost

    def send_alert(self, current_cost: float):
        """发送成本告警（示例：钉钉/企业微信 Webhook）"""
        import requests
        requests.post(
            "https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN",
            json={
                "msgtype": "text",
                "text": {"content": f"LLM 成本告警：今日已消费 ${current_cost:.2f}，超过预算 {self.alert_threshold:.0%}"}
            }
        )
```

成本优化的整体效果：

| 优化手段 | 典型节省比例 | 实施难度 |
|---------|------------|---------|
| 模型降级（大→小） | 60-90% | 低 |
| Prompt Caching | 50-90%（含缓存 token） | 低 |
| 语义缓存 | 30-70%（取决于重复率） | 中 |
| Prompt 压缩 | 30-50%（RAG场景） | 中 |
| Batch API | 50%（定价折扣） | 低 |
| Output 长度控制 | 20-40% | 低 |
| 组合以上所有 | 75-95% | 中 |
