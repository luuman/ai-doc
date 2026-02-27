# 多模型路由

在 AI 应用成熟的今天，单一模型很少是最优解。不同的模型在能力、成本、延迟、可用性上各有优劣。多模型路由（Multi-Model Routing）通过智能地将请求分发给最合适的模型，在满足用户需求的前提下最大化效率和可靠性。

## 路由动机

为何需要多模型路由：

**成本控制**：
- Claude Opus（约 $15/M tokens）vs Claude Haiku（约 $0.25/M tokens），成本相差 60 倍
- 简单的问候、格式转换等任务使用大模型是严重浪费
- 合理的路由可以在不影响质量的前提下降低 70-90% 的成本

**能力互补**：
- 不同模型在不同任务上有各自优势（代码、数学、多语言、多模态）
- 单一模型无法做到所有任务最优

**可用性保障**：
- API 服务存在故障（Rate Limit、服务中断）
- 主模型不可用时，自动切换到备用模型，保证服务连续性

**延迟优化**：
- 对于时间敏感的任务，选择延迟更低的模型（即使质量稍低）
- 对于离线批处理任务，选择更强的模型（延迟不敏感）

## 路由策略

### 按任务类型路由

基于请求的任务类型选择模型：

```python
TASK_MODEL_MAP = {
    "code_generation": "claude-opus-4-5",      # 代码质量优先，用最强模型
    "code_explain": "claude-sonnet-4-5",        # 解释任务中等复杂度
    "translation": "claude-haiku-3-5",          # 翻译任务简单，用小模型
    "creative_writing": "claude-opus-4-5",      # 创意写作需要丰富语言
    "simple_qa": "claude-haiku-3-5",            # 简单问答用最经济模型
    "math_reasoning": "claude-opus-4-5",        # 数学推理需要最强能力
    "summarization": "claude-haiku-3-5",        # 摘要任务性价比优先
}

def route_by_task(task_type: str) -> str:
    return TASK_MODEL_MAP.get(task_type, "claude-sonnet-4-5")
```

### 按成本预算路由

根据用户的订阅等级或剩余配额选择模型：

```python
def route_by_budget(user_id: str, prompt: str) -> str:
    remaining_budget = get_user_budget(user_id)
    estimated_tokens = estimate_tokens(prompt)
    estimated_cost = estimated_tokens * HAIKU_PRICE

    if remaining_budget <= 0:
        raise QuotaExceededException()
    elif remaining_budget > 10.0:  # 充足预算，使用最强模型
        return "claude-opus-4-5"
    elif remaining_budget > 1.0:   # 中等预算
        return "claude-sonnet-4-5"
    else:                           # 预算紧张，使用经济模型
        return "claude-haiku-3-5"
```

### 按延迟要求路由

```python
def route_by_latency(request: ChatRequest) -> str:
    if request.max_latency_ms and request.max_latency_ms < 500:
        # 极低延迟要求：使用速度最快的模型
        return "claude-haiku-3-5"
    elif request.stream:
        # 流式请求：Sonnet 在 TTFT 上优于 Opus
        return "claude-sonnet-4-5"
    else:
        return "claude-opus-4-5"
```

### 按负载均衡路由

在多个 API Key 或多个模型提供商之间分配负载：

```python
import random

LOAD_BALANCED_MODELS = [
    {"model": "gpt-4o", "weight": 0.5},
    {"model": "claude-opus-4-5", "weight": 0.3},
    {"model": "gemini-1.5-pro", "weight": 0.2},
]

def route_load_balanced() -> str:
    rand = random.random()
    cumulative = 0
    for config in LOAD_BALANCED_MODELS:
        cumulative += config["weight"]
        if rand < cumulative:
            return config["model"]
    return LOAD_BALANCED_MODELS[-1]["model"]
```

## 意图识别路由（小模型分类）

使用轻量级分类器分析用户请求，自动判断任务复杂度和类型：

```python
from transformers import pipeline

# 使用小型分类模型判断任务复杂度
classifier = pipeline(
    "text-classification",
    model="your-task-classifier",  # 自训练的二分类模型
    device="cpu"
)

COMPLEXITY_THRESHOLD = 0.7

def route_by_intent(user_message: str) -> str:
    result = classifier(user_message[:512])  # 截断避免过长
    complexity_score = result[0]["score"] if result[0]["label"] == "complex" else 1 - result[0]["score"]

    if complexity_score > COMPLEXITY_THRESHOLD:
        return "claude-opus-4-5"   # 复杂任务
    else:
        return "claude-haiku-3-5"  # 简单任务

# 也可以使用 embedding 相似度
from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

COMPLEX_EXAMPLES = ["解释量子纠缠的数学原理", "写一个多线程安全的红黑树实现", ...]
SIMPLE_EXAMPLES = ["你好", "谢谢", "今天天气怎么样", ...]

complex_embeddings = embed_model.encode(COMPLEX_EXAMPLES)
simple_embeddings = embed_model.encode(SIMPLE_EXAMPLES)

def route_by_embedding(query: str) -> str:
    query_embed = embed_model.encode([query])[0]
    complex_sim = np.max(np.dot(complex_embeddings, query_embed))
    simple_sim = np.max(np.dot(simple_embeddings, query_embed))
    return "claude-opus-4-5" if complex_sim > simple_sim else "claude-haiku-3-5"
```

## 基于规则的路由

快速、零成本的规则匹配，适合明确的路由场景：

```python
import re

ROUTING_RULES = [
    # (正则/关键词, 目标模型)
    (r"写.*代码|编程|函数|算法|debug", "claude-opus-4-5"),
    (r"翻译|translate", "claude-haiku-3-5"),
    (r"图片|图像|image|photo", "gpt-4o"),          # 多模态任务
    (r"你好|Hello|Hi|谢谢|Thanks", "claude-haiku-3-5"),  # 简单交互
    (r"数学|方程|积分|导数|证明", "claude-opus-4-5"),
]

def route_by_rules(query: str) -> str:
    query_lower = query.lower()
    for pattern, model in ROUTING_RULES:
        if re.search(pattern, query_lower):
            return model
    return "claude-sonnet-4-5"  # 默认模型
```

## LiteLLM（统一多模型网关）

LiteLLM 是一个开源的统一 LLM 网关，提供单一 OpenAI 兼容接口，后端连接 100+ 个模型提供商：

```bash
pip install litellm
litellm --model gpt-4o  # 启动代理服务
```

```python
from litellm import completion, Router

# 统一调用不同提供商
response = completion(
    model="claude-opus-4-5",  # 或 "gpt-4o" 或 "gemini/gemini-1.5-pro"
    messages=[{"role": "user", "content": "你好"}]
)

# 配置路由器（自动负载均衡 + 降级）
router = Router(
    model_list=[
        {
            "model_name": "primary",
            "litellm_params": {
                "model": "claude-opus-4-5",
                "api_key": os.getenv("ANTHROPIC_API_KEY")
            }
        },
        {
            "model_name": "fallback",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        }
    ],
    fallbacks=[{"primary": ["fallback"]}],  # primary 失败时使用 fallback
    num_retries=3,
    retry_after=5
)

response = await router.acompletion(
    model="primary",
    messages=[{"role": "user", "content": "你好"}]
)
```

LiteLLM 功能：
- 统一 API（一次集成，切换模型无需改代码）
- 自动重试和降级（Fallback）
- 成本追踪（记录每次调用的 Token 和成本）
- 速率限制管理（多 Key 轮换）
- 缓存（基于 Redis 的语义缓存）

## OpenRouter（云端路由）

OpenRouter 是一个云端 LLM 路由服务，提供统一的 API 接入数十个模型：

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client.chat.completions.create(
    model="anthropic/claude-opus-4-5",  # 或 openai/gpt-4o, google/gemini-pro
    messages=[{"role": "user", "content": "你好"}],
    extra_headers={
        "OR-Organization": "my-org",
        "OR-Site-URL": "https://myapp.com"
    }
)
```

OpenRouter 的特色：
- 自动选择最低价格的可用提供商（模型相同但提供商不同）
- 实时模型价格和可用性监控
- 无需管理多个 API Key

## 降级策略

主模型失败时的自动降级设计：

```python
async def call_with_fallback(
    messages: list,
    primary_model: str = "claude-opus-4-5",
    fallback_models: list = ["claude-sonnet-4-5", "claude-haiku-3-5"]
) -> str:
    models_to_try = [primary_model] + fallback_models

    for model in models_to_try:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=30.0
            )
            return response.choices[0].message.content
        except RateLimitError:
            logger.warning(f"Rate limit on {model}, trying next...")
            continue
        except APIStatusError as e:
            if e.status_code >= 500:
                logger.error(f"Server error on {model}: {e}")
                continue
            raise  # 4xx 错误不降级，直接抛出

    raise AllModelsFailedException("所有模型均不可用")
```

## A/B 测试不同模型

系统性地评估不同模型对业务指标的影响：

```python
import hashlib

def ab_test_model(user_id: str) -> str:
    """基于用户 ID 的稳定分组（同一用户始终分到同一组）"""
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    bucket = hash_val % 100  # 0-99

    if bucket < 50:
        return "claude-opus-4-5"      # 组A：50% 用户
    else:
        return "claude-haiku-3-5"     # 组B：50% 用户

# 记录实验数据
def log_experiment(user_id: str, model: str, response: str, user_feedback: int):
    metrics_db.insert({
        "experiment": "model_ab_test",
        "user_id": user_id,
        "model": model,
        "response_length": len(response),
        "user_feedback": user_feedback,  # 点赞=1，点踩=-1，无反馈=0
        "timestamp": datetime.utcnow()
    })
```

## 路由效果监控

持续追踪路由决策的质量：

```python
# 路由监控指标
ROUTING_METRICS = {
    "model_usage_ratio": {},        # 各模型使用比例
    "model_error_rate": {},         # 各模型错误率
    "model_latency_p99": {},        # 各模型 P99 延迟
    "model_cost_per_request": {},   # 各模型平均请求成本
    "routing_accuracy": 0.0         # 路由决策与人工评估的一致率
}

# 发现路由问题的信号
# - 某模型错误率持续上升 → 考虑临时降低其权重
# - 意图识别路由与人工复核差异 > 20% → 重新训练分类器
# - 成本超出预期 → 检查是否有过多简单任务被路由到大模型
```
