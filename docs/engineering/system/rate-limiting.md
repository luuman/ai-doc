# 限流设计

限流（Rate Limiting）是 AI 服务稳定运营的核心基础设施。无论是对上游 LLM API 的调用限制，还是对下游用户的访问控制，合理的限流设计都是防止成本失控、保证服务公平性的必要手段。

## 为何需要限流

**API 限额**：LLM 提供商对 API 调用有严格限制。OpenAI GPT-4o 的默认速率限制为 10,000 RPM（每分钟请求数）和 30,000 TPM（每分钟 Token 数）。超出限额会收到 `429 Too Many Requests` 错误，导致请求失败。

**成本控制**：单个用户或脚本可能在短时间内消耗大量 Token，造成意外账单。限流是成本控制的最后一道防线。

**公平使用**：在多租户系统中，单个高频用户会占用大量资源，影响其他用户的体验。限流保证资源在用户间公平分配。

**安全防护**：限流可以减缓 Prompt 注入攻击、爬虫抓取等恶意行为的影响。

## 令牌桶算法（Token Bucket）

令牌桶是最常用的限流算法，允许一定程度的突发流量：

**工作原理**：
- 桶的容量为 `capacity`（最大令牌数）
- 系统以固定速率 `rate` 向桶中添加令牌（如每秒 10 个）
- 每个请求消耗 1 个（或多个）令牌
- 桶满时不再添加令牌
- 桶空时请求被拒绝或等待

**优势**：允许短时间内的突发请求（桶中积累的令牌），同时保证长期的平均速率不超限。

```python
import time
import threading

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        """
        capacity: 桶的最大容量（令牌数）
        refill_rate: 补充速率（令牌/秒）
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity  # 初始化为满桶
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        """补充令牌（根据经过时间计算）"""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """消耗令牌，返回是否成功"""
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_and_consume(self, tokens: int = 1):
        """等待直到有足够令牌"""
        while not self.consume(tokens):
            time.sleep(0.1)

# 使用示例
bucket = TokenBucket(capacity=100, refill_rate=10)  # 桶容量100，每秒补充10个令牌

if bucket.consume():
    response = call_llm(prompt)
else:
    raise RateLimitException("请求过于频繁")
```

**LLM 的 Token 感知限流**：LLM API 限制的是 Token 数而非请求数，需要将每次请求的 Token 消耗作为令牌桶的消耗量：

```python
# 估算 Token 数
def estimate_tokens(text: str) -> int:
    return len(text) // 4  # 粗略估算（英文约4字符/token，中文约2字符/token）

# 基于 Token 的令牌桶
token_bucket = TokenBucket(capacity=30000, refill_rate=500)  # 每秒500 tokens

input_tokens = estimate_tokens(prompt)
if not token_bucket.consume(input_tokens):
    raise RateLimitException("Token 配额不足，请稍后重试")
```

## 滑动窗口算法

滑动窗口算法统计时间窗口内的请求数，比固定窗口更准确：

```python
from collections import deque
import time

class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()  # 存储请求时间戳
        self.lock = threading.Lock()

    def is_allowed(self) -> bool:
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds

            # 移除窗口外的请求记录
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()

            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

# 示例：每分钟最多100个请求
limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
```

## 用户级配额（每日/每月 Token 限制）

多租户系统需要对每个用户设置独立的配额：

```python
class UserQuotaManager:
    def __init__(self, redis_client):
        self.redis = redis_client

    def get_quota_key(self, user_id: str, period: str) -> str:
        from datetime import datetime
        if period == "daily":
            date = datetime.utcnow().strftime("%Y-%m-%d")
            return f"quota:daily:{user_id}:{date}"
        elif period == "monthly":
            month = datetime.utcnow().strftime("%Y-%m")
            return f"quota:monthly:{user_id}:{month}"

    def check_and_consume(
        self,
        user_id: str,
        tokens: int,
        daily_limit: int,
        monthly_limit: int
    ) -> bool:
        daily_key = self.get_quota_key(user_id, "daily")
        monthly_key = self.get_quota_key(user_id, "monthly")

        # 使用 Redis 事务原子性地检查和更新配额
        with self.redis.pipeline() as pipe:
            pipe.watch(daily_key, monthly_key)
            daily_used = int(self.redis.get(daily_key) or 0)
            monthly_used = int(self.redis.get(monthly_key) or 0)

            if daily_used + tokens > daily_limit:
                raise QuotaExceededException(f"每日配额不足：已用 {daily_used}，上限 {daily_limit}")
            if monthly_used + tokens > monthly_limit:
                raise QuotaExceededException(f"每月配额不足")

            pipe.multi()
            pipe.incrby(daily_key, tokens)
            pipe.expire(daily_key, 86400)   # 1天后过期
            pipe.incrby(monthly_key, tokens)
            pipe.expire(monthly_key, 2592000)  # 30天后过期
            pipe.execute()

        return True

# 不同用户等级的配额配置
QUOTA_TIERS = {
    "free": {"daily": 10_000, "monthly": 100_000},
    "pro": {"daily": 100_000, "monthly": 2_000_000},
    "enterprise": {"daily": 10_000_000, "monthly": 200_000_000}
}
```

## 请求队列设计

当请求超过当前处理能力时，排队等候优于直接拒绝：

```python
import asyncio
from asyncio import Queue

class RequestQueue:
    def __init__(self, max_concurrent: int = 10, max_queue_size: int = 1000):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = Queue(maxsize=max_queue_size)

    async def enqueue(self, request_fn, *args, **kwargs):
        """将请求加入队列并等待处理"""
        future = asyncio.Future()
        try:
            self.queue.put_nowait((request_fn, args, kwargs, future))
        except asyncio.QueueFull:
            raise ServiceUnavailableException("服务繁忙，请稍后重试")
        return await future

    async def worker(self):
        """消费队列中的请求"""
        while True:
            fn, args, kwargs, future = await self.queue.get()
            async with self.semaphore:
                try:
                    result = await fn(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
```

## 限流响应（429 + Retry-After）

限流触发时，需要给客户端明确的反馈：

```python
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

@app.exception_handler(RateLimitException)
async def rate_limit_handler(request: Request, exc: RateLimitException):
    retry_after = exc.retry_after_seconds  # 告知客户端等待时间

    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "type": "rate_limit_error",
                "message": exc.message,
                "retry_after": retry_after
            }
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": str(exc.limit),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time()) + retry_after)
        }
    )
```

标准 HTTP 限流响应头：
- `Retry-After: 60`：告知客户端 60 秒后重试
- `X-RateLimit-Limit: 100`：总配额
- `X-RateLimit-Remaining: 0`：剩余配额
- `X-RateLimit-Reset: 1735689600`：配额重置的 Unix 时间戳

## 多 API Key 轮换（Key Pool）

当单个 API Key 达到速率限制时，自动切换到其他 Key：

```python
import itertools
from threading import Lock

class APIKeyPool:
    def __init__(self, api_keys: list[str]):
        self.keys = api_keys
        self.key_iterator = itertools.cycle(api_keys)
        self.failed_keys = {}  # key -> cooldown_until
        self.lock = Lock()

    def get_key(self) -> str:
        """获取下一个可用的 API Key"""
        now = time.time()
        with self.lock:
            for _ in range(len(self.keys)):
                key = next(self.key_iterator)
                # 检查是否在冷却期
                if self.failed_keys.get(key, 0) < now:
                    return key
            raise AllKeysExhaustedException("所有 API Key 均处于冷却期")

    def mark_rate_limited(self, key: str, cooldown_seconds: int = 60):
        """标记某个 Key 触发了限流，进入冷却期"""
        with self.lock:
            self.failed_keys[key] = time.time() + cooldown_seconds

# 使用示例
key_pool = APIKeyPool(api_keys=[
    "sk-key-1",
    "sk-key-2",
    "sk-key-3"
])

async def call_llm_with_key_rotation(prompt: str):
    for attempt in range(3):
        key = key_pool.get_key()
        try:
            client = OpenAI(api_key=key)
            return await client.chat.completions.create(...)
        except RateLimitError as e:
            retry_after = int(e.response.headers.get("Retry-After", 60))
            key_pool.mark_rate_limited(key, retry_after)
            continue
    raise AllRetriesExhaustedException()
```

## 企业多租户限流

企业场景中，通常需要多级限流（全局 → 租户 → 用户）：

```
全局限流层（防止整体超出 API 供应商限额）
    ↓
租户限流层（按企业配额分配，如 A 企业 50%，B 企业 30%）
    ↓
用户限流层（企业内每个用户的个人配额）
```

```python
class MultiTenantRateLimiter:
    def __init__(self, redis):
        self.redis = redis

    def check(self, tenant_id: str, user_id: str, tokens: int):
        # 检查全局限额
        global_key = "global:rate:tokens"
        # 检查租户限额
        tenant_key = f"tenant:{tenant_id}:rate:tokens"
        # 检查用户限额
        user_key = f"user:{user_id}:rate:tokens"

        # 原子性地检查所有层级
        # 任何一层超限则拒绝
        ...
```

## Redis 限流实现

Redis 是生产环境分布式限流的标准选择：

```lua
-- Redis Lua 脚本：原子性令牌桶（防止竞态条件）
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1]) or capacity
local last_refill = tonumber(bucket[2]) or now

-- 计算应补充的令牌
local elapsed = now - last_refill
local new_tokens = math.min(capacity, tokens + elapsed * refill_rate)

if new_tokens >= requested then
    redis.call('HMSET', key, 'tokens', new_tokens - requested, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return 1  -- 允许
else
    redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return 0  -- 拒绝
end
```

```python
import redis

r = redis.Redis()
rate_limit_script = r.register_script(lua_script)

def is_allowed(user_id: str, tokens: int) -> bool:
    result = rate_limit_script(
        keys=[f"rate:{user_id}"],
        args=[1000, 10, tokens, time.time()]  # capacity=1000, refill=10/s
    )
    return result == 1
```

Redis 实现分布式限流的优势：
- 原子性：Lua 脚本在单个 Redis 实例上原子执行，无竞态条件
- 分布式一致：多个应用实例共享同一个 Redis，限流状态全局一致
- 高性能：Redis 单实例每秒处理 10 万+ 操作，限流开销极小
- 自动过期：通过 EXPIRE 命令自动清理过期的限流记录
