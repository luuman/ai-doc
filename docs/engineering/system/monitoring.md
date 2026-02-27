# AI 系统监控

传统软件的监控（CPU、内存、错误率）不足以全面反映 AI 系统的健康状况。LLM 系统需要额外关注输出质量、Token 成本、Prompt 效果等 AI 特有维度。完善的可观测性（Observability）是持续改进 AI 产品的基础。

## LLM 可观测性三要素

借鉴传统可观测性的"三支柱"理念，LLM 系统的可观测性同样包含三个层次：

### Trace（链路追踪）

Trace 记录单个请求的完整处理链路，对于 RAG 和 Agent 等多步骤系统尤为重要：

```
用户请求
  └─ 意图识别（100ms）
  └─ 向量检索（50ms，返回3个文档）
  └─ Prompt 组装（10ms）
  └─ LLM 调用（1200ms，input: 1024 tokens, output: 256 tokens）
  └─ 输出解析（5ms）
  总耗时: 1365ms，总成本: $0.003
```

每个 Trace 应包含：
- 请求 ID（贯穿全链路）
- 各步骤的开始时间和耗时
- 输入/输出的摘要（非完整内容，注意隐私）
- Token 消耗和成本
- 工具调用详情

### Metric（指标）

持续监控的数值型指标，用于发现趋势和设置告警：

- **延迟指标**：TTFT（首 Token 延迟）、总生成时间、P50/P95/P99 分位数
- **吞吐量指标**：TPS（每秒 Token 数）、RPS（每秒请求数）
- **成本指标**：每请求成本、日/月 Token 消耗
- **质量指标**：用户满意率（点赞率）、重生成率、对话轮数
- **可用性指标**：API 错误率（5xx）、超时率、降级触发率

### Log（日志）

结构化日志记录请求和响应的关键信息：

```json
{
    "timestamp": "2024-01-15T10:23:45Z",
    "request_id": "req-abc123",
    "user_id": "user-xyz",
    "model": "claude-opus-4-5",
    "input_tokens": 512,
    "output_tokens": 128,
    "latency_ms": 1350,
    "ttft_ms": 450,
    "cost_usd": 0.002,
    "finish_reason": "stop",
    "error": null,
    "session_id": "sess-def456"
}
```

## 关键指标

### 延迟指标

| 指标 | 定义 | 目标值 | 告警阈值 |
|------|------|-------|---------|
| TTFT P50 | 50th percentile 首 Token 时间 | < 500ms | > 1000ms |
| TTFT P99 | 99th percentile 首 Token 时间 | < 2000ms | > 5000ms |
| TPOT P50 | 每 Token 生成时间（中位数） | < 50ms | > 100ms |
| 请求总时长 P99 | 99th percentile 端到端延迟 | < 30s | > 60s |

### 吞吐量指标

- **TPS（Token Per Second）**：整个服务每秒生成的总 Token 数，反映 GPU 利用效率
- **RPS（Requests Per Second）**：每秒处理的请求数
- **队列等待时间**：请求在队列中等待的时间，队列积压是扩容信号

### 成本指标

```python
# 成本追踪的基本框架
class CostMetrics:
    def record(self, model: str, input_tokens: int, output_tokens: int):
        price = MODEL_PRICES[model]
        cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

        # 发送到 Prometheus / Grafana
        cost_counter.labels(model=model).inc(cost)
        input_token_counter.labels(model=model).inc(input_tokens)
        output_token_counter.labels(model=model).inc(output_tokens)
```

## LangSmith 调试追踪

LangSmith 是 LangChain 团队开发的 LLM 调试和评估平台：

```python
from langsmith import Client, traceable
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-ai-app"

@traceable(name="chat_with_rag")
def chat_with_rag(query: str) -> str:
    # 所有在此函数内的 LLM 调用都会被自动追踪
    docs = retrieve_documents(query)
    prompt = build_prompt(query, docs)
    response = llm.invoke(prompt)
    return response.content

# 调用后，LangSmith UI 中可以看到：
# - 完整的 Prompt（System + Context + User）
# - 模型响应
# - Token 消耗
# - 延迟
# - 嵌套调用链
```

LangSmith 的核心功能：
- **Playground**：在 UI 中直接调试 Prompt，无需改代码
- **Dataset**：建立评估数据集
- **Evaluation**：批量运行评估，对比不同 Prompt 版本
- **Human Annotation**：人工标注追踪记录
- **自动化测试**：CI/CD 集成，每次 Prompt 变更都运行回归测试

## Langfuse（开源监控）

Langfuse 是 LangSmith 的开源替代品，支持私有部署：

```python
from langfuse import Langfuse
from langfuse.decorators import observe

langfuse = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com"  # 或自托管地址
)

@observe()  # 自动追踪此函数
def process_request(user_id: str, query: str):
    # 追踪用户输入
    langfuse.trace(
        name="user_request",
        user_id=user_id,
        input=query,
        metadata={"version": "v2.1"}
    )

    response = call_llm(query)

    # 追踪输出和成本
    langfuse.generation(
        name="llm_call",
        model="claude-opus-4-5",
        input=query,
        output=response.content,
        usage={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens
        }
    )

    return response.content

# 记录用户反馈（点赞/点踩）
def record_feedback(trace_id: str, score: int):
    langfuse.score(
        trace_id=trace_id,
        name="user_feedback",
        value=score  # 1=好评，0=差评
    )
```

Langfuse vs LangSmith：
- **数据主权**：Langfuse 支持自托管，敏感数据不出境
- **成本**：Langfuse 开源版免费，LangSmith 免费版有限额
- **功能**：LangSmith 功能更完整（尤其是评估），Langfuse 更轻量

## Arize Phoenix

Arize Phoenix 是专注于 AI 可观测性的开源工具，特别适合 RAG 系统的调试：

```python
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor

# 启动 Phoenix UI（本地）
session = px.launch_app()

# 自动插桩 LangChain
LangChainInstrumentor().instrument()

# 之后的 LangChain 调用自动被追踪
chain.invoke({"query": "什么是 RAG？"})

# Phoenix UI 中查看：
# - 检索到的文档和相关度分数
# - Prompt 的完整内容
# - Embedding 的可视化（t-SNE）
# - 失败案例分析
```

## Prompt 版本追踪

Prompt 的频繁迭代需要版本管理，避免"不知道上次改了什么"的混乱：

```python
# 使用 LangSmith Hub 管理 Prompt 版本
from langchain import hub

# 拉取特定版本的 Prompt
prompt = hub.pull("my-org/rag-prompt:v2.3")

# 推送新版本
hub.push("my-org/rag-prompt", new_prompt, new_tags=["v2.4", "production"])
```

Prompt 版本管理最佳实践：
- 使用语义化版本号（v1.0.0 → 功能变更，v1.1.0 → 优化，v1.0.1 → 修复）
- 每次 Prompt 变更记录变更原因和预期效果
- 在测试集上评估新版本后再上线
- 保留至少 3 个历史版本，支持快速回滚

## 输出质量评估（LLM-as-Judge）

使用一个更强的模型（或相同模型）来评估另一个模型的输出质量：

```python
from anthropic import Anthropic

judge_client = Anthropic()

JUDGE_PROMPT = """你是一个专业的 AI 输出质量评估专家。请评估以下 AI 回答的质量。

用户问题：{question}
AI 回答：{answer}

请从以下维度评分（1-5分）：
1. 准确性：回答是否正确、事实是否有误
2. 相关性：回答是否切题
3. 完整性：是否充分回答了问题
4. 简洁性：是否避免了不必要的冗余

请以 JSON 格式输出：
{{"accuracy": 分数, "relevance": 分数, "completeness": 分数, "conciseness": 分数, "overall": 综合评分, "feedback": "简要反馈"}}
"""

def evaluate_response(question: str, answer: str) -> dict:
    response = judge_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(question=question, answer=answer)
        }]
    )
    return json.loads(response.content[0].text)

# 批量评估（监控日常输出质量）
def daily_quality_audit(sample_size: int = 100):
    samples = get_random_requests_from_yesterday(sample_size)
    scores = []
    for sample in samples:
        score = evaluate_response(sample.question, sample.answer)
        scores.append(score)

    avg_overall = sum(s["overall"] for s in scores) / len(scores)
    if avg_overall < 3.5:
        alert("输出质量下降！平均分: {avg_overall:.2f}")
```

## 告警设置

关键告警规则：

```yaml
# Prometheus AlertManager 配置示例
groups:
  - name: llm_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(llm_errors_total[5m]) / rate(llm_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "LLM 错误率过高：{{ $value | humanizePercentage }}"

      - alert: HighCost
        expr: increase(llm_cost_usd_total[1h]) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "过去1小时 LLM 成本超过 $100"

      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(llm_ttft_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "首 Token 延迟 P99 超过 5 秒"
```

## 用户反馈收集

用户反馈是最直接的输出质量信号：

```typescript
// 前端：点赞/点踩按钮
const FeedbackButtons = ({ messageId }: { messageId: string }) => {
    const sendFeedback = async (value: 1 | -1) => {
        await fetch("/api/feedback", {
            method: "POST",
            body: JSON.stringify({ messageId, value, timestamp: Date.now() })
        });
    };

    return (
        <div>
            <button onClick={() => sendFeedback(1)}>👍</button>
            <button onClick={() => sendFeedback(-1)}>👎</button>
        </div>
    );
};
```

除显式反馈外，以下隐式信号也很有价值：
- **重新生成率**：用户点击"重新生成"表示对当前回答不满意
- **复制率**：用户复制了回答的内容，表示认可
- **对话继续率**：在AI回答后继续对话，表示交互有效
- **会话时长**：较长的会话通常表示用户在有效使用

## A/B 测试框架

系统性地评估 Prompt 或模型变更的效果：

```python
class ABTestFramework:
    def __init__(self, experiment_name: str, variants: dict, traffic_split: dict):
        """
        variants: {"control": prompt_v1, "treatment": prompt_v2}
        traffic_split: {"control": 0.5, "treatment": 0.5}
        """
        self.name = experiment_name
        self.variants = variants
        self.traffic_split = traffic_split

    def get_variant(self, user_id: str) -> str:
        """基于用户 ID 的稳定分桶"""
        import hashlib
        hash_val = int(hashlib.sha256(
            f"{self.name}:{user_id}".encode()
        ).hexdigest(), 16)
        bucket = (hash_val % 1000) / 1000.0  # 0.0 - 1.0

        cumulative = 0.0
        for variant, ratio in self.traffic_split.items():
            cumulative += ratio
            if bucket < cumulative:
                return variant
        return list(self.variants.keys())[-1]

# A/B 测试统计显著性检验
from scipy import stats

def check_significance(control_scores: list, treatment_scores: list) -> dict:
    t_stat, p_value = stats.ttest_ind(control_scores, treatment_scores)
    return {
        "control_mean": sum(control_scores) / len(control_scores),
        "treatment_mean": sum(treatment_scores) / len(treatment_scores),
        "p_value": p_value,
        "significant": p_value < 0.05,  # 95% 置信度
        "improvement": (sum(treatment_scores)/len(treatment_scores) -
                       sum(control_scores)/len(control_scores)) /
                      (sum(control_scores)/len(control_scores))
    }
```
