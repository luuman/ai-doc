# 模型服务化

将训练好的大语言模型转化为稳定、高效、可扩展的线上服务，是 AI 工程的核心环节。本章介绍主流的 LLM 推理服务框架、部署方案以及生产级服务设计。

## vLLM

vLLM 是 UC Berkeley 开发的高性能 LLM 推理服务框架，自 2023 年发布以来迅速成为生产部署的首选方案。

### PagedAttention 原理

vLLM 的核心创新是 **PagedAttention**，灵感来自操作系统的虚拟内存分页机制：

**传统 KV Cache 的问题**：
- 预先为每个请求分配最大序列长度的连续显存块
- 大量显存因"碎片化"和"预留"而浪费
- 实际显存利用率通常只有 20-40%

**PagedAttention 方案**：
- 将 KV Cache 划分为固定大小的**物理块**（Block，通常 16 Token/块）
- 每个序列的 KV Cache 由虚拟块到物理块的映射表维护
- 物理块按需动态分配，无需连续
- 不同请求可以共享相同前缀的物理块（Prefix Sharing）

效果：显存利用率从 20-40% 提升至 90%+ ，吞吐量提升 2-4 倍。

### 快速启动

```bash
pip install vllm

# 启动 OpenAI 兼容 API 服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    --port 8000
```

```python
# 使用 OpenAI SDK 调用
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "你好"}]
)
```

### 高并发支持

vLLM 内置 Continuous Batching，自动将多个并发请求合并处理。关键参数：

- `--max-num-seqs`：最大并发序列数（默认 256）
- `--max-num-batched-tokens`：每步最大 Token 数（影响内存和吞吐量）
- `--enable-prefix-caching`：启用前缀缓存
- `--speculative-model`：指定投机解码的草稿模型

## TGI（Text Generation Inference）

HuggingFace 的 Text Generation Inference 是另一个主流推理框架：

```bash
docker run --gpus all -p 8080:80 \
    -v $HOME/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --max-batch-prefill-tokens 4096 \
    --max-total-tokens 8192
```

TGI 的特点：
- 原生支持 HuggingFace Hub 模型直接加载
- 内置流式生成、Token 流式传输
- 支持 GPTQ、AWQ、EETQ 量化格式
- Speculative Decoding（Medusa 头部方案）
- 与 HuggingFace 生态深度集成

## SGLang（结构化生成加速）

SGLang（Structured Generation Language）是斯坦福开发的推理框架，专注于**结构化输出**（JSON、正则表达式约束）和多步骤推理的加速：

核心特性：
- **RadixAttention**：KV Cache 的 Radix 树管理，支持高效前缀共享
- **Constraint Decoding**：JSON Schema 约束生成，比后处理约束更高效
- **Continuous Batching**：类似 vLLM 的动态批处理
- **SGLang 语言**：Python DSL，表达多步骤 LLM 程序（Prompt + 条件分支 + 并行调用）

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, document, questions):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(f"Document: {document}")
    for q in questions:
        s += sgl.user(q)
        s += sgl.assistant(sgl.gen("answer", max_tokens=200))
    return s
```

SGLang 在需要 JSON 输出或多步骤推理的场景下，性能通常优于 vLLM。

## Ollama（本地轻量服务）

Ollama 是面向开发者本地环境的轻量级 LLM 服务工具：

```bash
# 安装并启动
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# 拉取并运行模型
ollama pull llama3.1:8b
ollama run llama3.1:8b

# OpenAI 兼容 API
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "llama3.1:8b", "messages": [{"role": "user", "content": "你好"}]}'
```

Ollama 的定位是快速本地运行，不适合生产高并发场景。

## OpenAI 兼容 API 标准

OpenAI API 格式已成为 LLM 服务的事实标准，主要端点：

```
POST /v1/chat/completions  # 对话补全
POST /v1/completions       # 文本补全（Legacy）
GET  /v1/models            # 列出可用模型
POST /v1/embeddings        # 文本嵌入
```

Chat Completions 请求格式：

```json
{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请解释量子纠缠"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000,
    "stream": true
}
```

遵循 OpenAI 兼容标准意味着：可以使用任何 OpenAI SDK（Python/JS/Go）无缝切换后端，无需修改应用代码。

## Docker 部署示例

```dockerfile
# Dockerfile
FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN pip install vllm

EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/models/llama3-8b", \
     "--dtype", "bfloat16", \
     "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0,1
    volumes:
      - /path/to/models:/models
    ports:
      - "8000:8000"
    command: >
      --model /models/llama3-8b
      --tensor-parallel-size 2
      --gpu-memory-utilization 0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

## 水平扩展（多实例负载均衡）

单实例 vLLM 的吞吐量受 GPU 数量限制。水平扩展通过多实例负载均衡来提升整体吞吐：

```nginx
# Nginx 负载均衡配置
upstream vllm_backend {
    least_conn;  # 最少连接数算法，适合推理延迟不均的场景
    server vllm-instance-1:8000;
    server vllm-instance-2:8000;
    server vllm-instance-3:8000;
    server vllm-instance-4:8000;
}

server {
    listen 80;
    location /v1/ {
        proxy_pass http://vllm_backend;
        proxy_buffering off;      # 流式输出必须关闭缓冲
        proxy_read_timeout 300s;  # 长文本生成需要更长超时
    }
}
```

注意：流式推理（Streaming）需要关闭 Nginx 缓冲（`proxy_buffering off`），否则用户无法实时收到 Token。

## 自动扩缩容（K8s HPA）

在 Kubernetes 环境下，可以基于 GPU 利用率或请求队列长度自动调整 vLLM 实例数：

```yaml
# HPA 配置（基于自定义指标）
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 1
  maxReplicas: 8
  metrics:
  - type: External
    external:
      metric:
        name: vllm_queue_length  # 自定义指标：待处理请求队列长度
      target:
        type: AverageValue
        averageValue: "10"
```

扩缩容的挑战：LLM 实例启动时间较长（加载模型权重通常需要 30-120 秒），因此 HPA 需要配置较长的稳定窗口，避免频繁扩缩。建议：
- `scaleUpStabilizationWindowSeconds: 30`（快速扩容）
- `scaleDownStabilizationWindowSeconds: 300`（缓慢缩容，避免抖动）

## 服务 SLA 设计

LLM 服务的 SLA（Service Level Agreement）需要考虑 LLM 特有的延迟特性：

| 指标 | 建议目标 | 说明 |
|------|---------|------|
| TTFT P50 | < 500ms | 交互式场景的基准 |
| TTFT P99 | < 2000ms | 高负载下的保障 |
| TPOT P50 | < 50ms/token | 流畅阅读体验（约 20 token/s） |
| 可用性 | 99.9% | 每月允许 ~43 分钟故障 |
| 错误率 | < 0.1% | 5xx 错误比例 |

设计建议：
- 设置合理的 `max_tokens` 上限，防止单请求占用过长时间
- 为不同优先级请求设置独立队列（如付费用户 vs 免费用户）
- 超时后返回已生成的部分内容，而非空响应
- 实现请求 ID 追踪，便于问题排查
