# vLLM

vLLM 是 UC Berkeley 开发的高性能 LLM 推理和服务框架，于 2023 年 6 月发布。它以 **PagedAttention** 为核心创新，解决了传统 LLM 推理中 KV Cache 内存利用率低的根本问题，在相同硬件上实现了数倍于传统方案的吞吐量。vLLM 已成为生产 LLM 服务的事实标准之一。

## vLLM 定位

vLLM 的目标用户是**需要高并发、高吞吐量 LLM 推理服务的工程团队**：

- 互联网产品团队：为 ChatBot、写作助手等高并发应用提供推理后端
- AI 平台团队：构建内部 LLM 服务平台
- 研究团队：大规模评估和基准测试
- 模型提供商：对外提供 API 服务

vLLM 不适合：轻量本地使用（用 Ollama 更方便）、边缘设备（用 llama.cpp 更适合）。

## PagedAttention 原理

PagedAttention 是 vLLM 的核心创新，解决了 LLM 推理中 KV Cache 的内存碎片问题。

### 传统 KV Cache 的问题

传统方案为每个请求预先分配**连续的**显存块，大小等于最大序列长度：

```
请求A（实际长度：100 tokens，分配：2048 tokens）
  [██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
   已使用          大量浪费（内部碎片）

请求B（实际长度：50 tokens，分配：2048 tokens）
  [█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]

结果：40-80% 的 KV Cache 显存被浪费
```

### PagedAttention 方案

类比操作系统的虚拟内存分页：

```
物理内存分为固定大小的"页"（通常 16 tokens/页）：
Page 0 [████] Page 1 [████] Page 2 [████] ...

请求A 的 KV Cache：由虚拟块映射到不同物理块
虚拟块: [0] → 物理块 Page 3
虚拟块: [1] → 物理块 Page 7
虚拟块: [2] → 物理块 Page 1

结果：几乎无内存浪费，利用率 > 95%
```

PagedAttention 的额外优势：不同请求可以**共享**相同前缀的物理块（Prefix Sharing），进一步节省内存。

实测效果：与 HuggingFace Transformers 相比，vLLM 的吞吐量提升约 **24x**（官方数据，测试条件为 LLaMA-13B，RTX 3090）。

## Continuous Batching（动态批处理）

vLLM 使用 Continuous Batching（连续批处理）而非传统的静态批处理：

**静态批处理**：等待一批请求都完成后再处理下一批，批内最短请求完成后 GPU 部分空闲。

**Continuous Batching**：每个生成步结束后，立即将完成的请求移出、将等待队列的新请求加入，GPU 始终处于满载状态。

这两个技术的组合使 vLLM 能够在高并发场景下将 GPU 利用率保持在 90%+。

## OpenAI 兼容 API

vLLM 提供完整的 OpenAI 兼容 REST API，应用代码无需修改即可迁移：

```bash
# 启动 API 服务器
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --port 8000

# 支持的端点
# POST /v1/chat/completions
# POST /v1/completions
# GET  /v1/models
# POST /v1/embeddings
```

## 安装与快速启动

### 安装

```bash
# 基本安装（NVIDIA CUDA 12.1）
pip install vllm

# 或指定 CUDA 版本
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# 从源码安装（最新特性）
pip install git+https://github.com/vllm-project/vllm.git

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

### 快速启动

```bash
# 方式1：命令行启动（最简单）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192

# 方式2：vllm serve 命令（vllm >= 0.4.0）
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --port 8000

# 方式3：Python 代码中使用
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["你好，介绍一下自己"], sampling_params)
print(outputs[0].outputs[0].text)
```

### 关键启动参数

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \           # 模型路径（本地）或 HuggingFace ID
    --dtype bfloat16 \                 # 数据类型（推荐 bfloat16）
    --max-model-len 8192 \             # 最大上下文长度（影响 KV Cache 大小）
    --gpu-memory-utilization 0.9 \     # GPU 显存利用率（留 10% 给系统）
    --tensor-parallel-size 2 \         # 张量并行度（多 GPU）
    --pipeline-parallel-size 1 \       # 流水线并行度
    --max-num-seqs 256 \               # 最大并发序列数
    --enable-prefix-caching \          # 启用前缀缓存
    --enable-chunked-prefill \         # 分块预填充（长上下文优化）
    --speculative-model /path/to/draft \ # 投机解码草稿模型
    --port 8000 \
    --host 0.0.0.0
```

## 量化支持

### AWQ 量化

```bash
vllm serve TheBloke/Llama-2-7B-AWQ \
    --quantization awq \
    --dtype auto

# AWQ 模型显存占用约为 FP16 的一半，速度接近 FP16
```

### GPTQ 量化

```bash
vllm serve TheBloke/Llama-2-7B-GPTQ \
    --quantization gptq \
    --dtype auto
```

### FP8 量化（H100 专用）

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --dtype bfloat16

# FP8 在 H100 上速度约为 BF16 的 1.5-2 倍
```

### 动态量化（INT8，无需预量化）

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype bfloat16
```

## 多 GPU 部署（Tensor Parallel）

张量并行将模型的权重矩阵水平切分到多张 GPU：

```bash
# 2 张 GPU 张量并行（模型参数在 GPU 间均等分配）
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16

# 4 张 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4

# 8 张 GPU（运行 405B 模型）
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2  # 流水线并行用于超大模型
```

多机 Tensor Parallel（Ray 集群）：

```python
# 在主节点启动 Ray
ray start --head

# 在工作节点
ray start --address='主节点IP:6379'

# 启动 vLLM（自动利用 Ray 集群中的 GPU）
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 16 \  # 跨机器 16 张 GPU
    --distributed-executor-backend ray
```

## 性能基准（vs HuggingFace Transformers）

在 LLaMA-2-7B，RTX 3090 24GB，随机输入/输出（约 512 input，512 output tokens），并发 100 请求场景下：

| 框架 | 吞吐量（requests/s） | 吞吐量（tokens/s） | P99 延迟 |
|------|-------------------|-----------------|---------|
| HuggingFace | ~0.5 | ~500 | >100s |
| vLLM | ~12 | ~12,000 | ~10s |
| vLLM + prefix cache | ~15 | ~15,000 | ~8s |

（数据仅供参考，实际性能因模型、硬件、请求分布而异）

## Docker 部署

```dockerfile
# Dockerfile
FROM vllm/vllm-openai:latest

# 或从 NVIDIA 基础镜像构建
FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN pip install vllm

EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "meta-llama/Llama-3.1-8B-Instruct", "--dtype", "bfloat16", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface  # 共享模型缓存
    ports:
      - "8000:8000"
    command:
      - "--model"
      - "meta-llama/Llama-3.1-8B-Instruct"
      - "--dtype"
      - "bfloat16"
      - "--gpu-memory-utilization"
      - "0.9"
    shm_size: "1g"
    ulimits:
      memlock: -1
```

## 最佳实践

### 显存规划

```
可用 vLLM 的 GPU 显存 = 总显存 × gpu_memory_utilization

KV Cache 显存 = 可用显存 - 模型权重显存
max_num_sequences = KV Cache 显存 / 单序列最大 KV 大小
```

推荐值：
- `--gpu-memory-utilization 0.9`（留 10% 给 CUDA 运行时）
- 根据实际并发量调整 `--max-model-len`（上下文越短，可服务的并发越多）

### 吞吐量 vs 延迟权衡

- 高吞吐（批量处理）：增大 `--max-num-seqs`，允许更多请求同时进队
- 低延迟（交互式）：减小 `--max-num-seqs`，增大 `--max-num-batched-tokens`（确保每个请求快速得到 GPU 时间）

### 启用前缀缓存

```bash
# 强烈推荐：几乎无代价，对固定 System Prompt 场景收益显著
--enable-prefix-caching
```

### 模型预热

vLLM 在收到第一个请求时触发 CUDA Graph Capture（如果启用），导致第一个请求延迟较高。在生产部署中建议在启动后立即发送几个测试请求进行预热：

```python
import requests
import time

# 服务启动后等待就绪
time.sleep(30)

# 发送预热请求
for _ in range(3):
    requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10
    })

print("预热完成，服务就绪")
```
