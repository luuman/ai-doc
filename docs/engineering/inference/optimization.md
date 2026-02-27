# 推理优化

模型训练完成后，如何高效地为用户提供服务成为工程的核心挑战。推理优化追求的目标与训练不同，它需要在极低延迟的前提下，实现尽可能高的系统吞吐量，同时控制单次请求的成本。本章介绍推理优化的核心技术体系。

## 推理优化目标

推理性能有三个核心维度，三者之间存在内在张力：

- **首 Token 延迟（TTFT，Time To First Token）**：从用户发送请求到收到第一个 Token 的时间。对于交互式应用（对话、代码补全），TTFT 直接影响用户体验，目标通常是 500ms 以内
- **每 Token 生成时间（TPOT，Time Per Output Token）**：生成每个后续 Token 的时间，影响长文本的总生成时长
- **吞吐量（Throughput）**：系统每秒能处理的总 Token 数（TPS，Tokens Per Second），影响服务成本

增大批次大小（Batch Size）可以提升吞吐量，但会增加 TTFT；减小批次大小可以降低延迟，但降低了 GPU 利用率。现代推理优化的核心是打破这一传统权衡。

## KV Cache（键值缓存）

KV Cache 是推理优化中最基础也最重要的技术，理解它需要先理解 Transformer 的自注意力机制：

在自注意力中，每个 Token 需要与序列中所有之前的 Token 计算注意力分数。对于长度为 N 的序列，第 N 个 Token 的注意力计算涉及前 N-1 个 Token 的 K（Key）和 V（Value）矩阵。

**没有 KV Cache 的情况**：每生成一个新 Token，都需要重新计算所有历史 Token 的 K/V，计算量为 O(N²)，严重浪费。

**有 KV Cache 的情况**：将历史 Token 的 K/V 矩阵缓存起来，生成新 Token 时只需计算新 Token 的 K/V 并与缓存拼接。每步计算量降为 O(N)，整体计算复杂度从 O(N³) 降为 O(N²)。

KV Cache 的显存占用估算：

```
KV Cache 大小 = 2 × num_layers × num_heads × head_dim × seq_len × batch_size × dtype_bytes
```

以 LLaMA-7B（32 层，32 头，head_dim=128，BF16）为例，处理 4096 Token 序列的 KV Cache 约为：
2 × 32 × 32 × 128 × 4096 × 2 bytes ≈ 2 GB

这意味着 KV Cache 随序列长度线性增长，是长上下文推理的主要显存瓶颈。

## Prefix Caching（共享前缀缓存）

在实际应用中，许多请求共享相同的前缀（System Prompt、Few-shot 示例、文档内容）。Prefix Caching 将这些共享前缀的 KV Cache 计算一次，在多个请求间复用：

应用场景：
- **固定 System Prompt**：所有请求都有相同的系统提示，只需计算一次
- **RAG 场景**：同一文档被多次查询，文档的 KV Cache 可复用
- **Few-shot 示例**：固定的示例部分只需缓存一次

Anthropic Claude 的 Prompt Caching 功能、OpenAI 的 Context Caching 都是 Prefix Caching 的商业实现。

vLLM 通过 `enable_prefix_caching=True` 启用此功能，SGLang 的 RadixAttention 是更高级的实现，支持树状前缀复用。

## 投机解码（Speculative Decoding）

大语言模型的推理是顺序生成的，每生成一个 Token 都需要完整的前向传播，无法并行化。投机解码通过引入一个小型"草稿"模型来打破这一限制：

**工作流程**：

1. **草稿阶段**：用小模型（Draft Model，如 7B）快速生成 k 个候选 Token（例如 k=4）
2. **验证阶段**：用大模型（Target Model，如 70B）并行验证这 k 个 Token（一次前向传播）
3. **接受/拒绝**：大模型从左到右检查每个草稿 Token 是否与自身分布一致；接受的 Token 保留，遇到不一致的 Token 则拒绝后续所有草稿
4. **补充采样**：在最后一个被接受的 Token 后，大模型生成一个 Token（保证每次验证至少贡献 1 个 Token）

关键性质：投机解码**不改变输出分布**，大模型的采样结果与不使用投机解码完全等价。

加速效果取决于草稿模型的接受率（Acceptance Rate），接受率越高加速越明显。实践中，7B+70B 组合在代码生成任务上可达 2-3 倍加速，在随机文本生成上效果有限。

Self-Speculative Decoding 的变体使用同一个模型的早期层（Early Exit）作为草稿模型，无需额外加载小模型。

## Continuous Batching（连续批处理）

传统静态批处理的问题：一批请求中最长的序列完成前，较短序列虽然已完成，但其 GPU 资源仍被占用（填充 Padding），GPU 利用率低。

Continuous Batching 将推理服务器的调度粒度从"请求"降低到"迭代步"：

- 每个生成步结束后，检查哪些序列已经生成了 EOS Token（结束标记）
- 将已完成的序列从批次中移除，将等待队列中的新请求插入空出的位置
- 不同长度的请求动态组合，始终保持 GPU 高利用率

这种方式的实现依赖 **PagedAttention**（vLLM 提出）：

- 将 KV Cache 划分为固定大小的"块"（Block，通常 16 或 32 Token/块）
- 每个序列的 KV Cache 由不连续的块组成（类似操作系统的虚拟内存分页）
- 不同序列的 KV Cache 块可以动态分配和回收，支持 Continuous Batching

## 批处理大小对吞吐量的影响

大语言模型推理分为两个阶段：

- **Prefill（预填充）**：处理输入 Prompt，计算密集型（Compute-bound），大批次效率高
- **Decode（解码）**：逐步生成 Token，内存带宽密集型（Memory-bound），增大批次可以摊薄内存带宽成本

对于 Decode 阶段，每一步只生成 1 个 Token，但需要从显存加载完整的模型权重。单个请求时，GPU 利用率极低（内存带宽是瓶颈，算力远未饱和）。增大批次大小可以同时处理多个请求，使算力和内存带宽都得到充分利用。

实践建议：
- 交互式场景：优先保证 TTFT，批次大小适中（8-32）
- 离线批量处理：最大化批次大小，提升吞吐量
- 自动批处理调度：vLLM、TGI 等框架会自动寻找最优批次大小

## CUDA Graph 减少 CPU 开销

在小批次、短序列的推理场景下，CUDA Kernel 的启动开销（CPU 端的 Python→CUDA 调用延迟）可能成为瓶颈，占总推理时间的 10-30%。

CUDA Graph 的工作原理：

1. **捕获阶段**：运行一次"热身"推理，记录所有 CUDA Kernel 的调用序列（Kernel 参数、内存地址）
2. **执行阶段**：后续推理直接重放已录制的 CUDA Graph，绕过 Python 层的 Kernel 启动开销

限制：CUDA Graph 假设每次推理的 Kernel 序列和内存布局完全相同，因此**只适用于固定形状（batch size, seq len）的推理**。

vLLM 和 TGI 对固定形状的批次自动使用 CUDA Graph，对动态形状（如 Continuous Batching 的变长序列）则使用普通 CUDA 调用。

典型加速效果：小批次（batch size = 1）时，CUDA Graph 可减少约 30-50% 的延迟。

## 综合优化效果

将以上技术组合使用：

| 技术 | 主要收益 | 场景 |
|------|---------|------|
| KV Cache | 避免重复计算 | 所有推理场景（必须启用） |
| Prefix Caching | 减少 TTFT | System Prompt 固定的场景 |
| Continuous Batching | 提升 GPU 利用率 | 高并发服务场景 |
| 投机解码 | 提升解码速度 | 强调延迟、草稿模型接受率高的场景 |
| CUDA Graph | 减少 CPU 开销 | 小批次、低延迟场景 |

vLLM 等现代推理框架已将上述大多数优化集成，开箱即用，用户无需手动实现。
