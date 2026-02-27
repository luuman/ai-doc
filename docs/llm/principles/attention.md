# 注意力机制详解

注意力机制（Attention Mechanism）是 Transformer 架构的核心组件，也是现代大语言模型的基础。从 2017 年"Attention Is All You Need"到今日万亿参数的 LLM，注意力机制在不断演进以应对更长上下文、更低显存和更快推理的需求。

## Scaled Dot-Product Attention 数学公式

注意力机制的核心计算如下：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

其中：

- **Q（Query）**：查询矩阵，维度 `[seq_len, d_k]`
- **K（Key）**：键矩阵，维度 `[seq_len, d_k]`
- **V（Value）**：值矩阵，维度 `[seq_len, d_v]`
- **√d_k**：缩放因子，防止内积过大导致 Softmax 梯度消失

计算步骤：

1. 计算 Q 与所有 K 的点积（相似度分数）：`QK^T`，维度 `[seq_len, seq_len]`
2. 除以 `√d_k` 进行缩放
3. 对因果语言模型（CLM）施加因果掩码（Causal Mask）：将未来位置的分数设为 `-∞`
4. Softmax 归一化为注意力权重（每行和为 1）
5. 用注意力权重对 V 加权求和，得到输出

## Query/Key/Value 的直觉理解

可以将注意力机制类比为**信息检索系统**：

- **Query（查询）**：当前位置想要"找什么"。例如，处理代词"它"时，Query 代表"它指代的实体是什么"
- **Key（键）**：每个位置对外"声明自己是什么"。序列中每个 Token 都有一个 Key，描述自身的语义特征
- **Value（值）**：每个位置实际"贡献的信息内容"。当 Query 与某个 Key 匹配度高时，对应的 Value 被更多地加入输出

注意力分数矩阵的直觉：

- 行 i 对应位置 i 的 Query
- 列 j 对应位置 j 的 Key
- 分数 `A[i][j]` 越高，位置 i 在生成时越"关注"位置 j 的信息

## 注意力分数矩阵的可视化意义

注意力分数矩阵（Attention Map）是模型内部工作机制的可视化窗口：

- **局部注意力模式**：某些 Head 主要关注相邻位置（捕捉局部语法）
- **长程依赖**：某些 Head 在处理代词时关注远处的先行词（捕捉指代关系）
- **因果掩码的效果**：下三角矩阵——每个位置只能看到自身及左侧位置，右侧的注意力权重为 0

## Multi-Head Attention（多头注意力）

单个注意力头只能捕捉一种关系模式。Multi-Head Attention 并行运行 h 个注意力头，每个头学习不同的关注方式：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O

where head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)
```

- 每个头拥有独立的投影矩阵 `W_i^Q, W_i^K, W_i^V`
- 各头的输出拼接后，经过输出投影 `W^O` 合并
- 典型配置：LLaMA 3 8B 使用 32 个头，每头维度 128（总 d_model = 4096）

多头设计的优势：

- 不同头可以同时捕捉语法关系、语义相似性、位置偏好等多种模式
- 提供更丰富的表示空间

## KV Cache：推理加速的关键

自回归生成时，每步只生成一个 Token，但需要对完整前缀计算注意力。如果每步都重新计算所有 Token 的 K 和 V，计算量将是 `O(T^2)`，随序列增长急剧膨胀。

**KV Cache** 的做法：

- 将每一步计算得到的 K 和 V 缓存起来
- 下一步生成时，只需计算新 Token 的 Q/K/V，并将新 K/V 追加到缓存
- 用缓存中所有历史 K/V 计算注意力，避免重复计算

代价：KV Cache 占用显存随序列长度线性增长。对于长上下文（100K+ Token）的场景，KV Cache 可能占用数十 GB 显存，成为实际部署的主要瓶颈。

## GQA（Grouped Query Attention）与 MQA 减少显存

标准 Multi-Head Attention 中，Q、K、V 的头数相同（均为 h 个）。为减少 KV Cache 显存占用，提出了以下变体：

### MQA（Multi-Query Attention）

- K 和 V 只有 1 个头（所有 Q 头共享同一套 K/V）
- KV Cache 减少到原来的 1/h
- 缺点：性能有所下降，训练稳定性降低

### GQA（Grouped Query Attention）

- K 和 V 有 g 个头（g 介于 1 和 h 之间），每 h/g 个 Q 头共享一套 K/V
- 在 MQA（g=1）和 MHA（g=h）之间取平衡
- LLaMA 3、Mistral、Qwen2 等现代模型均采用 GQA

具体配置示例（LLaMA 3 8B）：

```
Q heads: 32
K/V heads: 8  (GQA，每 4 个 Q 头共享 1 套 KV)
KV Cache 减少到原来的 1/4
```

## FlashAttention：硬件级优化

标准 Scaled Dot-Product Attention 存在两个 IO 瓶颈：

1. 注意力分数矩阵 `[seq_len, seq_len]` 需要写入 HBM（GPU 高带宽存储），再读回
2. 当 seq_len 较大时，矩阵尺寸为 `O(seq_len^2)`，内存占用巨大

**FlashAttention**（Dao 等，2022）通过**分块计算**（Tiling）和**在线 Softmax**技术，将注意力计算完全在 SRAM 中进行，避免了 HBM 读写：

- 时间复杂度不变，仍为 `O(seq_len^2 d)`
- 内存复杂度从 `O(seq_len^2)` 降至 `O(seq_len)`
- 实测速度提升 2-4 倍，内存减少 5-20 倍

FlashAttention 2（2023）进一步优化了并行策略和工作分配，在 A100 GPU 上达到约 73% 的理论最大吞吐量（FLOP 利用率）。FlashAttention 3（2024）针对 H100 GPU 的 FP8 精度和 Hopper 架构特性进行了专项优化。

## 长上下文挑战与 RoPE 位置编码

随着 LLM 上下文窗口从 2K 扩展到 128K、1M Token，位置编码成为核心挑战之一。

### 传统绝对位置编码的问题

Transformer 原始论文使用绝对正弦位置编码，无法外推到训练时未见过的序列长度。

### RoPE（Rotary Position Embedding）

RoPE 是目前 LLaMA、Mistral、Qwen 等主流模型采用的位置编码方案：

- 在 Q 和 K 向量上施加旋转变换，旋转角度与位置相关
- 点积 `q_i · k_j` 天然包含相对位置信息 `(i-j)` 而非绝对位置
- 相对位置编码形式使模型更容易外推到更长序列

RoPE 的外推方法：

- **NTK-aware Scaling**：调整旋转基频（base），线性缩放到更长上下文
- **YaRN（Yet another RoPE extensioN）**：非均匀频率缩放，保持高频位置信息的精确性
- **LongRoPE**：微软提出的渐进式上下文扩展，配合少量长文本微调

### 长上下文的注意力计算优化

即使有 FlashAttention，长上下文仍面临 `O(seq_len^2)` 的计算挑战：

- **稀疏注意力（Sparse Attention）**：仅计算局部窗口或选定位置的注意力
- **滑动窗口注意力（Sliding Window）**：Mistral/Mixtral 采用，每个 Token 只关注固定大小的局部窗口加少量全局 Token
- **Ring Attention**：将超长序列分布到多个 GPU，每个 GPU 仅持有序列的一段，通过通信传递 KV

长上下文 LLM 的实际性能问题——"Lost in the Middle"现象：研究发现，当关键信息位于超长上下文的中间位置时，LLM 的利用率显著下降，开头和结尾的信息更受关注。这是当前长上下文模型需要持续改进的方向。
