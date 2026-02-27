# Transformer 架构

2017年，谷歌大脑团队的论文《Attention is All You Need》提出了 Transformer 架构，以纯注意力机制完全取代 RNN，成为现代 AI 的基石。BERT、GPT、T5、ViT、Whisper、DALL-E——几乎所有顶尖 AI 模型都建立在 Transformer 之上。理解 Transformer，就是理解当代 AI 的核心语言。

## 为什么 Transformer 取代了 RNN

RNN 的根本局限：

- **顺序计算**：h_t 依赖 h_{t-1}，必须串行处理，无法并行，训练极慢
- **长程依赖**：梯度通过多步传播衰减，难以建模距离远的 token 之间的关系
- **信息瓶颈**：上下文信息压缩在固定大小的隐藏状态中

Transformer 的突破：所有 token 之间**直接建立注意力连接**，任意两个位置的距离为 1（不论实际序列距离），并行计算所有注意力权重。

## Self-Attention（自注意力）

### Q、K、V 矩阵的直觉

自注意力的核心类比是数据库查询：

- **Query（查询，Q）**：当前 token "我想找什么信息？"
- **Key（键，K）**：所有 token "我有什么信息？"
- **Value（值，V）**：所有 token "我实际携带的内容是什么？"

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

计算步骤：
1. QKᵀ：计算所有 Query-Key 对之间的相似度分数
   形状：(seq_len, d_k) × (d_k, seq_len) → (seq_len, seq_len)

2. / √d_k：缩放，防止点积结果过大导致 softmax 梯度消失
   d_k 是 Key 向量的维度

3. softmax：将分数归一化为注意力权重（每行和为 1）
   表示每个 token 对其他 token 的关注程度

4. × V：用注意力权重对 Value 加权求和，得到上下文感知的表示
   形状：(seq_len, seq_len) × (seq_len, d_v) → (seq_len, d_v)
```

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    # 注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)  # 沿 Key 维度归一化

    # 加权聚合 Value
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

Q、K、V 是输入 X 通过三个独立的线性投影矩阵得到的：

```
Q = X W_Q  (seq_len, d_model) × (d_model, d_k) → (seq_len, d_k)
K = X W_K
V = X W_V  → (seq_len, d_v)
```

## Multi-Head Attention（多头注意力）

单头注意力只能关注一种类型的依赖关系（如"主谓关系"），多头注意力并行运行多个独立的注意力头，每个头关注不同类型的信息：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O

其中 head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        # 投影并分头：(B, L, D) → (B, L, num_heads, d_k) → (B, num_heads, L, d_k)
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头：(B, num_heads, L, d_k) → (B, L, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(attn_out)
```

GPT-3（d_model=12288, num_heads=96）：每个头的维度 d_k = 12288/96 = 128，不同头可以学习语法、语义、共指消解等不同类型的语言关系。

## 位置编码（Positional Encoding）

自注意力对序列位置天然不敏感（打乱顺序不影响注意力分数集合）。位置编码将位置信息注入模型：

### 绝对位置编码（原始 Transformer）

使用正弦/余弦函数编码位置，频率随维度变化：

```python
def get_sinusoidal_encoding(max_seq_len, d_model):
    PE = torch.zeros(max_seq_len, d_model)
    pos = torch.arange(0, max_seq_len).unsqueeze(1)  # (L, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    PE[:, 0::2] = torch.sin(pos * div_term)  # 偶数维度：sin
    PE[:, 1::2] = torch.cos(pos * div_term)  # 奇数维度：cos
    return PE
```

正弦编码的优点：能推广到训练时未见过的序列长度（外推性）。

### 相对位置编码（RoPE / ALiBi）

**RoPE（旋转位置编码，Su et al.，2021）**：通过在注意力计算时旋转 Q/K 向量来编码相对位置，使注意力分数天然包含相对位置信息，外推性更好，被 LLaMA、Qwen 等主流 LLM 采用。

**ALiBi（Attention with Linear Biases）**：直接在注意力分数上添加与相对距离成比例的偏置，简洁高效，支持长上下文。

## Encoder（编码器）架构

原始 Transformer 的编码器由 N=6 个相同层堆叠组成，每层包含两个子层：

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 子层1：Multi-Head Self-Attention + 残差连接 + LayerNorm
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))  # Pre-LN 或 Post-LN

        # 子层2：Feed-Forward Network + 残差连接 + LayerNorm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
```

**Feed-Forward Network（FFN）**：两层全连接，中间层维度通常为 4×d_model（如 d_model=512，则 ff_dim=2048）。FFN 在每个位置独立应用，相当于对每个 token 做独立的特征变换。研究发现 FFN 层存储了模型的事实知识（"键值存储"假说）。

## Decoder（解码器）架构

解码器在编码器基础上增加了**Masked Multi-Head Attention**（防止看到未来 token）和**Cross-Attention**（关注编码器输出）：

```
Decoder Layer 三个子层：
1. Masked Self-Attention：只能关注当前位置及其左侧（因果掩码）
2. Cross-Attention：Query 来自 Decoder，Key/Value 来自 Encoder 输出
3. Feed-Forward Network
```

**因果掩码（Causal Mask）**：

```python
def create_causal_mask(seq_len):
    # 下三角矩阵：位置 i 只能看到 j <= i 的位置
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask
```

GPT 系列只有 Decoder（无编码器），通过因果掩码实现自回归语言建模。

## Layer Norm

Transformer 使用 **LayerNorm** 而非 BatchNorm，对每个样本在特征维度上归一化：

```python
# LayerNorm：对每个样本的 d_model 维特征归一化（与 batch size 无关）
layer_norm = nn.LayerNorm(d_model)

# 对比：BatchNorm 对每个特征在 batch 维度归一化（依赖大 batch）
batch_norm = nn.BatchNorm1d(d_model)
```

LayerNorm 不依赖 batch size，在序列长度可变、batch size 小（语言模型训练时常见）的场景下更稳定。

**Pre-LN vs Post-LN**：原始 Transformer 用 Post-LN（注意力/FFN 之后归一化），现代 LLM 几乎全部改用 Pre-LN（注意力/FFN 之前归一化），训练更稳定，不需要学习率预热的"warm-up trick"。

## BERT vs GPT：双向 Encoder vs 单向 Decoder

| 维度 | BERT | GPT |
|------|------|-----|
| 架构 | 双向 Transformer Encoder | 单向（因果）Transformer Decoder |
| 预训练任务 | MLM（掩码语言模型）+ NSP | 自回归语言建模（预测下一词） |
| 注意力 | 每个 token 关注所有其他 token | 每个 token 只关注左侧 token |
| 适合任务 | 分类、问答、NER（理解任务） | 文本生成、对话（生成任务） |
| 代表模型 | BERT、RoBERTa、DeBERTa | GPT-2/3/4、LLaMA、Claude |

**为什么双向 vs 单向？**

- BERT 的 MLM 任务需要根据上下文（左右两侧）预测被遮盖的词，因此必须双向
- GPT 的自回归生成任务（逐词预测）必须保证预测时只能看已有词，因此单向

## 《Attention is All You Need》的历史意义

Vaswani et al.（2017）的这篇论文在 AI 历史上的意义难以估量：

- **技术突破**：首次证明纯注意力机制（无 RNN/CNN）在机器翻译上超越当时最优 LSTM 模型
- **并行训练**：Transformer 的并行特性完美契合 GPU 算力，使训练数十亿参数的模型成为可能
- **统一架构**：Transformer 逐步扩展到视觉（ViT）、语音（Wav2Vec）、多模态（CLIP、GPT-4o），成为跨模态的统一架构
- **扩展定律的基础**：Transformer 的扩展性（参数量 × 数据量 → 性能提升）催生了大模型时代

这篇 8 位作者撰写的 15 页论文，直接开创了 ChatGPT、Claude、Gemini 的时代。
