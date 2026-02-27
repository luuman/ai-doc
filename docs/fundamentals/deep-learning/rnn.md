# 循环神经网络（RNN）

循环神经网络（Recurrent Neural Network，RNN）是为处理序列数据而设计的神经网络架构。文本、语音、时间序列、视频帧——这些数据的显著特点是**顺序依赖性**：理解当前元素需要参考前面的上下文。RNN 通过隐藏状态（Hidden State）在时间步之间传递信息，是 LSTM、GRU 和现代 Transformer 注意力机制的前身。

## 序列数据建模需求

传统全连接网络的局限：

- **固定输入长度**：无法处理变长序列（不同长度的句子、不等长时间序列）
- **无上下文记忆**：处理"The cat, which had been sitting on the mat, **was**..."时，无法记住句子开头的主语
- **参数不共享**：同一个词在不同位置需要不同参数，参数量随序列长度爆炸

RNN 的核心思想：**参数共享 + 隐藏状态传递**。同一组参数 W 在每个时间步复用，通过隐藏状态 h_t 携带历史信息。

## RNN 基本结构

### 展开计算图

```
h_t = σ(W_h · h_{t-1} + W_x · x_t + b_h)
y_t = W_y · h_t + b_y

其中：
- x_t：t 时刻的输入（如词向量）
- h_t：t 时刻的隐藏状态（"记忆"）
- y_t：t 时刻的输出（可选）
- W_h, W_x, W_y, b_h, b_y：可学习参数（所有时间步共用）
- σ：非线性激活函数（通常 tanh 或 ReLU）
```

将 RNN 按时间步展开，等价于一个非常深的前馈网络（每个时间步对应一层），但所有层**共用同一套参数**。

```python
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2,
             batch_first=True, dropout=0.3)

# 输入序列：(batch=32, seq_len=100, input_size=128)
x = torch.randn(32, 100, 128)
h0 = torch.zeros(2, 32, 256)  # (num_layers, batch, hidden_size)
output, h_n = rnn(x, h0)
# output: (32, 100, 256)，每个时间步的隐藏状态
# h_n: (2, 32, 256)，最后时间步的隐藏状态
```

### 不同输入/输出配置

RNN 的灵活性在于可以处理多种序列任务：

- **一对多（One-to-Many）**：图像描述生成（输入单张图像，输出描述句子）
- **多对一（Many-to-One）**：情感分析（输入句子，输出正/负标签）
- **多对多（同步）**：词性标注（每个词输出对应词性）
- **多对多（异步/seq2seq）**：机器翻译（输入源语言句子，输出目标语言句子）

## 梯度消失与梯度爆炸问题

RNN 在处理长序列时面临严重的训练困难，根源在于 BPTT（Backpropagation Through Time）中的梯度连乘：

```
∂L/∂h_0 = ∂L/∂h_T · ∏_{t=1}^{T} ∂h_t/∂h_{t-1}
         = ∂L/∂h_T · ∏_{t=1}^{T} (W_h · diag(σ'(z_t)))
```

- **梯度消失**：若 `||∂h_t/∂h_{t-1}|| < 1`，乘积 T 次后梯度指数衰减→0，网络遗忘早期信息
- **梯度爆炸**：若 `||∂h_t/∂h_{t-1}|| > 1`，乘积 T 次后梯度指数增长，训练发散

应对策略：
- 梯度爆炸：梯度裁剪（Gradient Clipping），将梯度范数限制在阈值以内
- 梯度消失：**LSTM/GRU** 架构（从根本上解决），残差连接

## LSTM——长短期记忆网络

LSTM（Long Short-Term Memory，Hochreiter & Schmidhuber，1997）通过精心设计的门控机制，选择性地记住或忘记信息：

### LSTM 的核心创新：细胞状态（Cell State）

LSTM 引入**细胞状态 c_t**作为长期记忆的载体。细胞状态直接贯穿整个序列，与隐藏状态并行流动，梯度可以通过它直接回流到早期时间步（类似 ResNet 的残差连接）。

### 四个门控机制详解

```
给定输入：[h_{t-1}, x_t] （拼接上一时刻隐藏状态和当前输入）

1. 遗忘门（Forget Gate）：决定从细胞状态中丢弃什么
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   → 输出 0-1 之间的值；0 = "完全遗忘"，1 = "完全保留"

2. 输入门（Input Gate）：决定向细胞状态中写入什么
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   ← 控制写入比例
   g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) ← 候选新内容

3. 更新细胞状态：
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
   （遗忘部分旧内容 + 写入部分新内容，⊙ 为逐元素乘法）

4. 输出门（Output Gate）：决定输出什么
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t ⊙ tanh(c_t)
```

```python
lstm = nn.LSTM(input_size=128, hidden_size=256,
               num_layers=2, batch_first=True, dropout=0.3)

x = torch.randn(32, 100, 128)
h0 = torch.zeros(2, 32, 256)
c0 = torch.zeros(2, 32, 256)  # LSTM 额外维护细胞状态

output, (h_n, c_n) = lstm(x, (h0, c0))
```

**LSTM 为什么有效**：细胞状态 c_t 的更新是加法而非乘法（相比纯 RNN），梯度可以直接通过加法路径流回早期时间步，解决了长期梯度消失问题。这与 ResNet 残差连接的思想异曲同工。

## GRU——简化版 LSTM

GRU（Gated Recurrent Unit，Cho et al.，2014）是 LSTM 的简化版本，将遗忘门和输入门合并为**更新门**，去掉细胞状态（只保留隐藏状态）：

```
z_t = σ(W_z · [h_{t-1}, x_t])    ← 更新门（合并遗忘+输入）
r_t = σ(W_r · [h_{t-1}, x_t])    ← 重置门
g_t = tanh(W_g · [r_t ⊙ h_{t-1}, x_t])  ← 候选隐藏状态
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ g_t
```

GRU 参数更少（约 LSTM 的 3/4），训练更快，在中短序列上性能与 LSTM 相当，是资源受限场景的优选。

## 双向 RNN（Bidirectional RNN）

单向 RNN 只能利用过去的上下文。双向 RNN 分别用前向和后向 RNN 处理序列，并将两个方向的隐藏状态拼接：

```python
birnn = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
output, _ = birnn(x)
# output 形状: (batch, seq_len, 256*2=512)
# 每个位置都包含了左侧和右侧的上下文信息
```

BERT 通过双向 Transformer（等价于双向 RNN 的注意力机制版本）同时利用全部上下文，是其强大的词义表示能力的关键。

## Seq2Seq 架构——编码器-解码器

Seq2Seq（Sequence to Sequence，Sutskever et al.，2014）是机器翻译、文本摘要、对话系统的基础框架：

```
源序列 x₁,...,x_m → 编码器（Encoder LSTM）→ 语义向量 c
                                                        ↓
目标序列 y₁,...,y_n ← 解码器（Decoder LSTM）← c（作为初始隐藏状态）
```

编码器将整个源序列压缩为固定长度的上下文向量 c，解码器以 c 为初始状态逐步生成目标序列。

**瓶颈问题**：固定长度的向量 c 难以捕获长序列的全部信息（100词的句子压缩为256维向量信息损失严重）。

## 注意力机制的引入背景

为解决 Seq2Seq 的瓶颈问题，Bahdanau et al.（2014）提出了注意力机制：

```
解码器在每一步生成输出时，不依赖固定的上下文向量 c，
而是动态地"关注"编码器各时间步的输出：

注意力分数：e_ij = score(h_i^dec, h_j^enc)
注意力权重：α_ij = softmax(e_ij)
上下文向量：c_i = Σ_j α_ij · h_j^enc  ← 每步都重新计算！
```

注意力机制允许解码器直接访问编码器所有位置的表示，并动态决定哪些位置最重要。这一思想被 Transformer 发展为完全的自注意力机制，彻底取代了 RNN 的序列依赖结构。
