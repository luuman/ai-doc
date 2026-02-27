# 神经网络基础

人工神经网络（Artificial Neural Network，ANN）是深度学习的基础构件，灵感来源于生物神经系统但并非其精确模拟。理解神经网络的基本组件——神经元、激活函数、层次结构、反向传播——是掌握所有现代 AI 模型（CNN、RNN、Transformer）的前提。

## 从感知机到多层感知机

### 感知机（Perceptron）

感知机是神经网络的最小单元，由弗兰克·罗森布拉特于 1957 年提出：

```
输出 = sign(Σᵢ wᵢxᵢ + b) = sign(wᵀx + b)

- xᵢ：输入特征
- wᵢ：连接权重（可学习）
- b：偏置项（可学习）
- sign：符号函数（输出 +1 或 -1）
```

感知机只能解决线性可分问题（如 AND、OR），无法解决 XOR 等非线性问题（明斯基和帕珀特，1969）。

### 多层感知机（MLP）

通过堆叠多个感知机层（引入非线性激活函数），MLP 可以表示任意复杂的函数（通用逼近定理）：

```
输入层 → 隐藏层1 → 隐藏层2 → ... → 输出层

每层计算：
h⁽ˡ⁾ = σ(W⁽ˡ⁾ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)
```

```python
import torch.nn as nn

mlp = nn.Sequential(
    nn.Linear(784, 512),   # MNIST 输入：28×28=784 像素
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),    # 10 类输出
)
```

## 激活函数——引入非线性

没有激活函数，多层线性变换等价于单层线性变换，深度没有意义。激活函数引入非线性，使网络能表示复杂函数。

### Sigmoid

```
σ(x) = 1 / (1 + e⁻ˣ)   输出范围：(0, 1)
σ'(x) = σ(x)(1 - σ(x))  最大梯度 0.25（在 x=0 处）
```

问题：**梯度消失**——深层网络中 Sigmoid 梯度最大仅 0.25，经过多层连乘后梯度趋近于零。现在主要用于二分类输出层。

### ReLU（Rectified Linear Unit）

```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x>0, 0 if x≤0
```

ReLU 的优点：
- 梯度在正区间恒为 1，完全避免饱和区的梯度消失
- 计算极为高效（只是取最大值）
- 稀疏激活（约 50% 神经元输出为零），类似生物神经元的稀疏特性

问题：**神经元死亡**（Dead ReLU）——学习率过大时，某些神经元权重更新到负数区域后，梯度永远为零，该神经元永久停止学习。

### GELU 与 Swish

GELU（Gaussian Error Linear Unit）是 BERT、GPT 等现代 Transformer 的标准激活函数：

```
GELU(x) = x · Φ(x)    ← Φ 是标准正态累积分布函数
        ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
```

GELU 在 x<0 时仍有小梯度（不完全截断），训练效果通常优于 ReLU。

Swish（Google Brain 提出）：`Swish(x) = x · σ(βx)`，在 EfficientNet 等高效模型中常用。

## 前向传播

前向传播是数据从输入层流向输出层的过程：

```python
import torch

def forward_pass(x, weights, biases, activation_fn):
    """
    x: 输入张量 (batch_size, input_dim)
    """
    outputs = []
    h = x
    for W, b in zip(weights[:-1], biases[:-1]):
        z = h @ W + b          # 线性变换
        h = activation_fn(z)   # 非线性激活
        outputs.append(h)

    # 最后一层（通常不加激活，或用 softmax/sigmoid）
    logits = h @ weights[-1] + biases[-1]
    return logits, outputs
```

## 反向传播与梯度计算

反向传播（Backpropagation）是通过链式法则从输出层向输入层逐层计算梯度的算法：

```
给定损失 L，计算对所有参数 W⁽ˡ⁾, b⁽ˡ⁾ 的梯度：

1. 前向传播，记录每层的激活值
2. 计算输出层误差：δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾
3. 反向传播误差：δ⁽ˡ⁾ = ((W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾) ⊙ σ'(z⁽ˡ⁾)
4. 计算参数梯度：
   ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ (h⁽ˡ⁻¹⁾)ᵀ
   ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

PyTorch 的 `autograd` 自动完成这一过程，开发者只需：

```python
loss = criterion(outputs, labels)
loss.backward()          # 自动计算所有参数的梯度
optimizer.step()         # 用梯度更新参数
optimizer.zero_grad()    # 清空梯度
```

## 批归一化（Batch Normalization）

BatchNorm 在每个 mini-batch 上对每层的激活值进行归一化，解决"内部协变量偏移"：

```
BN(x) = γ · (x - μ_batch) / √(σ²_batch + ε) + β

μ_batch：batch 内均值
σ²_batch：batch 内方差
γ, β：可学习的缩放和偏移参数（恢复表达能力）
```

```python
layer = nn.Sequential(
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),  # 或 nn.LayerNorm(256) 用于 Transformer
    nn.ReLU(),
)
```

BatchNorm 的作用：
- 加速训练（允许更大学习率）
- 有一定正则化效果（batch 内的随机性）
- 减少对权重初始化的敏感性

注意：Transformer 通常使用 **LayerNorm**（对单个样本的特征维度归一化），因为其 batch size 可能很小或序列长度变化。

## Dropout——防过拟合

训练时随机将一定比例的神经元输出置零：

```python
class DropoutLayer(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)  # 保持期望值不变（inverted dropout）
        return x
```

Dropout 的正则化效果类比集成学习：每次训练步骤相当于训练不同子网络，最终预测相当于对指数级多个子网络的集成。

推理时（`model.eval()`），Dropout 自动关闭，所有神经元都激活。

## 权重初始化

合理的初始化对训练稳定性至关重要：

### Xavier/Glorot 初始化（适用于 Sigmoid/Tanh）

```python
# 方差 = 2 / (fan_in + fan_out)
nn.init.xavier_uniform_(layer.weight)   # 均匀分布
nn.init.xavier_normal_(layer.weight)    # 正态分布
```

目标：使每层的激活值和梯度在前向/反向传播过程中保持大致相同的方差。

### He/Kaiming 初始化（适用于 ReLU）

ReLU 会将负数置零，相当于有效减半方差，因此需要更大的初始方差：

```python
# 方差 = 2 / fan_in
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

**不良初始化的后果**：
- 初始值太小：激活值逐层缩小，最终消失（梯度消失）
- 初始值太大：激活值逐层爆炸，训练发散（梯度爆炸）
- 全零初始化：所有神经元对称，永远输出相同的激活值（对称性破坏失败）

## 训练实践流程

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = MLP(input_dim=784, hidden_dim=512, output_dim=10)
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        # 前向传播
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

    # 验证
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)

    print(f"Epoch {epoch}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")
```
