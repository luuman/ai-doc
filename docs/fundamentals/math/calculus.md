# 微积分与优化

微积分是深度学习训练的引擎。反向传播算法本质上是链式法则的应用，梯度下降是寻找函数最小值的核心方法，而现代优化器（Adam、AdaGrad）则是在梯度下降基础上的系统性改进。理解这些概念，是从"使用深度学习"到"驾驭深度学习"的关键跨越。

## 偏导数与梯度

### 偏导数（Partial Derivative）

对于多变量函数 `f(x₁, x₂, ..., xₙ)`，偏导数衡量函数值随某一变量变化的速率，同时保持其他变量不变：

```
∂f/∂xᵢ = lim_{h→0} [f(..., xᵢ+h, ...) - f(..., xᵢ, ...)] / h
```

**直觉**：偏导数回答"当我稍微增大 xᵢ 时，函数值会增大还是减小，变化多快？"

### 梯度（Gradient）

梯度是所有偏导数组成的向量，指向函数值增长最快的方向：

```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**关键性质**：
- 梯度方向是函数值上升最快的方向
- 负梯度方向是函数值下降最快的方向（梯度下降的基础）
- 在极值点处，梯度为零向量（∇f = 0）

```python
# PyTorch 自动微分示例
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
f = x[0]**2 + 2*x[1]**2 + x[0]*x[1]  # f = x₁² + 2x₂² + x₁x₂

f.backward()  # 计算梯度
print(x.grad)  # tensor([2*x₁ + x₂, 4*x₂ + x₁]) = tensor([7., 14.])
```

## 链式法则——反向传播的数学基础

### 单变量链式法则

若 `z = f(y), y = g(x)`，则：

```
dz/dx = (dz/dy) · (dy/dx)
```

### 多变量链式法则

若 z 依赖于多个中间变量：

```
∂z/∂x = Σᵢ (∂z/∂yᵢ) · (∂yᵢ/∂x)
```

### 神经网络中的链式法则应用

考虑一个简单的两层网络：

```
输入 x → 线性层 z₁=Wx+b → 激活 a₁=ReLU(z₁) → 线性层 z₂=W₂a₁ → 损失 L=MSE(z₂, y)
```

反向传播通过链式法则逐层计算梯度：

```
∂L/∂W = (∂L/∂z₂) · (∂z₂/∂a₁) · (∂a₁/∂z₁) · (∂z₁/∂W)

分别是：
- ∂L/∂z₂ = 2(z₂ - y)          （MSE 损失的梯度）
- ∂z₂/∂a₁ = W₂ᵀ                （线性层的梯度）
- ∂a₁/∂z₁ = 1 if z₁>0 else 0  （ReLU 的梯度）
- ∂z₁/∂W = xᵀ                  （线性层对权重的梯度）
```

PyTorch 的 `autograd` 引擎自动构建计算图并执行这一链式求导过程，开发者无需手动推导。

## 梯度下降——下山比喻

### 核心思想

想象你站在一座有雾的山上，无法看到山的全貌，只能感知脚下的坡度。梯度下降的策略是：**始终朝坡度最陡的下山方向走一小步**，如此迭代直到到达谷底（函数最小值）。

```
θ_new = θ_old - η · ∇L(θ_old)

其中：
- θ：模型参数（神经网络的所有权重）
- η（eta）：学习率（步长大小）
- ∇L(θ)：损失函数关于参数的梯度
```

### 批量梯度下降（Batch Gradient Descent）

每次使用**全部训练数据**计算梯度：

```python
for epoch in range(num_epochs):
    grad = compute_gradient(all_training_data, theta)
    theta = theta - learning_rate * grad
```

- 优点：梯度估计精确，收敛方向稳定
- 缺点：计算代价极高（百万数据集时每步都遍历全集），无法在线学习

### 随机梯度下降（SGD，Stochastic Gradient Descent）

每次只使用**单个样本**计算梯度：

```python
for epoch in range(num_epochs):
    for x, y in training_data:  # 每次只用一个样本
        grad = compute_gradient(x, y, theta)
        theta = theta - learning_rate * grad
```

- 优点：更新频繁，能逃离局部最优（梯度噪声有正则化效果）
- 缺点：梯度估计有噪声，收敛不稳定，难以利用 GPU 并行

### 小批量梯度下降（Mini-batch SGD）——实践中的标准

折中方案，每次使用**小批量**（通常 32-512 个样本）计算梯度：

```python
for epoch in range(num_epochs):
    for batch in dataloader:  # batch_size 通常 32-256
        grad = compute_gradient(batch, theta)
        theta = theta - learning_rate * grad
```

- 梯度估计比单样本更准确
- 可充分利用 GPU 并行矩阵运算
- 训练效率和稳定性兼顾，是现代深度学习的标准做法

## 现代优化器

### SGD + Momentum（动量）

普通 SGD 在峡谷形损失曲面上会产生剧烈震荡。Momentum 引入历史梯度的指数移动平均，平滑更新方向：

```
v_t = β · v_{t-1} + (1-β) · ∇L(θ)   ← 动量项，β 通常 0.9
θ = θ - η · v_t
```

类比：小球在山坡上滚动时，由于惯性不会骤然改变方向，而是逐渐转向——这使其能更快地通过平坦区域，减少在陡峭区域的震荡。

### AdaGrad（自适应学习率）

不同参数使用不同的学习率，历史梯度大的参数获得更小的有效学习率：

```
G_t = G_{t-1} + (∇L(θ))²    ← 累积梯度平方
θ = θ - (η / √(G_t + ε)) · ∇L(θ)
```

- 优点：稀疏特征（如词向量）更新时效果好（频繁出现的词梯度大，学习率小；罕见词反之）
- 缺点：G_t 单调递增，学习率会不断缩小直至停止学习

### Adam（Adaptive Moment Estimation）——现代深度学习的首选

Adam 结合了 Momentum 和 RMSProp 的优势：

```
m_t = β₁ · m_{t-1} + (1-β₁) · ∇L(θ)         ← 一阶矩（均值）
v_t = β₂ · v_{t-1} + (1-β₂) · (∇L(θ))²      ← 二阶矩（方差）

偏差修正：
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)

参数更新：
θ = θ - η · m̂_t / (√v̂_t + ε)

超参数默认值：β₁=0.9, β₂=0.999, ε=1e-8, η=1e-3（常用3e-4）
```

Adam 对学习率不敏感（适应性调整），在大多数任务上开箱即用，是最广泛使用的优化器。

```python
import torch.optim as optim

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

for batch in dataloader:
    optimizer.zero_grad()      # 清空上一步的梯度
    loss = model(batch)        # 前向传播计算损失
    loss.backward()            # 反向传播计算梯度
    optimizer.step()           # Adam 更新参数
```

### AdamW（Adam + Weight Decay）

Transformer 训练中最常用的优化器，修正了 Adam 中权重衰减的实现方式：

```
θ = θ - η · (m̂_t / (√v̂_t + ε) + λ · θ)
```

AdamW 将 L2 正则化直接作用于参数（而非梯度），在语言模型训练中优于 Adam。

## 学习率与收敛

学习率 η 是训练中最重要的超参数：

| 学习率 | 效果 |
|--------|------|
| 过大（如 0.1） | 损失震荡不收敛，可能发散 |
| 过小（如 1e-7） | 收敛极慢，容易陷入局部最优 |
| 适中（如 1e-3~3e-4） | 平稳收敛，效果好 |

### 学习率调度策略

- **学习率预热（Warmup）**：从小学习率线性增大到目标值，防止早期训练不稳定
- **余弦退火（Cosine Annealing）**：按余弦函数周期性衰减，BERT/GPT 训练标准做法
- **学习率查找（LR Finder）**：Fast.ai 提出，通过扫描找到最优学习率范围

```python
# Transformers 库中的 cosine schedule with warmup
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=total_steps
)
```

## 鞍点与局部最优问题

### 鞍点（Saddle Points）

在高维空间中，"局部最优陷阱"的威胁远小于人们直觉上的担忧。更常见的问题是**鞍点**：在某些方向曲率为正（最小值方向），在其他方向曲率为负（最大值方向）。

鞍点处梯度为零，但不是极值，普通 SGD 可能在鞍点附近停滞。Momentum 和 Adam 的噪声有助于逃离鞍点。

### 损失曲面的高维特性

研究表明，在充分参数化的深度网络中：
- 大多数局部最优点的损失值与全局最优相差无几
- 真正危险的不是局部最优，而是平坦区域（梯度接近零但损失不低）
- 批归一化（BatchNorm）和残差连接有助于平滑损失曲面，使训练更稳定

### 梯度裁剪（Gradient Clipping）

Transformer 训练中常用的技巧，防止梯度爆炸：

```python
# 将梯度的 L2 范数限制在 max_norm 以内
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
