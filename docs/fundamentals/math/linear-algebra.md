# 线性代数基础

线性代数是人工智能和机器学习的数学语言。神经网络的前向传播是一系列矩阵乘法，注意力机制是向量点积的集合，Embedding 空间是高维向量空间中的几何结构。没有线性代数的直觉，就无法真正理解深度学习模型的工作原理。

## 向量与矩阵

### 向量（Vector）

向量是有序的数字列表，可以理解为 n 维空间中的一个点或方向：

```
列向量（Column Vector）：
x = [x₁]
    [x₂]
    [x₃]

行向量（Row Vector）：
xᵀ = [x₁, x₂, x₃]
```

在 AI 中，向量无处不在：

- **词向量（Word Embedding）**：每个词被表示为 768 维或 1536 维的实值向量，捕捉语义关系
- **图像特征向量**：CNN 最后一层的特征图展平后得到的向量，表示图像的高层语义
- **用户/物品向量**：推荐系统中，用户偏好和物品特征各用一个向量表示

### 矩阵（Matrix）

矩阵是二维的数字数组，形状记为 m×n（m 行 n 列）：

```python
# NumPy 示例
import numpy as np
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 形状 2×3
```

矩阵的核心运算：

- **加法**：同形状矩阵对应元素相加，`(A+B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ`
- **标量乘法**：每个元素乘以标量 c，`(cA)ᵢⱼ = c·Aᵢⱼ`
- **转置**：行列互换，`(Aᵀ)ᵢⱼ = Aⱼᵢ`，形状从 m×n 变为 n×m

## 矩阵乘法——深度学习的核心操作

### 维度规则

矩阵乘法 `C = A × B` 要求 A 的列数等于 B 的行数：

```
A: (m×k)  ×  B: (k×n)  →  C: (m×n)

C[i,j] = Σₖ A[i,k] · B[k,j]  （第 i 行与第 j 列的点积）
```

**维度检查是调试神经网络的关键技能**：

```python
# 神经网络线性层示例
X = np.random.randn(32, 784)   # 批次 32，输入维度 784（MNIST 展平）
W = np.random.randn(784, 256)  # 权重矩阵
b = np.zeros(256)              # 偏置向量

# 前向传播：(32, 784) × (784, 256) → (32, 256)
output = X @ W + b
```

### 批量矩阵乘法（Batch Matrix Multiply）

深度学习框架中常见的批量操作：

```python
# PyTorch 中的 bmm
# 输入形状：(batch, seq_len, d_model)
Q = torch.randn(8, 512, 64)   # 8个样本，512个token，64维
K = torch.randn(8, 512, 64)
# 注意力分数：(8, 512, 512)
scores = torch.bmm(Q, K.transpose(-2, -1))
```

## 点积与相似度

向量点积（Dot Product）衡量两个向量的相似程度：

```
a · b = Σᵢ aᵢbᵢ = |a| · |b| · cos(θ)
```

- 当 θ=0（方向相同）时，cos(θ)=1，点积最大（最相似）
- 当 θ=90°（正交）时，cos(θ)=0，点积为零（无关）
- 当 θ=180°（方向相反）时，cos(θ)=-1（最不相似）

**余弦相似度**在 AI 中的应用：

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 词向量相似度
king = word_vectors["king"]    # 形状 (300,)
queen = word_vectors["queen"]  # 形状 (300,)
sim = cosine_similarity(king, queen)  # 接近 1.0
```

Word2Vec 的著名类比：`king - man + woman ≈ queen` 正是在向量空间中做算术。

## 特征值与特征向量

### 直觉理解

对于方阵 A，如果存在非零向量 v 使得 `Av = λv`，则 λ 为特征值，v 为对应的特征向量。

直觉：矩阵 A 作用于特征向量时，只改变其长度（缩放因子 λ），不改变方向。这揭示了矩阵变换的"固有方向"。

```python
import numpy as np
A = np.array([[4, 1],
              [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print("特征值:", eigenvalues)  # [5., 2.]
```

### 在 AI 中的应用

- **PCA 降维**：协方差矩阵的特征向量指向数据方差最大的方向（主成分）
- **图神经网络**：图拉普拉斯矩阵的特征分解用于谱图卷积
- **稳定性分析**：Hessian 矩阵的特征值用于分析优化景观（鞍点检测）

## 奇异值分解（SVD）

SVD 是线性代数中最强大的分解之一，将任意矩阵分解为三个特殊矩阵的乘积：

```
A = U Σ Vᵀ

其中：
- U: m×m 正交矩阵（左奇异向量）
- Σ: m×n 对角矩阵（奇异值，按降序排列）
- V: n×n 正交矩阵（右奇异向量）
```

### 应用一：图像压缩

```python
from PIL import Image
import numpy as np

img = np.array(Image.open("photo.jpg").convert("L"))  # 灰度图，(M, N)
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# 只保留前 k 个奇异值实现压缩
k = 50
img_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
# k=50 时，通常保留 90%+ 的视觉质量，但存储量大幅压缩
```

### 应用二：推荐系统（矩阵分解）

用户-物品评分矩阵 R（用户×电影）通常极度稀疏，SVD（或其变体 ALS、FunkSVD）将其分解为：

```
R ≈ U × Σ × Vᵀ

- U 的每行：用户的"隐因子"表示（对不同类型电影的偏好）
- V 的每行：电影的"隐因子"表示（属于哪些类型）
- 点积 U[user] · V[item] 预测评分
```

### 应用三：Transformer 中的低秩分解

LoRA（Low-Rank Adaptation）技术将大型权重矩阵的更新分解为两个低秩矩阵的乘积：

```
ΔW = B × A，其中 rank(ΔW) = r << min(m, n)
```

这使得微调大模型所需的可训练参数量从数十亿降至数百万。

## 在 AI 中的具体应用总结

### 权重矩阵（Neural Network Weights）

神经网络的每一层本质上是一个矩阵变换：

```python
# 全连接层
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        return x @ self.W + self.b  # 矩阵乘法 + 广播加法
```

### Embedding 空间的几何结构

Transformer 中的词嵌入矩阵是一个 `(vocab_size, d_model)` 的矩阵。每个词对应一行向量，语义相似的词在高维空间中距离更近。注意力机制中的 `Q, K, V` 投影矩阵进一步将词向量映射到不同的子空间。

### 多头注意力计算

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V

维度分析（以 GPT-2 为例）：
- Q, K, V 形状：(batch, heads, seq_len, head_dim)
- QKᵀ 形状：(batch, heads, seq_len, seq_len)  ← 注意力分数矩阵
- 输出形状：(batch, heads, seq_len, head_dim)
```

线性代数的直觉理解：注意力机制是在用 Q（查询）与所有 K（键）做内积，找到最相关的位置，然后按相关性加权求和 V（值）——本质是加权矩阵乘法。
