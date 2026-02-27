# 无监督学习

无监督学习面向一个现实挑战：现实世界中大量数据缺乏标注，而人工标注成本极高。无监督学习算法无需标签，直接从数据的内在结构中发现规律——寻找自然聚类、发现低维表示、建模数据分布。它是现代大模型预训练的核心方法论基础。

## 无监督学习定义

无监督学习的输入：
- 只有特征矩阵 X，没有标签向量 y
- 目标是发现数据的内在结构：聚类（相似样本分组）、降维（压缩表示）、密度估计（数据分布建模）

与监督学习的本质区别：
- 没有"正确答案"，评估标准更复杂（内部指标 vs 外部指标）
- 更依赖人类对结果的解读和验证
- 数据获取成本低（大量未标注数据），但结果解释困难

## 聚类算法

### K-means 聚类

K-means 是最经典的聚类算法，将 n 个样本划分为 K 个簇，使每个样本与其所属簇中心的距离之和最小：

**目标函数**：
```
min Σₖ Σ_{x∈Cₖ} ||x - μₖ||²

μₖ = 簇 k 的质心 = (1/|Cₖ|) Σ_{x∈Cₖ} x
```

**EM 迭代过程**：

```python
from sklearn.cluster import KMeans
import numpy as np

X = np.random.randn(1000, 2)  # 1000 个二维样本
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_            # 每个样本的簇标签
centers = kmeans.cluster_centers_  # 5 个簇心坐标
inertia = kmeans.inertia_          # 总误差平方和（越小越好）
```

**E步（分配）**：将每个样本分配给最近的质心
**M步（更新）**：重新计算每个簇的质心

**K-means 的局限**：
- 需要预先指定 K（使用肘部法则或轮廓系数帮助选择）
- 对初始质心敏感（K-means++ 改进初始化）
- 假设簇为球形、各向同性，无法处理任意形状的簇
- 对异常值敏感

### DBSCAN（基于密度的聚类）

DBSCAN 不需要预先指定簇数，通过核心点、边界点和噪声点的概念发现任意形状的簇：

**关键参数**：
- `ε`（eps）：邻域半径
- `MinPts`：核心点的最少邻居数

**点的分类**：
- **核心点**：eps 邻域内有至少 MinPts 个点
- **边界点**：在核心点的 eps 邻域内，但自身邻域内点数不足
- **噪声点**：既非核心点也非边界点，标记为异常值（标签 = -1）

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X)
# labels=-1 的点是噪声/异常值
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
```

DBSCAN 的优势：自动检测异常值，发现非球形簇，不受初始化影响。

### 层次聚类（Hierarchical Clustering）

自底向上（凝聚式）或自顶向下（分裂式）构建簇的树状结构（Dendrogram）：

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

Z = linkage(X, method='ward')  # Ward 最小化簇内方差增量
dendrogram(Z)
plt.show()
```

优点：不需要预先指定 K，可视化 Dendrogram 帮助确定最优层次；缺点：计算复杂度 O(n²log n)，不适合大规模数据。

## 降维

### PCA（主成分分析）

PCA 找到数据方差最大的方向（主成分），将高维数据投影到低维空间，同时最大化保留信息量：

**数学原理**：
1. 中心化数据：`X̃ = X - μ`
2. 计算协方差矩阵：`C = X̃ᵀX̃ / (n-1)`
3. 对 C 进行特征分解：`C = VΛVᵀ`
4. 选取最大的 k 个特征向量作为主成分
5. 投影：`Z = X̃V_k`

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=50)              # 降到 50 维
X_reduced = pca.fit_transform(X)        # (n_samples, 50)
explained_var = pca.explained_variance_ratio_  # 每个主成分解释的方差比例
print(f"前50个主成分解释了 {sum(explained_var)*100:.1f}% 的方差")
```

**在 AI 中的应用**：
- 图像压缩（特征脸 Eigenfaces）
- 去噪：丢弃低方差成分等效于去除噪声
- 预处理：降维后训练的 ML 模型更快、过拟合风险更低
- 可视化：降到 2-3 维后可直接绘制散点图

### t-SNE（t 分布随机邻居嵌入）

t-SNE 专为高维数据的二维/三维可视化设计，能保留局部邻居关系：

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_2d = tsne.fit_transform(X_high_dim)  # 通常先用 PCA 降到 50 维再 t-SNE

import matplotlib.pyplot as plt
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10')
plt.colorbar()
plt.show()
```

**t-SNE 的特点**：
- 同类样本在 2D 空间聚集成团，可直觉验证模型的表示质量
- 计算代价高（O(n²) 或 O(n log n) 的 Barnes-Hut 近似）
- 超参 `perplexity`（通常 5-50）控制"有效邻居数"，对结果影响显著
- **不能**用于下游任务的降维（不保留全局结构），仅供可视化

### UMAP（统一流形近似与投影）

UMAP 是 t-SNE 的现代替代，速度更快，同时保留更好的全局结构：

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
X_2d = reducer.fit_transform(X)
```

UMAP 在大规模数据集（百万级）上更实用，且可用于真实降维（不只是可视化），输出可作为 ML 的输入特征。

## 自编码器（Autoencoder）

自编码器用神经网络学习数据的压缩表示：

```
输入 x → 编码器 Encoder → 潜变量 z → 解码器 Decoder → 重建输出 x̂

目标：最小化重建误差 ||x - x̂||²
```

```python
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
```

**变体**：
- **稀疏自编码器**：加 L1 正则约束潜变量稀疏，迫使编码更具代表性
- **去噪自编码器**：输入加噪声，训练重建干净版本，学习更鲁棒的表示
- **变分自编码器（VAE）**：潜空间服从正态分布，支持生成新样本
- **对比自编码器**：用对比学习损失替代重建损失，学习更具判别性的表示

## 异常检测

无监督异常检测假设正常数据符合某种分布，偏离该分布的样本视为异常：

- **孤立森林（Isolation Forest）**：随机分裂特征空间，异常点更容易被早期隔离（路径短），计算高效
- **OneClass-SVM**：在特征空间中找到包含大多数正常样本的最小超球面
- **自编码器重建误差**：训练仅在正常数据上，异常样本重建误差远高于正常样本阈值

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
labels = iso.fit_predict(X)  # -1 为异常点，1 为正常点
```

## 在 AI 预训练中的核心角色

无监督学习是现代大模型的基石：

### 自监督学习（Self-Supervised Learning）

自监督学习是一种特殊的无监督学习，通过从数据本身构造伪标签来训练：

- **掩码语言模型（MLM）**（BERT）：随机遮盖 15% 的 token，训练模型预测被遮盖的词——不需要任何人工标注
- **自回归语言模型**（GPT）：预测序列的下一个 token，整个互联网文本都是"标签"
- **对比学习**（CLIP、SimCLR）：对同一样本的不同增强视角拉近表示，对不同样本推远，在无监督条件下学习强大的视觉/多模态表示

大型语言模型预训练的数据效率惊人：数千亿 token 的文本，无需一条人工标注，训练出的模型已具备惊人的语言理解和生成能力，然后通过少量有监督微调（SFT）和 RLHF 进一步对齐人类偏好。

### 聚类在半监督学习中的作用

在标注数据稀缺时，聚类可帮助选择最有代表性的样本进行标注（主动学习），或用伪标签扩展训练集，降低人工标注成本。
