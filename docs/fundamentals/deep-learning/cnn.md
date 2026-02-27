# 卷积神经网络（CNN）

卷积神经网络（Convolutional Neural Network，CNN）是计算机视觉领域的革命性架构，专为处理具有网格状拓扑结构的数据（图像、音频、视频）而设计。与全连接网络相比，CNN 通过参数共享和局部连接，在保持平移不变性的同时极大减少了参数量，是 AlexNet 掀起深度学习革命的核心技术。

## 卷积操作原理

### 为什么需要卷积？

对 224×224×3 的 RGB 图像使用全连接层，输入维度为 150,528。若第一个隐藏层有 4096 个神经元，仅这一层就需要 150,528 × 4096 ≈ 6 亿个参数，内存和计算完全不可行。

卷积通过三个关键思想解决这一问题：
- **局部连接**：每个神经元只与局部感受野（如 3×3）相连，而非整张图像
- **参数共享**：同一卷积核在整张图像上滑动，所有位置共用参数
- **平移等变性**：对图像进行平移，对应的特征图也相应平移

### 卷积操作详解

卷积核（Kernel/Filter）是一个小矩阵（通常 3×3 或 5×5），在输入上滑动，每次计算局部区域的加权和：

```
输出特征图的每个元素：
output[i,j] = Σ_m Σ_n input[i+m, j+n] × kernel[m,n] + bias
```

**关键参数**：

- **卷积核大小（Kernel Size）**：通常 3×3（足够捕捉局部特征，计算效率高）
- **步长（Stride）**：卷积核每次移动的步数，stride=2 时特征图大小减半
- **填充（Padding）**：在输入边缘补零，`same` 填充保持输入输出尺寸一致

**输出尺寸公式**：
```
H_out = (H_in - K + 2P) / S + 1

H_in：输入高度，K：核大小，P：padding，S：stride
```

**多通道卷积**：

```python
import torch.nn as nn

# 输入：(batch, 3, 224, 224)   ← 3 通道 RGB
# 输出：(batch, 64, 224, 224)  ← 64 个特征图（same padding）
conv = nn.Conv2d(in_channels=3, out_channels=64,
                 kernel_size=3, stride=1, padding=1)

# 参数量：3 × 64 × 3 × 3 + 64 = 1,792（极少！）
```

每个输出通道对应一个独立卷积核，学习不同的特征（边缘、纹理、颜色等）。

## 池化层

池化层下采样特征图，减少空间维度，提高平移不变性和计算效率：

### Max Pooling（最大池化）

```python
# 取每个 2×2 区域的最大值，特征图尺寸减半
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# 输入 (batch, 64, 224, 224) → 输出 (batch, 64, 112, 112)
```

Max Pooling 保留最显著的特征，对小位移具有鲁棒性。

### Average Pooling（平均池化）

取区域平均值，比 Max Pooling 更平滑，现代网络（ResNet、Transformer）倾向于用全局平均池化（GAP）代替最后的全连接层：

```python
gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，任意输入尺寸 → (batch, C, 1, 1)
```

## 感受野（Receptive Field）

感受野是输出特征图的某个位置对应于输入图像的区域大小。随着网络层数增加，感受野快速扩大：

```
1层 3×3 卷积：感受野 3×3
2层 3×3 卷积：感受野 5×5
3层 3×3 卷积：感受野 7×7
...

关键洞见：两层 3×3 卷积的感受野等同于一层 5×5 卷积，
但参数量 (2×3×3=18) 远少于 5×5 (25)！
```

这是 VGG 网络系统性使用小卷积核的理论依据，也是深层网络优于浅层网络的重要原因。

## 经典架构演进

### LeNet-5（1998，杨立昆）

最早成功的卷积网络，用于手写数字识别（MNIST），结构：

```
输入(32×32) → Conv(6) → AvgPool → Conv(16) → AvgPool → FC(120) → FC(84) → 输出(10)
```

参数量约 6 万，奠定了卷积-池化-全连接的基本模式。

### AlexNet（2012，Krizhevsky et al.）

深度学习革命的开山之作，在 ImageNet 上将 Top-5 错误率从 26% 降至 15%：

```
输入(224×224×3) → Conv(96, 11×11, stride=4) → MaxPool →
Conv(256, 5×5) → MaxPool →
Conv(384, 3×3) → Conv(384, 3×3) → Conv(256, 3×3) → MaxPool →
FC(4096) → Dropout → FC(4096) → Dropout → FC(1000)
```

关键创新：GPU 并行训练、ReLU 激活函数、Dropout、数据增强（翻转/裁剪/颜色抖动）。

### VGG（2014，牛津大学）

系统使用 3×3 小卷积核，通道数翻倍策略：

```
VGG-16：13层卷积 + 3层全连接
特征：简洁统一的架构，极深（16/19层），1.38亿参数
贡献：证明深度是关键，3×3卷积的有效性
```

### Inception（GoogLeNet，2014）

并行使用不同尺寸的卷积核（1×1、3×3、5×5），让网络自己选择最合适的感受野：

```python
# Inception 模块伪代码
class InceptionModule(nn.Module):
    def forward(self, x):
        b1 = self.conv1x1(x)          # 1×1 卷积
        b2 = self.conv3x3(self.red3x3(x))  # 1×1 降维 + 3×3
        b3 = self.conv5x5(self.red5x5(x))  # 1×1 降维 + 5×5
        b4 = self.proj(self.pool(x))   # 池化 + 1×1
        return torch.cat([b1, b2, b3, b4], dim=1)  # 通道维度拼接
```

引入 **1×1 卷积**用于降维（"瓶颈层"），在大幅减少参数量的同时增加非线性。

### ResNet（2015，何恺明等，微软研究院）

**残差连接（Residual Connection / Skip Connection）** 是 ResNet 的核心创新，解决了深层网络训练中的梯度消失问题：

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x                     # 保存输入（跳跃连接）
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual             # 残差相加
        return F.relu(out)
```

**残差连接的数学直觉**：

网络学习的是残差 `F(x) = H(x) - x`，而非直接学习 `H(x)`。如果恒等变换是最优解，则 `F(x)=0` 比学习 `H(x)=x` 更容易（权重趋向零比趋向某个特定函数容易得多）。

梯度流分析：残差连接创造了梯度的"高速公路"，梯度可以直接从深层反向流向浅层，不经过卷积层，彻底解决了梯度消失问题。

ResNet-152 达到 152 层，Top-5 错误率 3.57%，首次超越人类水平（约 5.1%）。

### EfficientNet（2019，谷歌）

通过神经架构搜索（NAS）找到最优的网络宽度（通道数）、深度（层数）、输入分辨率的组合，实现参数效率的最优化。在相同参数量下，EfficientNet 显著优于同期所有 CNN 架构。

## CNN 在计算机视觉中的应用

### 图像分类

输入图像 → CNN 特征提取 → 全局平均池化 → 全连接分类头 → 类别概率

预训练 + 微调（Transfer Learning）是标准做法：

```python
import torchvision.models as models

model = models.resnet50(pretrained=True)  # 加载 ImageNet 预训练权重

# 冻结特征提取层，只训练分类头
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, num_classes)  # 替换分类头
```

### 目标检测

- **Faster R-CNN**：区域建议网络（RPN）+ RoI 对齐 + 分类/回归头，经典两阶段检测器
- **YOLO**（You Only Look Once）：单次前向传播预测所有目标框，实时检测的标准选择
- **DETR**（Detection Transformer）：用 Transformer 替代 CNN，端到端目标检测无需手工设计锚框

### 语义分割

- **FCN**（全卷积网络）：将分类网络最后的全连接层替换为卷积，输出与输入同尺寸的密集预测
- **U-Net**：编码器-解码器结构配合跳跃连接，在医学图像分割中广泛应用
