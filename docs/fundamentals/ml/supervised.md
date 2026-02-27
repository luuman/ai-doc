# 监督学习

监督学习是机器学习中最成熟、应用最广泛的范式。其核心思想是：给定一批带有标签的训练样本，学习从输入到输出的映射函数，使得对新输入能够准确预测对应的输出。工业界绝大多数 AI 应用——图像分类、垃圾邮件过滤、信用评分、语音识别——都建立在监督学习之上。

## 监督学习定义

监督学习的基本框架：

- **训练集**：`{(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}`，每个样本包含特征向量 xᵢ 和对应标签 yᵢ
- **目标**：学习函数 `f: X → Y` 使预测 `ŷ = f(x)` 尽可能接近真实标签 y
- **泛化**：模型应对训练集之外的新样本也有良好预测能力

根据标签类型，分为两大任务：

- **回归（Regression）**：标签为连续值，如预测房价、气温、股价
- **分类（Classification）**：标签为离散类别，如垃圾邮件判断、图像分类、情感分析

## 数据集划分

合理的数据划分是防止评估偏差的关键：

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 最终比例：训练集 70%，验证集 15%，测试集 15%
```

- **训练集**：用于学习模型参数（反向传播更新权重）
- **验证集**（开发集）：用于调整超参数（学习率、层数、正则化系数）和早停（Early Stopping）
- **测试集**：仅用于最终评估，**绝不能用于任何训练决策**（否则测试集数据"泄露"导致过于乐观的评估）

**交叉验证（K-Fold CV）**：数据量不足时，将数据分为 K 折轮流作为验证集，结果更稳健：

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"准确率: {scores.mean():.3f} ± {scores.std():.3f}")
```

## 回归算法

### 线性回归（Linear Regression）

假设目标变量与特征之间存在线性关系：

```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b
```

损失函数为均方误差（MSE）：

```
L(w) = (1/n) Σᵢ (yᵢ - ŷᵢ)² = (1/n) ||y - Xw||²
```

**解析解**（小数据集）：`w* = (XᵀX)⁻¹Xᵀy`

**梯度下降解**（大数据集）：`w = w - η · ∂L/∂w`

### 多项式回归（Polynomial Regression）

通过特征工程将非线性关系线性化：

```
ŷ = w₀ + w₁x + w₂x² + w₃x³ + ...
```

本质上仍是线性回归，只是将 `x, x², x³` 等视为新特征。多项式阶数是重要超参数，阶数过高导致过拟合。

## 分类算法

### 逻辑回归（Logistic Regression）

虽名为"回归"，实为二分类算法。将线性输出通过 Sigmoid 函数压缩到 (0,1) 区间作为概率：

```
P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^(-wᵀx-b))
```

损失函数为二元交叉熵：`L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`

- 简单、可解释、训练快，适合线性可分问题
- 可通过 softmax 推广到多分类（Multinomial Logistic Regression）

### 决策树（Decision Tree）

通过递归分裂特征空间构建树形决策规则：

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
tree.fit(X_train, y_train)
```

分裂准则（选择哪个特征分裂）：
- **信息增益**（ID3）：按熵减少量选择，`IG = H(parent) - Σ (|child|/|parent|)·H(child)`
- **基尼不纯度**（CART）：`Gini = 1 - Σ pₖ²`，计算更高效，sklearn 默认

- 优点：可解释性极强（可视化树形结构），无需特征归一化
- 缺点：单棵树容易过拟合，对训练数据中的噪声敏感

### 支持向量机（SVM）

寻找最大化类别间隔的超平面：

```
最大化间隔 = 2/||w||
约束：yᵢ(wᵀxᵢ + b) ≥ 1
```

- **核技巧（Kernel Trick）**：通过 `K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)` 隐式映射到高维空间，处理非线性分类
- 常用核：线性核、多项式核、RBF 核（径向基函数）
- 高维小数据集（文本、基因组）上表现优异，计算复杂度随数据量增加

### 随机森林（Random Forest）

集成多棵决策树，通过投票（分类）或平均（回归）得到最终预测：

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
rf.fit(X_train, y_train)
print(rf.feature_importances_)  # 特征重要性
```

两个随机化来源：
- **Bagging**：每棵树用不同的有放回随机子样本训练
- **特征随机**：每次分裂随机选取特征子集，降低树之间的相关性

随机森林对过拟合有天然抵抗力，且不需要特征归一化。

### XGBoost（梯度提升树）

以梯度提升（Gradient Boosting）方式串行训练树，每棵新树拟合前一轮的残差：

```python
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)
```

XGBoost 在结构化/表格数据竞赛（Kaggle）中长期霸榜，其关键改进包括：
- 二阶泰勒展开优化目标
- 正则化项防止过拟合（L1/L2 对叶节点权重）
- 列抽样、行抽样减少过拟合
- 支持缺失值处理和并行计算

## 损失函数

| 任务 | 损失函数 | 公式 |
|------|---------|------|
| 回归 | MSE（均方误差） | `(1/n) Σ(y-ŷ)²` |
| 回归 | MAE（平均绝对误差） | `(1/n) Σ|y-ŷ|` |
| 回归 | Huber Loss | 结合 MSE 和 MAE，对异常值鲁棒 |
| 二分类 | Binary Cross-Entropy | `-[y·log(ŷ)+(1-y)·log(1-ŷ)]` |
| 多分类 | Cross-Entropy | `-Σₖ yₖ·log(ŷₖ)` |

## 过拟合与欠拟合

### 偏差-方差权衡（Bias-Variance Tradeoff）

泛化误差 = 偏差² + 方差 + 不可避免噪声

- **欠拟合（High Bias）**：模型过于简单，无法捕捉数据规律；训练误差和测试误差都高
  - 解决：增加模型复杂度、添加更多特征
- **过拟合（High Variance）**：模型记住了训练数据的噪声；训练误差低但测试误差高
  - 解决：正则化、数据增强、减少特征

### 正则化（Regularization）

在损失函数中添加惩罚项限制模型复杂度：

```
L2 正则（Ridge）：   L_total = L + λ·||w||²     ← 权重趋向均匀小值
L1 正则（Lasso）：   L_total = L + λ·||w||₁    ← 权重趋向稀疏（部分为 0）
弹性网络（ElasticNet）：L_total = L + α·λ·||w||₁ + (1-α)·λ/2·||w||²
```

**Dropout**（神经网络专用）：训练时随机将一定比例神经元的输出置零，强迫网络学习冗余表示：

```python
import torch.nn as nn
layer = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # 50% 的神经元随机失活
)
```

## 超参调优方法

找到最优超参数是 ML 工程的核心任务之一：

- **网格搜索（Grid Search）**：穷举所有超参数组合，计算代价高，适合参数少时
- **随机搜索（Random Search）**：随机采样超参数组合，通常比网格搜索更高效（Bergstra & Bengio, 2012）
- **贝叶斯优化（Bayesian Optimization）**：用代理模型（高斯过程）建模超参数与性能的关系，智能选择下一个尝试点，效率最高
- **AutoML 工具**：Optuna、Ray Tune、HyperOpt 等框架自动化超参搜索过程

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int('n_layers', 1, 5)
    model = build_model(n_layers)
    return train_and_evaluate(model, lr)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"最优参数: {study.best_params}")
```
