# CLIP 与对比学习

CLIP（Contrastive Language-Image Pre-training）是 OpenAI 于 2021 年发布的里程碑性工作，通过对比学习将图像和文本映射到同一语义空间，赋予了模型强大的零样本图像分类和跨模态检索能力，并成为后续几乎所有多模态模型的基础组件。

## CLIP 原理：对比学习

CLIP 的核心思想是**对比学习（Contrastive Learning）**：利用互联网上自然配对的图文数据（如带有 alt-text 的图片）训练一个共享的语义空间。

### 训练目标

给定一批 N 个图文对 {(图像1, 文本1), (图像2, 文本2), ..., (图像N, 文本N)}：

- **正样本**：第 i 张图像与第 i 条文本（真实配对）
- **负样本**：第 i 张图像与其他 N-1 条文本（随机配对）

训练目标是最大化正样本对的余弦相似度，同时最小化负样本对的相似度：

```
对比损失 = CrossEntropy(图文相似度矩阵, 单位矩阵)
```

这形成一个 N×N 的相似度矩阵，对角线元素应该最高（正对），非对角线元素应该最低（负对）。

### 双塔架构

CLIP 采用**双塔（Dual Encoder / Two-Tower）**架构：

```
图像 → 图像编码器（ViT 或 ResNet）→ 图像 Embedding（512/768/1024 维）
文本 → 文本编码器（Transformer）  → 文本 Embedding（同维度）

两者投影到同一超球面，使用 L2 归一化
相似度 = 余弦相似度（内积）
```

两个编码器独立处理各自模态，最终输出的 embedding 在同一向量空间中进行比较，这种设计使得图像和文本的检索非常高效（可以预计算任意一侧的 embedding）。

## 零样本图像分类

CLIP 最令人惊叹的能力是**无需任何训练数据**即可对新类别进行分类：

```python
# 零样本分类示例
import clip, torch
from PIL import Image

model, preprocess = clip.load("ViT-B/32")

image = preprocess(Image.open("cat.jpg")).unsqueeze(0)
text = clip.tokenize(["a photo of a cat", "a photo of a dog", "a photo of a bird"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    similarity = (image_features @ text_features.T).softmax(dim=-1)
    # 输出各类别的概率
```

工作原理：将每个类别名称转为文本描述（如"a photo of a {class_name}"），CLIP 计算图像与所有类别文本的相似度，相似度最高的即为预测类别。

在 ImageNet 上，CLIP ViT-L/14 的零样本 Top-1 精度约为 76%，与 ResNet-50 的监督训练精度相当。

## 图像搜索与以文搜图

CLIP 是**跨模态检索**的理想工具：

```python
# 以文搜图：给定文本查询，找到最相关的图片
query_text = "sunset over mountains"
text_emb = model.encode_text(clip.tokenize([query_text]))

# 预计算图库中所有图片的 embedding
image_embs = [model.encode_image(img) for img in image_library]

# 计算相似度，返回 Top-K 图片
similarities = [cos_sim(text_emb, img_emb) for img_emb in image_embs]
top_k = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:5]
```

- **以文搜图**：用文字描述查找图片库中的匹配图片，无需标签
- **以图搜图**：用图片 embedding 检索视觉相似的图片
- **以图搜文**：找到与图片内容最相关的文字描述

## CLIP 在生成模型中的角色

CLIP 的文本编码器在文本生成图像系统中扮演关键角色：

### 在 Stable Diffusion 中

Stable Diffusion 使用 **CLIP 文本编码器**（ViT-L 版本）将用户的文本 Prompt 转化为条件 Embedding，引导扩散模型的去噪方向：

```
Prompt: "a cat sitting on a red sofa, studio photography"
  → CLIP 文本编码器
  → 77×768 的 token embedding 矩阵
  → 通过 Cross-Attention 注入 UNet
  → 引导生成符合描述的图像
```

CLIP 的文本理解能力直接决定了文生图系统对 Prompt 的理解能力。SDXL 使用了两个 CLIP 模型（OpenCLIP ViT-bigG + CLIP ViT-L）来增强文本理解。

### 在 DALL-E 2 中

DALL-E 2 使用 CLIP 图像 Embedding 作为扩散模型的条件，先由文本生成 CLIP 图像 Embedding（Prior），再由图像 Embedding 生成实际图像（Decoder），实现了图文的语义映射。

## OpenCLIP：开源实现

**OpenCLIP（LAION-AI）** 是 CLIP 的开源复现，提供了更多规模的预训练模型：

- 在 LAION-2B（20 亿图文对）上训练，规模远超原版 CLIP 的 4 亿对
- 提供 ViT-B/32、ViT-L/14、ViT-H/14、ViT-G/14 等多个规模
- 在部分任务上精度超过 OpenAI 的原版 CLIP
- 完全开源，商业可用

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='laion2b_s32b_b82k'
)
```

## SigLIP：Google 的改进版

**SigLIP（Sigmoid Loss for Language-Image Pre-Training，Google，2023）** 提出了改进的训练目标：

- 用 **Sigmoid 损失**代替 Softmax 的对比损失，消除了 CLIP 训练时必须在大批量（Batch Size）内做全局归一化的问题
- 每个图文对独立计算 Sigmoid 二分类损失，训练更稳定
- 在小 Batch Size 下也能有效训练
- 在 VLM 中作为图像编码器时效果更好

SigLIP 是 InternVL、PaliGemma 等最新开源 VLM 的首选图像编码器。

## 在下游任务中的迁移

除了直接的零样本分类和检索，CLIP 特征在众多下游任务中被广泛使用：

- **VLM 的图像编码器**：LLaVA、InternVL 等 VLM 直接复用 CLIP/SigLIP 编码器作为视觉特征提取器
- **图像-文本对齐评分**：评估生成图像与文本描述的匹配程度（如 CLIPScore 指标）
- **少样本图像分类**：用 CLIP 提取特征，再训练简单分类头，用极少标注数据微调
- **图像质量评估**：CLIP 学到的语义空间可用于衡量图像语义质量

## 局限性

尽管 CLIP 功能强大，仍有以下局限性：

- **细粒度理解不足**：难以区分"一只白猫和一只黑猫"与"一只黑猫和一只白猫"（位置关系、数量关系弱）
- **复杂场景关系**：对多目标的空间关系（"左边的苹果在右边的香蕉后面"）理解较差
- **文字识别（OCR）**：对图像中的文字内容理解有限
- **长文本对齐**：文本端输入限制为 77 个 token，对长描述理解能力受限
- **领域泛化**：在特定专业领域（医学、卫星图像等）表现较弱，需要领域微调（如 BiomedCLIP、RemoteCLIP）
