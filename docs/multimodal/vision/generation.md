# AI 图像生成

AI 图像生成是多模态 AI 中最具视觉冲击力的方向，近年来以扩散模型为核心实现了跨越式发展。从 GAN 到扩散模型，再到 FLUX.1 的 Diffusion Transformer，图像生成质量已经达到让人难以分辨真伪的程度。

## 生成模型演进

### GAN（生成对抗网络，2014-2021）

GAN 由 Ian Goodfellow 于 2014 年提出，通过**生成器（Generator）**和**判别器（Discriminator）**的对抗博弈来学习生成真实图像：

- 生成器：将随机噪声映射为图像，目标是"骗过"判别器
- 判别器：区分真实图像和生成图像，目标是"识破"生成器

GAN 的问题：训练不稳定（模式崩溃、梯度消失），超参数敏感，生成结果多样性有限。代表作包括 StyleGAN2（高质量人脸生成）、BigGAN（大规模多类别生成）。

### VAE（变分自编码器，2013）

VAE 通过学习数据的隐空间分布（均值+方差）实现生成：将图像编码为高斯分布的参数，从分布中采样再解码生成图像。VAE 生成的图像偏模糊，单独使用效果有限，但其**编码器-隐空间-解码器**结构在扩散模型中被广泛复用。

### 扩散模型（Diffusion Model，2020 至今）

扩散模型以其**训练稳定、生成多样性高、可条件控制**的特点全面超越 GAN，成为图像生成的主流技术路线。

## 扩散模型原理

扩散模型的核心思想来自热力学中的扩散过程，分为两个阶段：

### 前向过程（加噪）

在训练时，对真实图像逐步添加高斯噪声，经过 T 步（通常 T=1000）后图像变为纯高斯噪声：

```
x_0（原始图像）
→ x_1（微小噪声）
→ x_2
→ ...
→ x_T（纯高斯噪声）
```

每步加噪由预定义的噪声调度（Noise Schedule）控制，数学上是马尔可夫链：

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) * x_{t-1}, β_t * I)
```

### 反向过程（去噪生成）

训练一个神经网络（通常是 UNet）学习**逆扩散**：给定任意时刻的噪声图像 x_t，预测添加的噪声 ε，从而逐步还原出干净图像：

```
x_T（纯高斯噪声）
→ x_{T-1}（去噪一步）
→ ...
→ x_0（生成的图像）
```

推理时，从标准高斯分布采样初始噪声，通过神经网络反复去噪，即可生成高质量图像。**文本条件**通过将文本 embedding 注入 UNet 的交叉注意力层来实现（Cross-Attention Conditioning）。

### 加速采样

原始 DDPM 需要 1000 步去噪，推理很慢。后续工作大幅加速：
- **DDIM**：确定性采样，20-50 步即可生成高质量图像
- **DPM-Solver**：数学优化的采样器，10-20 步达到 DDPM 1000 步效果
- **LCM（Latent Consistency Model）**：4-8 步实现高质量生成

## Stable Diffusion 架构

**Stable Diffusion（SD）** 是将扩散模型带入普通用户视野的里程碑开源项目。其核心创新是**潜在扩散模型（Latent Diffusion Model，LDM）**：在低维**潜在空间（Latent Space）**而非像素空间中进行扩散，大幅降低计算成本。

```
文本 Prompt → CLIP 文本编码器 → 文本 Embedding
                                      ↓
随机噪声 (Latent) → UNet 去噪（条件化）→ 去噪后的 Latent
                                      ↓
                                 VAE 解码器 → 生成图像
```

### 三大核心组件

- **VAE 编码器/解码器**：图像 ↔ 潜在空间的压缩/还原，压缩比通常为 8×（512×512 图像压缩为 64×64 的 latent）
- **CLIP 文本编码器**：将文本 Prompt 转化为语义 Embedding，引导扩散过程
- **UNet**：执行逐步去噪，通过交叉注意力接收文本条件

### SDXL 改进

2023 年发布的 SDXL 在 SD 1.5 基础上多项改进：
- 更大的 UNet（2.6B 参数）和更大的文本编码器（两个 CLIP 模型）
- 引入 Refiner 模型对生成结果进行高频细节增强
- 原生支持 1024×1024 分辨率
- 图像整体质量和文本遵循度显著提升

## FLUX.1：Diffusion Transformer

2024 年 Black Forest Labs（SD 原团队）发布 **FLUX.1**，将 UNet 替换为基于 Transformer 的去噪网络，引领图像生成进入 DiT（Diffusion Transformer）时代：

- **Multimodal Diffusion Transformer（MM-DiT）**：文本和图像 token 在同一 Transformer 中共同建模，实现更深度的文本-图像融合
- **Flow Matching**：比传统 DDPM 更高效的训练目标
- 三个版本：FLUX.1-dev（开源，研究用）、FLUX.1-schnell（快速，Apache 2.0）、FLUX.1-pro（闭源，商业）

FLUX.1 在文字渲染、手部细节、人体比例等传统扩散模型弱项上大幅提升。

## ControlNet 可控生成

**ControlNet（2023）** 允许在文本 Prompt 之外，额外添加结构性条件控制，包括：

- **Canny 边缘**：控制图像的轮廓结构
- **深度图**：控制场景的 3D 空间关系
- **人体姿态（OpenPose）**：精确控制人物姿态
- **线稿（Scribble/LineArt）**：将手绘草图转化为精细图像
- **法线贴图**：控制光照方向

ControlNet 的架构通过复制 UNet 编码器的权重，增加一个"控制"分支，以可训练的零卷积层注入条件，在保留 SD 原始能力的同时增加精确控制能力。

## LoRA 风格微调

**LoRA（Low-Rank Adaptation）** 是在扩散模型中引入特定风格/人物/概念的轻量微调方法：

- 只训练少量低秩矩阵（通常不超过 100MB），不修改原始模型权重
- 以 20-100 张图像、数十分钟训练即可学会特定人物面貌或艺术风格
- 可叠加使用多个 LoRA，并通过权重控制各 LoRA 的影响强度

LoRA 是 Stable Diffusion 生态爆发的核心推动力，Civitai 等平台上有数十万个用户创作的 LoRA 模型。

## 文生图 Prompt 技巧

- **质量词**：`masterpiece, best quality, ultra-detailed, 8k` 提升整体精度
- **风格词**：`oil painting, watercolor, digital art, anime style` 指定艺术风格
- **构图控制**：`close-up portrait, wide angle, bird's-eye view`
- **负向 Prompt**：使用 `Negative Prompt` 排除不需要的元素，如 `blurry, low quality, deformed hands`
- **权重调整**：`(关键词:1.3)` 加强权重，`[关键词:0.7]` 减弱权重

## 商业应用

- **设计辅助**：LOGO 设计、产品效果图、营销素材快速生成
- **游戏开发**：概念美术、贴图、角色设计
- **电影/影视**：故事板、分镜稿、视觉特效合成
- **电商**：商品图背景替换、模特换装
- **个性化定制**：个人写真、头像生成、礼品设计
