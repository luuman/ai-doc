# Stable Diffusion

Stable Diffusion（SD）是由 Stability AI 于 2022 年开源发布的图像生成模型，是 AI 图像生成领域最重要的开源项目。与 Midjourney 等闭源服务不同，SD 完全开源，可在本地运行，生态系统极为丰富。无论是个人创作者还是需要高度定制化的专业用户，SD 都是不可绕过的工具。

## SD 基础（Latent Diffusion Model）

Stable Diffusion 基于**潜在扩散模型**（Latent Diffusion Model，LDM）：

**工作原理**：
1. **前向扩散（加噪）**：训练阶段，将图像逐步加入随机噪声，直到变为纯噪声
2. **反向扩散（去噪）**：推理阶段，从随机噪声开始，通过 U-Net 逐步去噪，恢复图像
3. **潜在空间**：不在像素空间操作，而在经过 VAE 压缩的低维潜在空间操作（通常 8x 压缩），大幅降低计算量
4. **条件引导**：通过 CLIP 文本编码器将 Prompt 转换为向量，引导去噪方向

核心组件：
- **VAE（Variational Autoencoder）**：像素 ↔ 潜在空间的编解码器
- **U-Net**：带交叉注意力的去噪网络，是模型的核心
- **CLIP Text Encoder**：将文字 Prompt 编码为 Prompt Embedding

## WebUI（AUTOMATIC1111）

AUTOMATIC1111（A1111）是最流行的 SD 图形界面，几乎成为 SD 社区的标准工具：

### 安装

```bash
# 前提：安装 Python 3.10，NVIDIA CUDA 或 AMD ROCm

# 克隆仓库
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# macOS / Linux 一键启动（自动下载依赖）
./webui.sh

# Windows
webui-user.bat
```

启动后访问 `http://127.0.0.1:7860`

### 核心界面

- **txt2img**：文字生成图像（最常用）
- **img2img**：以现有图像为基础进行修改
- **Extras**：图像放大、清晰度增强
- **PNG Info**：读取图像的生成参数（方便复现）

## ComfyUI 节点工作流

ComfyUI 是另一种主流 SD 界面，采用节点图（Node Graph）工作流，比 A1111 更灵活但学习曲线更陡：

```
[CLIP Text Encode] → [KSampler] → [VAE Decode] → [Save Image]
      ↑                  ↑
[Load Checkpoint]    [Empty Latent]
```

ComfyUI 的优势：
- 完全可视化的数据流，清楚看到每个步骤
- 支持复杂的条件分支（多个 ControlNet 并联）
- 工作流可以保存为 JSON 文件，方便分享和复用
- API 接口完善，适合生产集成

## 模型选择

### SD1.5（Stable Diffusion 1.5）

- 分辨率：512×512（原生）
- 显存要求：4-6 GB
- 优点：LoRA 和 ControlNet 生态最成熟，模型数量最多（Civitai 上 90% 的模型）
- 缺点：质量相对较低，尤其是手部和复杂场景

### SDXL（Stable Diffusion XL）

- 分辨率：1024×1024（原生）
- 显存要求：8-10 GB
- 优点：比 SD1.5 质量大幅提升，支持更高分辨率
- 缺点：生成速度较慢，显存要求高，LoRA 生态较 SD1.5 少

### SD3 / SD3.5

- 架构：Diffusion Transformer（MMDiT）
- 特点：文字渲染能力大幅提升，构图能力更强
- 显存要求：8-16 GB

### Pony Diffusion（Pony）

- 基于 SDXL 微调，专注于动漫/卡通风格
- Civitai 上最受欢迎的风格化模型之一
- 需要特定的 Prompt 格式（score_9, score_8_up 等质量标签）

## ControlNet（姿势/深度/线稿控制）

ControlNet 是 SD 最重要的扩展之一，通过额外的条件图像精确控制生成结果：

**主要 ControlNet 类型**：
- **OpenPose**：输入骨骼关键点图，控制人物姿势
- **Depth**：输入深度图，控制空间关系和透视
- **Canny / Lineart**：输入线稿，保持轮廓结构生成（着色效果）
- **IP-Adapter**：输入参考图像，迁移图像风格（非 ControlNet 但类似）
- **Inpainting**：局部重绘，精确编辑图像特定区域
- **Tile（分块）**：高清放大时保持细节一致性

```
使用示例：
1. 上传一张人物照片 → OpenPose 提取骨骼 →
   用提取的骨骼 + 新的外貌 Prompt 生成新图
   （保持姿势，改变外貌/风格）
```

## LoRA 训练（风格/角色）

LoRA（Low-Rank Adaptation）是 SD 模型微调的轻量级方法，可以在小数据集上高效训练特定风格或角色：

**训练步骤**：
1. 收集 20-100 张目标风格/角色图像
2. 使用 WD14 或 BLIP-2 自动打标注（Caption）
3. 配置训练参数（Network Dim、Alpha、学习率）
4. 训练（通常 1000-3000 步）
5. 测试生成效果，调整 LoRA 权重（0.5-1.0）

常用训练工具：
- **kohya_ss**：功能最完整的 LoRA 训练 WebUI
- **DreamBooth**：人物特定化训练（更高显存要求）

## 采样器（Euler/DPM++）

采样器决定去噪过程的算法，影响速度和质量：

| 采样器 | 速度 | 质量 | 推荐步数 | 适用场景 |
|--------|------|------|---------|---------|
| Euler a | 快 | 好 | 20-30 | 日常生成 |
| DPM++ 2M Karras | 中 | 优秀 | 20-30 | 高质量生成（推荐） |
| DPM++ SDE Karras | 慢 | 最佳 | 20-30 | 质量优先 |
| DDIM | 快 | 中 | 30-50 | img2img |
| UniPC | 快 | 好 | 15-20 | 快速预览 |
| LCM | 极快 | 中 | 4-8 | 实时预览 |

**推荐默认**：DPM++ 2M Karras，Steps=25，是速度和质量的最佳平衡。

## 常用参数

- **CFG Scale（Classifier-Free Guidance Scale）**：控制 Prompt 与图像的相关程度。7-9 为常用范围，数值越高越"听话"但越僵硬，越低越随机创意。
- **Steps（采样步数）**：一般 20-30 步足够，过多步数边际收益递减。
- **Seed**：随机种子，固定 Seed 可以复现相同结果；-1 为随机。
- **Negative Prompt**：描述不想出现的内容（如 `ugly, deformed, blurry, watermark`）。
- **Hires. fix**：先生成小图再放大，有效避免高分辨率下的构图崩坏。

## Civitai 模型社区

Civitai（civitai.com）是最大的 SD 模型社区：

- **Checkpoint**：完整的基础模型（GB 级别）
- **LoRA**：轻量级风格/角色模型（MB 级别）
- **Embedding（Textual Inversion）**：嵌入到 Prompt 中的概念向量（KB 级别）
- **VAE**：改善颜色和细节的独立组件

搜索技巧：
- 按 Rating 排序找高质量模型
- 查看示例图的生成参数（Prompt、采样器、LoRA 权重）
- 注意 NSFW 内容的许可条款
- 下载时选择与你的 SD 版本匹配的模型（SD1.5 vs SDXL）

## 工作流建议

**标准写真工作流**：
1. 选择写实风格 SDXL Checkpoint
2. Txt2img，1024×1024，DPM++ 2M Karras，25 步
3. 启用 Hires. fix 放大到 2048×2048
4. 用 ControlNet Inpainting 修复手部等细节
5. 用 ADetailer 扩展自动修复面部

**动漫插画工作流**：
1. Pony 系列 Checkpoint + 角色 LoRA
2. 768×1152（竖版），Euler a，25 步
3. 写反向 Prompt 避免常见问题（worst quality, bad anatomy）
4. 用 ControlNet Lineart 保持线稿参考
