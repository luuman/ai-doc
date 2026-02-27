# 文生图技术

文本生成图像（Text-to-Image）是近年来 AI 应用最广泛的多模态能力之一，允许用户通过自然语言描述来生成高质量图像。从 2021 年的 DALL-E 到 2024 年的 FLUX.1，技术进步的速度令人叹为观止。

## 发展史

### DALL-E 1（OpenAI，2021）

第一个引起广泛关注的文生图系统。使用**自回归 Transformer**：将图像离散化为 token（使用 dVAE），与文本 token 拼接后用 GPT 风格的 Transformer 自回归生成。生成质量在当时令人印象深刻，但图像分辨率有限（256×256），概念组合能力不稳定。

### Stable Diffusion（Stability AI，2022）

**Stable Diffusion** 的开源发布是文生图领域的历史性事件。基于**潜在扩散模型（Latent Diffusion Model）**，首次将高质量文生图带给普通用户：

- 完全开源，任何人可以本地运行
- 消费级 GPU（8GB 显存）即可使用
- 支持无限制的创作（无内容审查限制）
- 催生了 WebUI、ComfyUI 等工具生态和 Civitai 等社区

### DALL-E 2（OpenAI，2022）

引入扩散模型，利用 CLIP 图像 Embedding 作为中间表示，生成质量大幅提升，图像分辨率达到 1024×1024。推出 Outpainting（图像扩展）和 Inpainting（局部填充）功能。

### Midjourney

Midjourney 是商业运营的闭源文生图服务，以其**独特的艺术风格**和用户友好的 Discord 界面迅速积累了大量用户。多个版本迭代（v3→v4→v5→v5.2→v6→v6.1）持续提升质量，是当前最受创意工作者欢迎的文生图工具之一。

### DALL-E 3（OpenAI，2023）

与前两代最大的区别是采用了重新描述（Re-captioning）技术：用 ChatGPT 对训练图像重新生成更详细、准确的描述，大幅改善了模型对复杂 Prompt 的理解能力。直接集成到 ChatGPT，降低了使用门槛。

### FLUX.1（Black Forest Labs，2024）

由 Stable Diffusion 原团队成员创立的 Black Forest Labs 发布，代表了当前开源文生图的技术巅峰，详见 [AI 图像生成](../vision/generation.md) 章节。

## 技术路径对比

| 技术路径 | 代表模型 | 优势 | 劣势 |
|---------|---------|------|------|
| 自回归（AR）| DALL-E 1、GPT-4o 图像生成 | 与 LLM 统一，可做图文交错生成 | 速度慢，分辨率受限 |
| 扩散模型（Diffusion）| SD、SDXL、FLUX.1 | 高质量，多步可控，生态丰富 | 推理需多步，慢于 AR |
| GAN | StyleGAN | 速度快 | 训练不稳定，多样性差，已被扩散模型超越 |
| Flow Matching | FLUX.1 | 训练更高效，采样更快 | 相对较新 |

## Stable Diffusion 生态

Stable Diffusion 开放源码后，围绕其形成了丰富的工具生态：

### AUTOMATIC1111 WebUI

最广泛使用的 SD 图形界面，通过浏览器操作：

```bash
# 安装
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
./webui.sh  # Linux/Mac
webui-user.bat  # Windows
```

功能：文生图、图生图、Inpainting、Outpainting、LoRA 加载、多种采样器选择、脚本批处理等。

### ComfyUI

基于节点图（Node Graph）的 SD 工作流工具，更灵活，更接近底层：

- 每个操作（加载模型、采样、解码、保存）都是一个节点
- 节点之间通过连线传递数据
- 支持复杂的多模型联动工作流
- 是专业用户和开发者的首选

### Civitai

全球最大的 SD 模型分享平台：
- 数十万个 Checkpoint、LoRA、Embedding、Hypernetwork 模型
- 用户共享生成的图片和对应的 Prompt 参数
- 支持模型标签和风格分类

## Midjourney 使用技巧

Midjourney 通过 Discord Bot 交互，主要参数：

```
/imagine prompt: a serene Japanese garden in autumn,
                 photorealistic, golden hour lighting --v 6.1 --ar 16:9 --stylize 200
```

### 常用参数

- `--v 6.1`：模型版本，v6.1 为当前最新
- `--ar 16:9`：宽高比（支持 1:1、2:3、16:9 等）
- `--style raw`：减少 Midjourney 的艺术性处理，更贴近 Prompt 描述
- `--stylize 0-1000`：风格化强度，越高越偏向 Midjourney 审美
- `--chaos 0-100`：随机性/变化度
- `--no [元素]`：排除不需要的元素（如 `--no text, watermark`）
- `--quality 0.25/0.5/1`：渲染质量，影响细节和速度

### Prompt 结构建议

```
[主题] + [环境/背景] + [风格] + [光照] + [相机参数（如摄影风格）]

示例：a young woman reading a book, cozy library interior,
      oil painting style, warm afternoon light, shallow depth of field
```

## FLUX.1 性能对比

FLUX.1 在 2024 年发布后迅速成为开源文生图的新标准：

| 对比项 | SDXL | FLUX.1-dev | Midjourney v6.1 |
|--------|------|-----------|-----------------|
| 文字渲染 | 差 | 优秀 | 良好 |
| 手部细节 | 较差 | 良好 | 良好 |
| 人体比例 | 一般 | 优秀 | 优秀 |
| 文本遵循度 | 中等 | 高 | 中等 |
| 开源可用 | 是 | 是（研究用）| 否 |
| 推理速度 | 较快 | 慢（模型大）| 中等 |

## 商业授权差异

| 模型 | 个人非商业 | 商业用途 | 注意事项 |
|------|-----------|---------|---------|
| Stable Diffusion 1.x/2.x | 自由 | 自由（CreativeML OpenRAIL-M）| 部分使用限制 |
| SDXL | 自由 | 自由（CreativeML OpenRAIL++-M）| 商业限制更少 |
| FLUX.1-schnell | 自由 | 自由（Apache 2.0）| 完全开放 |
| FLUX.1-dev | 自由（研究）| 需购买商业授权 | 不可用于商业 |
| Midjourney | 免费计划受限 | 付费计划可商业 | 需订阅 |
| DALL-E 3 | OpenAI 条款 | 允许（遵守政策）| 内容限制较多 |

## Prompt 工程在图像生成中的特殊技巧

文生图的 Prompt 与对话 AI 的 Prompt 有显著差异：

- **关键词堆叠**：扩散模型对关键词更敏感，通常使用逗号分隔的关键词列表而非完整句子
- **风格前置**：将最重要的风格描述放在 Prompt 前面（权重更高）
- **负向 Prompt**：专门告诉模型**不要**生成什么，是扩散模型独有的控制方式

```
# 正向 Prompt
masterpiece, best quality, ultra-detailed, 1girl, silver hair, blue eyes,
school uniform, cherry blossoms, soft bokeh, natural lighting

# 负向 Prompt
(worst quality:1.4), (low quality:1.4), blurry, deformed, ugly,
extra limbs, bad anatomy, watermark, text
```

- **括号权重**：`(关键词:1.3)` 增加权重，`[关键词:0.7]` 降低权重（SD 风格）
- **分步控制**：`[关键词::0.3]` 仅在去噪的前 30% 步数应用该关键词
