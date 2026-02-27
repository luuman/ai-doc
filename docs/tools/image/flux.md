# FLUX.1

FLUX.1 是由 Black Forest Labs（BFL）于 2024 年 8 月发布的新一代图像生成模型。Black Forest Labs 由 Stable Diffusion 的原班人马创立（包括 Robin Rombach，LDM 论文第一作者），FLUX.1 代表了扩散模型技术的最新突破，在多个关键质量维度上超越了此前的 SDXL 和 SD3。

## FLUX.1 架构（Diffusion Transformer）

FLUX.1 抛弃了 Stable Diffusion 系列使用的 U-Net 架构，改用 **Diffusion Transformer（DiT）** 作为核心网络：

**DiT 的核心创新**：
- 将扩散模型的去噪网络替换为 Transformer 架构（而非 U-Net）
- 多模态文本-图像 Transformer（MMDiT）：文本 Token 和图像 Token 在同一 Transformer 内交互，实现更深度的文本-图像对齐
- 12B 参数规模（FLUX.1 [pro]），远大于 SDXL 的约 3.5B 参数

**Flow Matching（流匹配）**：
- FLUX.1 采用 Flow Matching 训练范式（而非传统 DDPM 扩散）
- 更平滑的学习目标，使用更少的采样步数即可生成高质量图像
- 与 SD3 类似，但在实现细节上有所不同

## 三个版本

| 版本 | 许可证 | 推理速度 | 图像质量 | 适用场景 |
|------|--------|---------|---------|---------|
| FLUX.1 [schnell] | Apache 2.0 | 极快（4步出图）| 良好 | 本地快速预览、实时生成 |
| FLUX.1 [dev] | 非商业许可 | 中（25-50步）| 优秀 | 本地高质量生成、研究 |
| FLUX.1 [pro] | 仅 API | 中 | 最佳 | 商业 API 调用 |

- **schnell**（德语"快速"）：速度优先，4-8 步即可完成生成，适合实时应用
- **dev**：quality 优先，BFL 发布的开源高质量版本，不允许商业使用
- **pro**：BFL 自研的最强版本，只通过 Replicate 和 fal.ai 等平台的 API 提供

## 与 SDXL 的对比

FLUX.1 在以下维度有显著提升：

### 细节质量

FLUX.1 在皮肤质感、布料纹理、环境光照等细节上明显优于 SDXL。12B 的参数量赋予了模型更强的细节表现能力。

### 文字渲染

这是 SDXL 的长期短板，也是 FLUX.1 最引人注目的突破。FLUX.1 能够准确渲染 Prompt 中指定的文字：

```
Prompt: a sign that reads "Hello World" in red letters
SDXL: 通常生成模糊或错误的文字
FLUX.1: 能够准确生成"Hello World"文字
```

对于需要包含文字的设计（海报、标牌、名片）场景，FLUX.1 几乎是唯一可选的开源模型。

### 人体比例

SDXL 在生成全身人物时经常出现手部变形、身体比例失调等问题，FLUX.1 在这方面有明显改善，手部细节尤为突出。

### Prompt 遵循

FLUX.1 对复杂 Prompt 的理解和遵循能力更强，能够处理包含多个主体、复杂空间关系的 Prompt。

## 本地部署需求

FLUX.1 的参数规模导致显存要求远高于 SD1.5 和 SDXL：

| 版本 | 完整精度（FP16）| 量化版（Q4） | 推荐显存 |
|------|--------------|-----------|--------|
| schnell | ~23 GB | ~8 GB | 16 GB+ |
| dev | ~23 GB | ~8 GB | 16 GB+ |
| dev (NF4 量化) | - | ~7 GB | 12 GB |

在消费级 GPU（RTX 4090 24GB）上可以运行 FP16 版本，但速度较慢。Q4/Q8 量化版本（通过 GGUF 格式）可在 16 GB 显存上流畅运行。

Apple Silicon（Mac）：M3 Max/Ultra 或 M4 Pro/Max（36GB+ 统一内存）可以通过 llama.cpp 或 ComfyUI 运行量化版 FLUX.1，速度约 2-5 分钟/张。

## ComfyUI FLUX 工作流

FLUX.1 的 ComfyUI 工作流比 SDXL 更简单（无需单独的 VAE 节点配置），但 Sampler 设置不同：

```
[FLUX Checkpoint Loader] → [CLIP Text Encode (FLUX)] → [KSampler]
                                                            ↓
                                                    [VAE Decode]
                                                            ↓
                                                    [Save Image]
```

关键配置：
- **Sampler**：`euler` 或 `dpmpp_2m`
- **Scheduler**：`simple`（FLUX 专用）
- **Steps**：schnell 版 4 步；dev 版 25-30 步
- **CFG Scale**：schnell 版 `1.0`；dev 版 `3.5`（FLUX 的 CFG 范围与 SD 不同）

社区提供了多种优化的 FLUX ComfyUI 工作流，包括 ControlNet 集成、LoRA 加载、图像放大等完整流程。

## FLUX API（Replicate / fal.ai）

对于不想本地部署的用户，通过 API 调用 FLUX.1 [pro]：

### Replicate API

```python
import replicate

output = replicate.run(
    "black-forest-labs/flux-pro",
    input={
        "prompt": "A majestic dragon flying over a ancient city at sunset",
        "width": 1440,
        "height": 1440,
        "steps": 25,
        "guidance": 3.5,  # FLUX 的 CFG 等效参数
        "seed": 42
    }
)

# 返回图像 URL
image_url = output[0]
```

### fal.ai API

```python
import fal_client

result = fal_client.subscribe(
    "fal-ai/flux/dev",
    arguments={
        "prompt": "your prompt here",
        "image_size": "landscape_4_3",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1
    }
)

image_url = result["images"][0]["url"]
```

定价（大约）：
- Replicate FLUX.1 [pro]：约 $0.055/张（1MP）
- fal.ai FLUX.1 [dev]：约 $0.025/张

## FLUX LoRA 训练

FLUX.1 的 LoRA 训练生态正在快速发展：

**训练工具**：
- **ostris/ai-toolkit**：目前最常用的 FLUX LoRA 训练工具
- **kohya_ss**（正在添加 FLUX 支持）

**训练特点**：
- 训练数据：20-200 张高质量图像
- 训练步数：500-2000 步
- 显存要求：24 GB+（FLUX 参数量大，梯度检查点必须启用）
- LoRA Rank：通常 4-16

**与 SD LoRA 的区别**：FLUX LoRA 文件与 SDXL LoRA 不兼容，需要使用 FLUX 专属格式。但 ComfyUI 提供了统一的加载节点。

## 当前最佳实践

**本地使用（24 GB 显存）**：
- 使用 ComfyUI + FLUX.1 [dev]（FP8 量化版，约 12 GB 显存）
- Sampler：euler，Steps：25，CFG：3.5
- 分辨率：1024×1024 或 1024×1360（16:9 或 3:4）

**API 使用**：
- 生产应用：FLUX.1 [pro] via Replicate/fal.ai
- 快速原型：FLUX.1 [schnell] via fal.ai（成本更低）

**Prompt 技巧**：
- FLUX 比 SDXL 更理解自然语言，不需要过多风格词和权重语法
- 直接描述想要的内容，如"一张专业的商业宣传照"效果优于堆砌关键词
- 文字渲染时用引号明确标出要显示的文字："a poster that says 'Open 24/7'"
- 避免使用 SD 的 Prompt 权重语法（`(word:1.5)` 在 FLUX 中无效）

**FLUX vs Midjourney 选择**：
- 需要文字渲染 → FLUX
- 需要完全本地/私有 → FLUX（open weights）
- 追求最简单操作、最高审美水准 → Midjourney
- 需要高可控性（ControlNet、精确构图）→ FLUX + ComfyUI
