# ComfyUI

ComfyUI 是一个基于节点图（Node-based）的 Stable Diffusion/FLUX 图像生成界面，由 comfyanonymous 开发并于 2023 年开源。与 AUTOMATIC1111 WebUI 的表单式操作不同，ComfyUI 将图像生成流程可视化为一张数据流图，每个节点代表一个操作，节点间的连线代表数据流动。这种设计赋予了 ComfyUI 极大的灵活性，使其成为高级用户和专业工作流的首选。

## 节点工作流设计理念

ComfyUI 的核心理念是**将图像生成过程理解为一条有向无环图（DAG）**：

- 每个**节点（Node）**执行一个特定操作（加载模型、编码文本、采样、解码）
- 节点的**输入/输出端口**定义了可以连接的数据类型（MODEL、CONDITIONING、LATENT、IMAGE）
- 用户通过连线（Wire）将节点串联，构建完整的生成流程

这种设计的优势：
- **透明性**：清晰看到数据在每个步骤如何流动和变换
- **灵活性**：任意节点可以被替换或扩展，无需修改代码
- **可复用性**：将常用流程保存为工作流 JSON 文件，一键分享
- **调试便利**：可以在任意节点之间插入预览节点，观察中间结果

## 安装与基础工作流

### 安装

```bash
# 方法1：直接安装
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt

# 将模型放到 models/checkpoints/ 目录
# 启动
python main.py --port 8188

# 方法2：使用 ComfyUI Desktop（官方桌面版，2024年推出）
# 访问 comfy.org 下载安装包，内置 Python 环境，一键安装
```

### 最简基础工作流（5个节点）

```
[Load Checkpoint] ──MODEL──→ [KSampler] ──LATENT──→ [VAE Decode] ──IMAGE──→ [Save Image]
         └──VAE──────────────────┘                       ↑
                                                         └── VAE
[CLIP Text Encode (pos)] ──CONDITIONING──→ [KSampler]
[CLIP Text Encode (neg)] ──CONDITIONING──→ [KSampler]
[Empty Latent Image] ──────────LATENT──→ [KSampler]
```

这 5 类节点（Load Checkpoint、CLIP Text Encode、Empty Latent Image、KSampler、VAE Decode + Save Image）构成了所有 SD 工作流的核心骨架。

## 与 WebUI 的根本区别

AUTOMATIC1111 和 ComfyUI 虽然都用于 Stable Diffusion，但设计哲学截然不同：

| 维度 | AUTOMATIC1111 | ComfyUI |
|------|--------------|---------|
| 界面范式 | 表单（Form-based） | 节点图（Node-based） |
| 学习曲线 | 平缓，上手快 | 陡峭，需要理解数据流 |
| 灵活性 | 有限（功能固定，扩展靠脚本）| 极高（节点自由组合） |
| 工作流共享 | 难（只能分享参数）| 易（JSON 文件包含完整流程）|
| 多工作流 | 单一工作流 | 可以同时有多个工作流 |
| 性能 | 较慢 | 更快（更优化的执行引擎）|
| 调试 | 黑盒 | 透明（可检查任意中间结果）|
| 专业工作流 | 困难 | 非常适合 |

总结：WebUI 适合**入门和日常使用**，ComfyUI 适合**高级用户、专业工作流、研究和生产集成**。

## 核心节点解析

### Load Checkpoint

加载 SD/FLUX 模型，输出 MODEL、CLIP、VAE 三个关键组件。

### CLIP Text Encode

将文本 Prompt 通过 CLIP 编码为 Conditioning 向量。正向和负向 Prompt 分别有独立节点。

**FLUX 专用**：FLUX.1 使用双重文本编码器（CLIP-L + T5-XXL），需要使用 "Dual CLIP Loader" 节点。

### KSampler

执行扩散采样的核心节点，参数：
- `seed`：随机种子
- `steps`：采样步数
- `cfg`：CFG Scale（对 FLUX 使用 1.0 或 3.5）
- `sampler_name`：采样器算法
- `scheduler`：调度器
- `denoise`：去噪强度（1.0 = txt2img，0.5 = img2img 保留一半原图特征）

### VAE Decode / VAE Encode

VAE Decode：将潜在空间（Latent）解码为像素图像（必须在显示/保存前使用）
VAE Encode：将像素图像编码为潜在空间（img2img 工作流必须）

### Preview Image / Save Image

Preview：在界面中实时显示生成结果
Save Image：保存到磁盘（output/ 目录或自定义路径）

## ControlNet 工作流

ControlNet 在 ComfyUI 中通过额外节点串联实现：

```
[Load Image (骨骼图)] ──→ [DWPose Preprocessor] ──POSE──→ [Apply ControlNet]
                                                                    ↓CONDITIONING
[CLIP Text Encode] ──CONDITIONING──→ [Apply ControlNet] ──→ [KSampler]
                                                   ↑
                                     [ControlNet Loader]
```

多个 ControlNet 可以串联（一个 ControlNet 的输出 Conditioning 作为下一个的输入），实现同时控制姿势和深度等多个条件。

## IPAdapter（风格迁移）

IPAdapter 是一种在不使用 ControlNet 的情况下，将参考图像的"风格"迁移到生成图像的技术：

```
[Load IPAdapter Model] ──────────────────────────────────────→ [IPAdapter]
[Load CLIP Vision] ─→ [CLIP Vision Encode (参考图)] ──IMAGE_EMBEDS──→ [IPAdapter]
                                                                    ↓MODEL
                                                              [KSampler]
```

应用场景：
- **风格迁移**：将一张图的整体风格（色调、笔触）转移到新图
- **人物一致性**：保持人物外貌在不同 Prompt 下的一致性（类似 IP 形象生成）
- **产品展示**：将产品图放到不同的场景中

## AnimateDiff（视频生成）

AnimateDiff 是基于 SD 的视频生成扩展，在 ComfyUI 中通过 `ComfyUI-AnimateDiff-Evolved` 节点包支持：

```
[AnimateDiff Loader] ──→ [Apply AnimateDiff Model]
                                    ↓
[CLIP Text Encode] ──→ [KSampler] ──→ [VAE Decode] ──→ [Video Combine]
```

工作原理：
- 在 SD 的 U-Net 每层插入时序注意力模块（Temporal Attention）
- 将多帧潜在向量一起处理，确保帧间连贯性
- 通常生成 16-32 帧（约 1-2 秒），可通过帧插值扩展

## Upscaling 工作流

高清放大工作流（Hi-Res Fix 的 ComfyUI 实现）：

```
[生成 512×512 基础图] ──→ [Latent Upscale / Image Upscale] ──→ [二次 KSampler (denoise=0.5)]
```

常用放大模型（放到 models/upscale_models/ 目录）：
- **ESRGAN / RealESRGAN**：通用超分辨率，4x 放大
- **LDSR**：基于扩散模型的放大，质量最高但速度慢
- **Ultimate SD Upscale**：将图像分块放大后拼接，可处理超大分辨率

**Tiled Upscaling 工作流**：
1. 生成 768×768 基础图
2. 用 ESRGAN 快速放大到 3072×3072（4x）
3. 用 `UltimateSDUpscale` 节点分块（512×512 tiles）对每块进行 img2img（denoise=0.3）
4. 自动拼接生成无缝的超高分辨率图

## 自定义节点生态

ComfyUI 的自定义节点（Custom Nodes）通过 ComfyUI-Manager 安装：

```
# 安装 ComfyUI-Manager
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager

# 重启 ComfyUI，在 Manager 界面搜索并安装节点包
```

重要节点包：
- **ComfyUI-Manager**：节点包管理器（必装）
- **rgthree-comfy**：增强的节点组、备注功能
- **ComfyUI-Impact-Pack**：人脸修复、Mask 操作（ADetailer 功能）
- **ComfyUI_IPAdapter_plus**：IPAdapter 支持
- **ComfyUI-AnimateDiff-Evolved**：视频生成
- **comfyui-workspace-manager**：工作流版本管理
- **ComfyUI-GGUF**：支持 GGUF 量化模型（FLUX 量化版）

## API 调用（生产集成）

ComfyUI 提供 REST API，可以在生产系统中编程调用：

```python
import json
import requests
import websocket
import uuid

# 加载工作流 JSON（从 ComfyUI 界面导出）
with open("workflow.json", "r") as f:
    workflow = json.load(f)

# 修改工作流参数
workflow["3"]["inputs"]["seed"] = 42
workflow["6"]["inputs"]["text"] = "your custom prompt"

client_id = str(uuid.uuid4())

# 提交工作流
response = requests.post(
    "http://127.0.0.1:8188/prompt",
    json={"prompt": workflow, "client_id": client_id}
)
prompt_id = response.json()["prompt_id"]

# 通过 WebSocket 监听完成事件
ws = websocket.WebSocket()
ws.connect(f"ws://127.0.0.1:8188/ws?clientId={client_id}")

while True:
    message = json.loads(ws.recv())
    if message["type"] == "executing" and message["data"]["node"] is None:
        break  # 生成完成

# 获取生成结果
history = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}").json()
image_data = requests.get(f"http://127.0.0.1:8188/view?filename={filename}&type=output")
```

生产使用建议：
- 多个 ComfyUI 实例 + Nginx 负载均衡（横向扩展）
- 使用 `--listen 0.0.0.0` 允许外部访问（注意加认证保护）
- 将工作流 JSON 版本化管理（Git）
- 监控 GPU 利用率和队列深度
