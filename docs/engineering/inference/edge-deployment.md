# 边缘部署

边缘部署（Edge Deployment）是指在用户本地设备上运行 AI 模型，而非通过网络调用云端服务。边缘部署的价值在于：数据隐私（数据不离开设备）、离线可用（无需网络连接）、低延迟（无网络往返开销）和成本节约（无 API 调用费用）。然而，边缘设备的硬件限制是核心挑战。

## 边缘设备限制

边缘部署面临的硬件约束与数据中心环境天壤之别：

**计算能力**：
- 消费级笔记本 CPU（Intel Core i7/i9，Apple M 系列）：算力约 0.1-10 TFLOPS
- 手机 SoC（Snapdragon 8 Gen 3，Apple A17 Pro）：NPU 算力约 10-35 TOPS
- 对比 H100 GPU：494 TFLOPS（FP16）

**内存限制**：
- 手机：8-16 GB 统一内存（与系统共享）
- 笔记本：16-64 GB RAM，无独立显存（或 iGPU 共享内存）
- 入门级边缘设备：1-8 GB
- 对比数据中心：A100 80GB 独立显存

**功耗约束**：
- 手机：约 5-10W（热设计功耗）
- 笔记本：15-45W（CPU/APU 功耗）
- 数据中心 GPU：300-700W

因此，边缘部署的核心是**在极为有限的资源内运行尽可能强大的模型**。

## llama.cpp

llama.cpp 是 Georgi Gerganov 于 2023 年开发的纯 C/C++ LLM 推理框架，是边缘部署领域影响力最大的项目。

### 核心特性

- **纯 CPU 推理**：完全不依赖 CUDA，任何 CPU 均可运行（ARM/x86/RISC-V）
- **Apple Silicon Metal 加速**：充分利用 M1/M2/M3 的 GPU（Metal API），性能远超纯 CPU
- **量化支持**：原生支持 GGUF 格式的各种量化级别（Q4_K_M、Q5_K_M 等）
- **极低依赖**：编译只需 C++ 编译器，无其他依赖
- **跨平台**：Linux、macOS、Windows、Android、iOS

### 安装与使用

```bash
# 编译（macOS Metal 加速）
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_METAL=1 -j8

# 下载 GGUF 模型（以 Llama-3.1-8B 为例）
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# 交互式对话
./llama-cli \
    -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -p "<|system|>You are a helpful assistant.</s><|user|>你好</s><|assistant|>" \
    -n 512 \
    --ctx-size 4096 \
    -ngl 99              # 将所有层卸载到 GPU（Metal）

# 启动 OpenAI 兼容 HTTP 服务
./llama-server \
    -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --port 8080 \
    -ngl 99
```

### Apple Silicon 性能参考

在 M3 Max（36GB 统一内存）上：
- LLaMA-3.1-8B Q4_K_M：约 60-80 token/s（Metal GPU 加速）
- LLaMA-3.1-70B Q4_K_M：约 10-15 token/s（需要 36GB+ 内存）

对比云端 API 的流式输出速度（约 50-80 token/s），Apple Silicon 的本地推理体验已接近云端水准。

## MLC LLM（移动端优化）

MLC LLM（Machine Learning Compilation LLM）由 MLC AI 团队开发，基于 Apache TVM 编译框架，专为移动端和 Web 端优化。

### 核心技术

- **TVM 编译优化**：针对目标硬件（ARM GPU、Metal、Vulkan、WebGPU）生成最优计算内核
- **模型量化**：支持 4-bit 量化（q4f16_1、q4f32_0 格式）
- **跨平台**：Android（OpenCL/Vulkan）、iOS（Metal）、macOS（Metal）、WebGPU

### Android 部署

```bash
# 编译 Android 应用
git clone https://github.com/mlc-ai/mlc-llm
cd mlc-llm

# 使用预编译的 Android APK
# 下载：https://mlc.ai/mlc-chat-android.apk

# 支持的模型（手机端）：
# - Llama-3.2-1B / 3B（最轻量，手机首选）
# - Phi-3.5-Mini（3.8B，Microsoft）
# - Qwen2-1.5B / 7B
```

### iOS 部署

MLC LLM 提供 Swift 包（MLC Swift Package），可集成到 iOS App：

```swift
import MLCSwift

let engine = try await MLCEngine()
await engine.reload(modelPath: "Llama-3.2-3B-Instruct-q4f16_1-MLC")

let reply = engine.chat.completions.create(
    messages: [ChatCompletionMessage(role: .user, content: "你好")]
)
```

## Ollama 本地部署

Ollama 将 llama.cpp 封装为用户友好的工具，是最适合开发者本地使用的方案：

```bash
# 一键安装（macOS/Linux）
curl -fsSL https://ollama.ai/install.sh | sh

# 拉取并运行模型（自动选择合适的量化版本）
ollama pull llama3.2:3b    # 2GB，适合 8GB 内存设备
ollama pull qwen2.5:7b     # 4.7GB，适合 16GB 内存设备
ollama pull deepseek-r1:8b # 5GB

# 对话
ollama run llama3.2:3b

# 后台服务（OpenAI 兼容 API on localhost:11434）
ollama serve
```

Ollama 自动处理：
- 模型下载和缓存（`~/.ollama/models/`）
- GGUF 量化选择（根据可用内存自动选择）
- Metal/CUDA 加速检测和启用
- OpenAI 兼容 API 服务启动

## GGUF 在边缘的应用

GGUF 格式是边缘部署的标准模型格式，其设计充分考虑了边缘场景的需求：

- **内存映射（mmap）**：GGUF 文件可以通过 mmap 直接映射到地址空间，无需将整个模型加载到内存——部分层可以按需加载，适合内存不足的设备
- **元数据嵌入**：模型架构、分词器、聊天模板等所有信息嵌入单个文件，无需额外配置
- **量化灵活性**：同一模型可以有多个量化版本，用户可根据设备选择

内存与量化等级的对应关系（8B 模型）：

| 内存 | 推荐量化 | 文件大小 | 速度（M3）|
|------|---------|---------|---------|
| 8 GB | Q4_0 | ~4.3 GB | ~45 t/s |
| 16 GB | Q5_K_M | ~5.7 GB | ~55 t/s |
| 32 GB | Q8_0 | ~8.5 GB | ~65 t/s |
| 64 GB+ | F16 | ~16 GB | ~70 t/s |

## WebLLM（浏览器端 WebGPU 推理）

WebLLM 是 MLC AI 团队开发的浏览器端 LLM 推理库，基于 WebGPU API（现代浏览器的 GPU 访问接口）：

```javascript
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

const engine = await webllm.CreateMLCEngine(
    "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    { initProgressCallback: (report) => console.log(report.text) }
);

const reply = await engine.chat.completions.create({
    messages: [{ role: "user", content: "你好" }],
    stream: true
});

for await (const chunk of reply) {
    process.stdout.write(chunk.choices[0].delta.content || "");
}
```

WebLLM 的限制：
- 模型首次运行需要下载（存储在 IndexedDB，后续可缓存）
- 浏览器 GPU 访问权限和内存限制（通常 2-4 GB）
- 需要支持 WebGPU 的现代浏览器（Chrome 113+、Edge 113+）
- Safari 对 WebGPU 支持尚不完整

适用场景：隐私敏感的前端应用（法律、医疗咨询），演示项目，无后端部署需求的应用。

## IoT 场景限制

在更受限的 IoT 设备上（如树莓派、微控制器），运行 LLM 的可行性极低：

- **树莓派 4（8GB RAM）**：可运行极小量化模型（1B Q4），速度约 1-3 token/s，勉强可用
- **树莓派 Zero / Arduino / ESP32**：内存仅 256KB-8MB，完全无法运行 LLM

IoT 场景的替代方案：
- **语义路由**：在设备端运行极小的分类模型（BERT-tiny 等），仅判断意图，复杂处理仍调用云端
- **KWS（关键词唤醒）**：设备端检测唤醒词，触发后发送音频到云端
- **TinyML**：在微控制器上运行专用小模型（图像分类、异常检测），不追求通用 LLM 能力

## 模型选择建议

针对边缘部署的模型选择指南（按设备内存分类）：

**4-8 GB 内存（低端手机/入门笔记本）**：
- Llama-3.2-1B（Q4，约 0.8 GB）：基础问答，速度快
- Phi-3.5-Mini-3.8B（Q4，约 2.3 GB）：微软优化，小尺寸高性能

**8-16 GB 内存（主流手机/中端笔记本）**：
- Llama-3.2-3B（Q4，约 2 GB）：对话能力良好
- Qwen2.5-7B（Q4，约 4.4 GB）：中文能力强，推荐中文场景

**16-32 GB 内存（高端笔记本/Apple Silicon）**：
- Llama-3.1-8B（Q5_K_M，约 5.7 GB）：能力均衡的首选
- Qwen2.5-14B（Q4_K_M，约 8.9 GB）：更强的推理能力

**32 GB+ 内存（Mac Studio/Mac Pro）**：
- Llama-3.1-70B（Q4_K_M，约 40 GB）：接近 GPT-3.5 水平的本地模型
- DeepSeek-R1-70B（量化版）：强推理能力

边缘部署的黄金法则：**选择能在目标设备上流畅运行（> 10 token/s）的最大量化模型**。速度低于 5 token/s 的体验难以实用。
