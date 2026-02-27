# ONNX 与 TensorRT

在将训练好的 AI 模型部署到生产环境时，往往需要跨越框架边界（从 PyTorch 训练到 C++ 推理），或针对特定硬件进行深度优化。ONNX 提供了跨框架的模型交换标准，TensorRT 则是 NVIDIA 硬件上最强的推理优化引擎。两者的结合构成了企业级模型部署的重要路径。

## ONNX 格式

ONNX（Open Neural Network Exchange）是一种开放的模型表示格式，由 Microsoft 和 Facebook 于 2017 年联合发布，旨在解决深度学习框架之间的互操作性问题。

ONNX 的核心是一套与框架无关的**计算图（Computation Graph）**表示：

- 节点（Node）：表示操作（如 Conv、MatMul、ReLU）
- 张量（Tensor）：节点之间传递的数据
- 属性（Attribute）：操作的超参数（如卷积的 stride、dilation）

ONNX 使用 Protocol Buffers 序列化，文件扩展名为 `.onnx`。

支持 ONNX 的框架和工具：
- **导出**：PyTorch、TensorFlow、scikit-learn、XGBoost 等几乎所有主流框架
- **推理**：ONNX Runtime、TensorRT、OpenVINO、CoreML（通过 coremltools 转换）、NCNN

## PyTorch 导出 ONNX

### torch.onnx.export（传统方法）

```python
import torch
import torch.onnx

model = MyModel()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,               # ONNX Opset 版本
    do_constant_folding=True,       # 常量折叠优化
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},    # 支持动态 batch
        "output": {0: "batch_size"}
    }
)
```

### torch.export + torch.onnx.dynamo_export（新方法，PyTorch 2.0+）

```python
# 使用 TorchDynamo 导出，支持更多 Python 控制流
export_program = torch.export.export(model, (dummy_input,))
onnx_program = torch.onnx.dynamo_export(export_program, dummy_input)
onnx_program.save("model.onnx")
```

### 常见导出问题

- **动态控制流**：Python 的 if/for 在 Tracing 时被固化，需要用 `torch.jit.script` 处理
- **自定义算子**：非标准算子需要手动注册 ONNX 符号函数
- **Opset 兼容性**：不同 opset 版本支持的算子不同，推荐使用 opset 17 或以上

## ONNX Runtime

ONNX Runtime（ORT）是 Microsoft 开发的跨平台推理引擎，支持在 CPU、GPU、NPU 等多种硬件上运行 ONNX 模型。

```python
import onnxruntime as ort
import numpy as np

# 创建推理 Session
session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]  # 优先使用 GPU
)

# 查看输入输出信息
print(session.get_inputs()[0].name)   # 输入名
print(session.get_outputs()[0].name)  # 输出名

# 推理
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {"input": input_data})
```

ORT 支持的 Execution Provider（EP）：
- `CPUExecutionProvider`：通用 CPU 推理，默认可用
- `CUDAExecutionProvider`：NVIDIA GPU 加速
- `TensorrtExecutionProvider`：TensorRT 集成（最高性能）
- `OpenVINOExecutionProvider`：Intel 硬件优化
- `CoreMLExecutionProvider`：Apple 设备（Mac/iOS）加速
- `DirectMLExecutionProvider`：Windows 下 DirectML 加速（AMD/Intel GPU）

ORT 的 Graph Optimization 功能会自动进行以下优化：
- 算子融合（如 LayerNorm、Attention 融合）
- 常量传播与折叠
- 内存布局优化

## TensorRT

TensorRT（TRT）是 NVIDIA 专为其 GPU 硬件设计的推理优化引擎，通常能在 ONNX Runtime + CUDA 的基础上再提升 2-5 倍推理速度。

### 核心优化技术

**Layer Fusion（层融合）**：将多个连续的小算子合并为一个大算子，减少内存读写和 Kernel 启动开销。例如：
- Conv + BN + ReLU → 融合为一个 Kernel
- LayerNorm（多个 Elementwise 操作）→ 单个 Fused Kernel

**Kernel Auto-tuning（内核自动调优）**：对同一操作，TensorRT 在首次构建时会对多种 CUDA Kernel 实现进行性能测试（称为 Profiling/Timing），选择在当前 GPU 上最快的实现。这是 TensorRT 优化离线（Build Phase）完成的原因。

**精度校准（INT8 Calibration）**：
- 收集校准数据集（通常 100-1000 个样本）
- 统计每层激活值的分布范围
- 选择最优的量化缩放因子（最小化量化误差）

**内存优化**：TensorRT 自动分析计算图中各层的内存依赖，将不再需要的中间结果的内存重用给后续层，降低峰值显存占用。

### TensorRT 构建与推理

```python
import tensorrt as trt
import pycuda.driver as cuda

# 构建 TRT Engine（离线步骤）
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
config.set_flag(trt.BuilderFlag.FP16)  # 启用 FP16

engine = builder.build_serialized_network(network, config)
with open("model.trt", "wb") as f:
    f.write(engine)
```

### TRT-LLM（大模型优化推理）

NVIDIA TensorRT-LLM 是专为大语言模型设计的推理优化库：

- 基于 TensorRT 构建，针对 LLM 的 Attention 机制深度优化
- 支持 FP16/BF16/INT8/FP8 精度
- 内置 Continuous Batching、PagedAttention 支持
- 支持 Tensor Parallelism（多 GPU 推理）
- 提供 Triton Inference Server 集成

```bash
# TRT-LLM 编译 LLaMA 模型示例
python convert_checkpoint.py \
    --model_dir llama-2-7b-hf \
    --output_dir tllm_checkpoint \
    --dtype float16

trtllm-build \
    --checkpoint_dir tllm_checkpoint \
    --output_dir engine_output \
    --gpt_attention_plugin float16 \
    --max_batch_size 32
```

## OpenVINO（Intel 优化）

OpenVINO（Open Visual Inference & Neural network Optimization）是 Intel 针对其硬件（CPU、iGPU、VPU、FPGA）的推理优化框架：

```python
from openvino.runtime import Core

core = Core()
model = core.read_model("model.onnx")
compiled = core.compile_model(model, "CPU")

input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = compiled(input_tensor)
```

OpenVINO 的优化亮点：
- 自动利用 Intel CPU 的 AVX-512 和 AMX 指令集
- 支持 INT8 量化（与 ONNX 量化兼容）
- 对 Intel Arc GPU 和 Movidius VPU 的专项优化

## 适用场景对比

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| NVIDIA 数据中心，追求极致性能 | TensorRT / TRT-LLM | 最高推理速度 |
| 跨平台部署，多硬件支持 | ONNX Runtime | 一份模型，多平台适配 |
| Intel 服务器/边缘设备 | OpenVINO | Intel 硬件深度优化 |
| 嵌入式 Linux / ARM | ONNX Runtime (EP=CPU) 或 NCNN | 轻量，无 CUDA 依赖 |
| Apple Mac (M 系列) | CoreML (via CoreMLTools 转换) | Metal GPU 加速 |

## 与 vLLM 的对比

TensorRT-LLM 与 vLLM 是两个主要的大模型推理框架，各有侧重：

| 维度 | TensorRT-LLM | vLLM |
|------|-------------|------|
| 开发方 | NVIDIA | UC Berkeley（开源社区） |
| 优化深度 | 更深（硬件级优化） | 中等（PyTorch 级别） |
| 易用性 | 复杂（需编译 Engine） | 简单（pip install 即用） |
| 模型支持 | 主流 LLM，需手动适配新模型 | 广泛，HuggingFace 模型直接支持 |
| 更新速度 | 较慢 | 快速跟进新模型 |
| 最适场景 | 追求极致性能的生产部署 | 快速上线、模型多样性场景 |

在 NVIDIA GPU 上，TensorRT-LLM 通常比 vLLM 快 20-50%，但工程化成本更高。对于大多数应用，vLLM 是更务实的选择。
