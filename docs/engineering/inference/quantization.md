# 模型量化

模型量化（Quantization）是将模型权重和/或激活值从高精度浮点数（FP32/BF16）转换为低精度整数（INT8/INT4）的过程。量化可以显著减少模型的存储空间和推理时的显存占用，同时在支持整数运算的硬件上加速推理。量化是在有限硬件上部署大模型的核心技术。

## 量化原理

量化的基本操作是将浮点数映射到整数区间。以对称量化（Symmetric Quantization）为例：

```
量化：q = round(x / scale)
反量化：x ≈ q × scale
```

其中 `scale`（缩放因子）= `max(|x|) / (2^(n-1) - 1)`，n 为量化位数。

非对称量化（Asymmetric Quantization）额外引入 `zero_point`（零点）：

```
量化：q = round(x / scale) + zero_point
反量化：x ≈ (q - zero_point) × scale
```

量化误差（Quantization Error）= `|x - dequantize(quantize(x))|`，量化位数越低，误差越大。

## 量化粒度

量化粒度决定了 scale 和 zero_point 的共享范围：

- **逐张量量化（Per-tensor）**：整个权重矩阵共享一个 scale，实现简单但精度损失较大
- **逐通道量化（Per-channel）**：每个输出通道（行）有独立的 scale，精度更好，硬件支持广泛
- **逐组量化（Per-group / Group-wise）**：将每个通道进一步分成 group（通常 64 或 128 个元素），每组有独立 scale；精度最好，但 scale 本身的存储开销增加

Group-wise 量化是 INT4 量化的标准实现（如 GPTQ、AWQ、GGUF Q4 格式），因为 INT4 只有 16 个离散值，逐通道量化精度不足。

## GPTQ（训练后量化，逐层校准）

GPTQ（Generative Pre-Trained Transformer Quantization）是 LLM 量化的重要里程碑，发表于 2022 年。

**核心思想**：最小化量化误差对模型输出的影响，而非最小化权重本身的量化误差。

**算法流程**：

1. 收集一小批校准数据（calibration dataset，通常 128 个样本）
2. 对每一层权重矩阵，逐列进行量化
3. 每量化一列后，利用 Hessian 矩阵（二阶导数信息）调整剩余列的权重，补偿量化误差
4. 重复直到所有列都被量化

GPTQ 实现了 INT4 量化后性能损失极小（在大多数 benchmark 上 <1%），且量化过程只需要在 GPU 上运行几分钟到几小时（取决于模型大小）。

```bash
# 使用 AutoGPTQ 量化
python -m auto_gptq.quantization.quant_with_calibration \
    --pretrained_model_dir meta-llama/Llama-2-7b-hf \
    --quantized_model_dir llama2-7b-gptq-int4 \
    --bits 4 \
    --group_size 128 \
    --dataset wikitext2
```

## AWQ（激活感知权重量化）

AWQ（Activation-aware Weight Quantization，MIT，2023）观察到：**并非所有权重都同等重要**。

关键发现：激活值（Activation）的分布告诉我们哪些权重通道对模型输出影响更大。那些对应较大激活值的权重通道，量化误差的影响会被"放大"。

AWQ 的方案：

1. 分析校准数据集上每个权重通道对应的平均激活值大小
2. 对"重要"通道（激活值大的通道）通过缩放（Scale）来降低其量化误差的影响
3. 通过数学等价变换，将这个缩放因子吸收到相邻层，不改变模型结构

AWQ 相比 GPTQ 的优势：
- 量化速度更快（无需逐列优化）
- 在低位（INT3、INT4）上精度通常优于 GPTQ
- 更适合硬件友好的量化内核实现

## GGUF 格式

GGUF（GGML Unified Format）是 llama.cpp 项目使用的模型格式，设计目标是跨平台、高效推理。

常见量化等级（按位数和精度从低到高）：

| 格式 | 平均位宽 | 精度损失 | 适用场景 |
|------|--------|---------|---------|
| Q2_K | ~2.6 bit | 大 | 极低内存设备 |
| Q3_K_M | ~3.35 bit | 较大 | 4GB 以下内存 |
| Q4_0 | 4 bit | 中 | 基础 4 bit 量化 |
| Q4_K_M | ~4.5 bit | 小 | 推荐的 4 bit 量化 |
| Q5_K_M | ~5.7 bit | 很小 | 精度与大小的平衡点 |
| Q6_K | ~6.6 bit | 极小 | 接近 FP16 质量 |
| Q8_0 | 8 bit | 几乎没有 | 需要高精度时 |
| F16 | 16 bit | 无 | 完整 FP16 |

**K-quants**（Q4_K_M、Q5_K_M 等）：K 代表 K-quantization，对不同层使用不同量化精度（混合精度量化），M 代表 Medium 配置（另有 S=Small, L=Large）。K-quants 在相同文件大小下通常优于同等位数的基础量化。

GGUF 文件包含：
- 模型架构元数据（超参数、分词器配置）
- 量化后的权重（Packed INT 格式）
- 量化参数（scale、zero_point，按 group 存储）

## 量化精度损失对比

以 LLaMA-2-7B 为例（MMLU 5-shot 准确率）：

| 精度 | MMLU | 文件大小 |
|------|------|---------|
| BF16 | 45.3% | 13.5 GB |
| Q8_0 | 45.2% | 7.2 GB |
| Q5_K_M | 45.0% | 4.8 GB |
| Q4_K_M | 44.7% | 4.1 GB |
| Q3_K_M | 43.8% | 3.1 GB |
| Q2_K | 40.5% | 2.4 GB |

可以看到，Q4_K_M 相比 BF16 只损失了约 0.6% 的准确率，但文件大小减少了 70%。这是 llama.cpp 用户的推荐默认选择。

## INT8 推理（bitsandbytes）

`bitsandbytes` 库（Hugging Face）提供 INT8 量化推理支持，特点是可以在加载时进行量化（Load-time Quantization）：

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,           # INT8 量化
    device_map="auto"
)
```

INT8 推理的显存占用约为 FP16 的一半，精度损失极小，推理速度在大批次时有 1.5-2x 提升（小批次可能略慢于 FP16，因为 INT8 计算后需要反量化）。

`bitsandbytes` 使用 LLM.int8() 算法：对激活值进行 per-token 量化，对权重进行 per-channel 量化，并对离群值（Outlier）保持 FP16 精度。

## INT4 推理（QLoRA）

QLoRA（Quantized Low-Rank Adaptation）将 INT4 量化与 LoRA 微调结合：

- 基础模型权重量化为 NF4（4-bit NormalFloat，专为正态分布权重设计）
- LoRA 适配器权重保持 BF16
- 双量化（Double Quantization）：对量化参数本身再次量化，进一步减少内存占用

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时反量化为 BF16
    bnb_4bit_use_double_quant=True     # 双量化
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

QLoRA 使得在单张 24GB GPU（如 RTX 3090）上微调 13B 模型成为可能。

## ExLlamaV2 速度优化

ExLlamaV2 是专为 GPTQ 量化模型设计的高速推理内核，相比基础实现速度提升 2-4 倍：

- 专为 GPTQ INT4 量化设计的 CUDA 内核
- 混合量化精度支持（EXL2 格式，支持不同层使用不同 bpw）
- 动态缓存分配，支持极长上下文
- 与 text-generation-webui 集成良好

## 选择量化级别的指南

根据不同场景选择合适的量化方案：

**按硬件内存选择（7B 模型）**：
- 8GB 显存：Q4_K_M（约 4.1 GB）
- 16GB 显存：Q8_0（7.2 GB）或 Q5_K_M（4.8 GB）
- 24GB 显存：BF16（13.5 GB）或 INT8（约 7 GB）

**按精度要求选择**：
- 生产服务，对精度敏感：INT8（bitsandbytes）或 AWQ INT4
- 本地使用，平衡精度/速度：Q4_K_M 或 Q5_K_M（GGUF）
- 极限内存节省，允许精度损失：Q3_K_M 或 Q2_K

**按使用场景选择**：
- llama.cpp / Ollama：GGUF 格式（Q4_K_M 为默认推荐）
- vLLM 生产部署：AWQ 或 GPTQ
- Python 快速实验：bitsandbytes（load_in_8bit/4bit）
- 高性能推理优化：ExLlamaV2 + EXL2
