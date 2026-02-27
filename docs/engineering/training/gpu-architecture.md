# GPU 架构与算力

理解 GPU 硬件架构是进行大模型训练和推理优化的基础。不同代际的 GPU 在算力、显存、互联带宽上有显著差异，直接影响训练效率和可部署的模型规模。

## NVIDIA GPU 架构演进

### Volta 架构（2017）：V100

V100 是深度学习训练的里程碑产品，首次引入 **Tensor Core**，专为矩阵乘法加速设计。

核心规格：
- FP16 算力：125 TFLOPS（Tensor Core）
- 显存：16GB / 32GB HBM2
- 显存带宽：900 GB/s
- NVLink 带宽：300 GB/s（双向）

V100 奠定了大规模分布式训练的基础，GPT-2、BERT 等早期大模型均在 V100 集群上训练。

### Ampere 架构（2020）：A100

A100 是目前（2024 年）仍被广泛使用的主力训练卡，相比 V100 有数倍性能提升。

核心规格：
- FP16 算力：312 TFLOPS（Tensor Core，稀疏）/ 77.6 TFLOPS（密集）
- BF16 算力：312 TFLOPS（稀疏）
- TF32 算力：156 TFLOPS（稀疏）
- 显存：80GB HBM2e
- 显存带宽：2 TB/s
- NVLink 3.0 带宽：600 GB/s（双向）
- PCIe 4.0 接口

GPT-3、LLaMA、PaLM 等主流大模型均在 A100 集群上完成训练。

### Hopper 架构（2022）：H100

H100 是 NVIDIA 面向下一代 AI 的旗舰产品，引入了 FP8 训练支持和 Transformer Engine。

核心规格：
- FP8 算力：3958 TOPS（Tensor Core，稀疏）
- FP16 算力：989 TFLOPS（稀疏）/ 494 TFLOPS（密集）
- BF16 算力：989 TFLOPS（稀疏）
- 显存：80GB HBM3（SXM5）
- 显存带宽：3.35 TB/s（SXM5）
- NVLink 4.0 带宽：900 GB/s（双向）
- 新增：NVSwitch 3.0，支持 8 卡全互联

**Transformer Engine**：H100 的核心创新，可以在 FP8 和 BF16 之间动态切换精度，在保持训练稳定性的同时，最大化利用 FP8 算力。

### 出口管制变体：H800 / H20

受美国对华出口管制（Entity List 政策）影响，NVIDIA 推出了降规格版本：

- **H800**：降低 NVLink 带宽（NVLink 3.0，400 GB/s），算力保持 H100 水平
- **H20**：进一步削减，FP16 算力约 148 TFLOPS，但显存 96GB，显存带宽 4 TB/s（HBM3），定位推理

国内云厂商（阿里云、腾讯云、百度智能云）主要部署 H800 集群用于大模型训练。

## 核心指标对比

| 型号 | FP16 算力（密集） | BF16 算力 | 显存 | 显存带宽 | NVLink 带宽 |
|------|-----------------|----------|------|---------|------------|
| V100 | 125 TFLOPS | 不支持 | 32 GB | 900 GB/s | 300 GB/s |
| A100 | 77.6 TFLOPS | 77.6 TFLOPS | 80 GB | 2 TB/s | 600 GB/s |
| H100 SXM | 494 TFLOPS | 494 TFLOPS | 80 GB | 3.35 TB/s | 900 GB/s |
| H20 | 148 TFLOPS | 148 TFLOPS | 96 GB | 4 TB/s | 400 GB/s |

注意：训练性能不仅取决于算力，**显存带宽**对 Transformer 推理性能影响更大（带宽密集型操作）。

## CUDA 编程基础

理解 CUDA 编程模型有助于理解深度学习框架的底层行为。

### 并行层次结构

CUDA 将 GPU 计算组织为三层：

- **Thread（线程）**：最小执行单元，对应一次标量计算
- **Block（块）**：一组 Thread（最多 1024 个），同一 Block 内的 Thread 可以通过 Shared Memory 共享数据，并进行 `__syncthreads()` 同步
- **Grid（网格）**：所有 Block 的集合，一个 Kernel Launch 对应一个 Grid

```cuda
// CUDA Kernel 示例：向量加法
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 启动 Kernel：1024 个 Block，每 Block 256 个 Thread
int blocks = (n + 255) / 256;
vector_add<<<blocks, 256>>>(a, b, c, n);
```

### Warp 与 SIMT

实际执行中，Block 内的 32 个 Thread 组成一个 **Warp**，以 SIMT（Single Instruction, Multiple Thread）方式执行相同指令。分支（if/else）会导致 Warp 内部分 Thread 执行不同路径，称为 Warp Divergence，严重影响性能。

### 内存层次

- **全局内存（Global Memory / HBM）**：最大但延迟最高，所有 Thread 可访问
- **共享内存（Shared Memory / SRAM）**：每个 Block 独享，速度极快（约 19 TB/s），容量小（约 48-164 KB/SM）
- **寄存器（Registers）**：每个 Thread 独享，速度最快
- **L1/L2 Cache**：自动管理，缓解全局内存访问延迟

高效的 CUDA 编程核心是**最大化共享内存利用率，最小化全局内存访问次数**。

## Tensor Core

Tensor Core 是 NVIDIA 为矩阵乘法专门设计的硬件单元，从 Volta 架构（2017）开始引入。

每个 Tensor Core 在一个时钟周期内完成 4×4 矩阵的乘加操作（D = A × B + C）。相比传统 CUDA Core 的标量乘加，吞吐量提升约 16 倍（理论峰值）。

为了充分利用 Tensor Core，矩阵维度必须是 8 或 16 的倍数（视精度而定），且数据需要对齐排布。PyTorch 的 `torch.compile()` 和 Flash Attention 等优化库都自动调用 Tensor Core。

## NVLink vs PCIe

### NVLink

NVLink 是 NVIDIA 开发的 GPU 间高速互联总线，专为深度学习优化：

- **A100**：NVLink 3.0，总带宽 600 GB/s（双向）
- **H100**：NVLink 4.0，总带宽 900 GB/s（双向）
- NVSwitch 芯片：支持 8 张 GPU 的全互联（NVLink Switch）

NVLink 的带宽是 PCIe 4.0 x16（64 GB/s 双向）的约 10-14 倍，对于 ZeRO 等需要频繁通信的分布式训练至关重要。

### PCIe

- **PCIe 4.0 x16**：约 64 GB/s 双向带宽
- **PCIe 5.0 x16**：约 128 GB/s 双向带宽
- 成本远低于 NVLink，消费级 GPU（RTX 4090）只有 PCIe 接口

## InfiniBand（多机互联）

多机训练中，节点间通信依赖高速网络：

- **InfiniBand HDR（200G）**：常用于 A100 集群，约 200 Gbps/端口
- **InfiniBand NDR（400G）**：H100 集群标准，约 400 Gbps/端口
- **RDMA（Remote Direct Memory Access）**：绕过 CPU，直接 GPU 到 GPU 传输，延迟极低

与普通以太网（10/25/100 Gbps）相比，InfiniBand 的优势在于极低延迟（约 1μs）和对 RDMA 的原生支持。

## 算力估算

估算训练一个模型所需的算力：

**经验公式**：总 FLOPs ≈ 6 × N × D

其中 N 为模型参数量，D 为训练 Token 数，系数 6 来自前向（2）+ 反向（4，约等于前向的 2 倍）。

示例：训练 7B 参数模型，使用 1 万亿 Token：
- 总 FLOPs = 6 × 7×10⁹ × 10¹² = 4.2×10²² FLOP
- 使用 1000 张 A100（77.6 TFLOPS，MFU=50%）：
  - 有效算力 = 1000 × 77.6×10¹² × 0.5 = 3.88×10¹⁶ FLOPS
  - 训练时间 = 4.2×10²² / 3.88×10¹⁶ ≈ 108 万秒 ≈ 12.5 天

## 国产替代芯片

在出口管制背景下，国产 AI 芯片加速发展：

### 华为昇腾（Ascend）

- **Ascend 910B**：性能对标 A100，支持 Atlas 800T A2 训练服务器
- **CANN 软件栈**：华为自研，部分支持 PyTorch 前端（通过 Ascend Extension for PyTorch）
- **MindSpore 框架**：华为自研深度学习框架，生态较封闭

主要挑战：CUDA 生态兼容性差，需要代码修改，部分算子性能与 NVIDIA GPU 有差距。

### 寒武纪（Cambricon）

- **MLU370**：训练/推理一体芯片
- **BANG 编程语言**：类 CUDA 的异构编程语言
- 主要客户：国内云厂商和部分行业客户

国产芯片在生态成熟度（驱动、框架支持、算子库）上与 NVIDIA 仍有较大差距，但政策支持力度持续加大，是中长期的重要方向。
