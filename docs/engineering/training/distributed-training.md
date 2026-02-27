# 分布式训练

单张 GPU 的显存容量（A100 为 80GB，H100 为 80GB SXM / 141GB HBM3e）远不足以容纳现代大语言模型的参数。GPT-3 的 1750 亿参数仅参数本身就需要约 350GB（FP16），加上优化器状态和梯度则超过 1TB。分布式训练是大模型训练的必要手段，而非可选优化。

## 数据并行（Data Parallelism）

数据并行是最简单的分布式训练方式：每张 GPU 持有**完整的模型副本**，但处理不同的数据批次。

### DDP（DistributedDataParallel）

PyTorch 的 `DistributedDataParallel` 是数据并行的标准实现：

1. 每个 GPU 进程独立进行前向和反向传播，计算本地梯度
2. 反向传播完成后，通过 **AllReduce** 操作在所有 GPU 间同步梯度（取平均值）
3. 每个 GPU 用相同的平均梯度更新本地模型参数

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
model = DDP(model, device_ids=[local_rank])
```

DDP 的通信开销随 GPU 数量增加而增大，适合参数量在单卡显存范围内的模型（最多数十亿参数）。

### 梯度同步

AllReduce 的通信量与模型参数量成正比。对于 7B 参数的 FP16 模型，每步同步需要传输约 14GB 数据。优化手段：

- **梯度压缩**：PowerSGD、1-bit Adam 等算法减少通信量
- **梯度累积**：增大有效 batch size，减少同步频率
- **重叠通信与计算**：在反向传播计算梯度时，同步已完成的层的梯度

## 模型并行（Model Parallelism）

当模型无法放入单张 GPU 时，需要将模型切分到多张 GPU 上。

### 层间并行（Pipeline Parallelism 的特例）

最朴素的模型并行：将模型的不同层分配到不同 GPU：

- GPU 0：第 1-8 层
- GPU 1：第 9-16 层
- GPU 2：第 17-24 层
- GPU 3：第 25-32 层

问题：任何时刻只有一张 GPU 在工作，其他 GPU 等待，GPU 利用率极低（"气泡"问题）。

### 张量并行（Tensor Parallelism）

将单个矩阵乘法操作切分到多张 GPU 并行计算。对于 Transformer 中的多头注意力，可以将不同的注意力头分配到不同 GPU；对于 FFN 层，可以按列或行切分权重矩阵。

张量并行需要在每层结束后进行 AllReduce 通信，通信频繁但通信量可控。Megatron-LM 是张量并行的代表性实现。

## 流水线并行（Pipeline Parallelism）

流水线并行解决了朴素层间并行的 GPU 利用率问题，核心思想是**Micro-batch 流水线**：

将一个 Mini-batch 切分为多个 Micro-batch，像工厂流水线一样处理：

```
时间:  t1    t2    t3    t4    t5    t6    t7
GPU0:  F(m1) F(m2) F(m3) F(m4) B(m4) B(m3) B(m2) B(m1)
GPU1:        F(m1) F(m2) F(m3) F(m4) B(m4) B(m3) B(m2) B(m1)
GPU2:              F(m1) F(m2) F(m3) F(m4) B(m4) B(m3) B(m2) B(m1)
```

（F=前向，B=反向，m1-m4 为 Micro-batch）

流水线并行仍存在"流水线气泡"（pipeline bubble），即流水线填充和排空阶段的 GPU 空闲。GPipe 和 PipeDream 等算法进一步优化了气泡比例。

## ZeRO 优化（Zero Redundancy Optimizer）

ZeRO 是 DeepSpeed 提出的显存优化方案，通过切分**冗余状态**来降低每张 GPU 的显存需求。

训练时的显存组成（以 FP16 混合精度为例，7B 参数模型）：

| 状态类型 | 大小 |
|---------|------|
| FP16 参数 | 14 GB |
| FP16 梯度 | 14 GB |
| FP32 主权重 | 28 GB |
| Adam 优化器状态（FP32 m, v） | 56 GB |
| **总计** | **~112 GB** |

### ZeRO-1：切分优化器状态

每张 GPU 只保存 1/N 的优化器状态。更新时，每张 GPU 更新自己负责的参数，然后通过 AllGather 同步完整参数。显存节省约 4x（针对优化器状态）。

### ZeRO-2：切分梯度 + 优化器状态

在 ZeRO-1 基础上，进一步切分梯度。梯度计算完成后，每张 GPU 只保留自己需要更新的参数对应的梯度。显存节省约 8x。

### ZeRO-3：切分参数 + 梯度 + 优化器状态

将模型参数本身也切分到所有 GPU，显存节省与 GPU 数量成正比。代价是通信量增加——每次前向/反向传播都需要通过 AllGather 获取完整层的参数。

## Megatron-LM（NVIDIA 3D 并行）

Megatron-LM 是 NVIDIA 开发的大规模语言模型训练框架，将数据并行、张量并行和流水线并行组合为 **3D 并行**：

- **张量并行维度**：相邻 GPU（同一节点内，NVLink 互联，通信快）
- **流水线并行维度**：跨节点，按层分组
- **数据并行维度**：最外层，多个 3D 并行组

这种组合充分利用了节点内高速 NVLink 和节点间较慢的 InfiniBand，是目前训练千亿参数模型的主流方案。

## DeepSpeed 框架

DeepSpeed 是 Microsoft 开发的分布式训练框架，核心特性：

- **ZeRO-1/2/3**：上文描述的显存优化
- **ZeRO-Infinity**：利用 CPU 内存和 NVMe SSD 扩展可训练模型规模
- **ZeRO-Offload**：将优化器状态卸载到 CPU，减少 GPU 显存压力
- **Activation Checkpointing**：（即梯度检查点）
- **Sparse Attention**：稀疏注意力计算，减少长序列的计算量
- **1-bit Adam / 1-bit LAMB**：通信压缩算法

```python
# DeepSpeed 配置示例
ds_config = {
    "train_batch_size": 512,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    },
    "fp16": {"enabled": True}
}
```

## 梯度检查点（Gradient Checkpointing）

梯度检查点（也称 Activation Checkpointing）是一种**以计算换显存**的技术：

- **正常训练**：前向传播时保存所有中间激活值，用于反向传播计算梯度，显存占用大
- **梯度检查点**：只保存部分"检查点"层的激活值，反向传播时重新计算中间激活

代价是增加约 30-40% 的计算量，但显存占用降低约 10 倍（取决于序列长度）。对于长序列训练（16K、32K Token），梯度检查点几乎是必须的。

PyTorch 实现：

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.layer1, x)  # 只保存输入，反向时重算
    x = checkpoint(self.layer2, x)
    return x
```

## 通信原语

分布式训练依赖以下集合通信操作（Collective Communication）：

- **AllReduce**：所有 GPU 的张量求和/平均，结果广播给所有 GPU（用于梯度同步）
- **AllGather**：每张 GPU 贡献一块数据，所有 GPU 收集完整数据（ZeRO-3 参数收集）
- **ReduceScatter**：AllReduce 的分解操作，每张 GPU 只保留聚合结果的一部分（ZeRO 梯度切分）
- **Broadcast**：一张 GPU 将数据发送给所有其他 GPU（参数初始化）

NCCL（NVIDIA Collective Communications Library）是 GPU 集合通信的标准实现，针对 NVLink 和 InfiniBand 做了深度优化。

## 多机多卡训练配置

典型的多机训练配置示例（8 台 8xA100 服务器，共 64 张 GPU）：

```bash
# 在主节点（rank 0）启动
torchrun \
    --nproc_per_node=8 \
    --nnodes=8 \
    --node_rank=0 \
    --master_addr="10.0.0.1" \
    --master_port=29500 \
    train.py

# 在其他节点启动（修改 node_rank）
torchrun \
    --nproc_per_node=8 \
    --nnodes=8 \
    --node_rank=1 \
    --master_addr="10.0.0.1" \
    --master_port=29500 \
    train.py
```

网络配置注意事项：

- 多机通信依赖 InfiniBand（带宽 200-400 Gbps）或高速以太网
- 需要配置低延迟网络（RDMA），普通以太网会成为严重瓶颈
- 防火墙需要开放 NCCL 通信端口
- 建议使用共享存储（NFS 或 GPFS）存放训练数据和检查点

容错机制：

- 定期保存训练检查点（每 N 步保存一次）
- GPU 故障检测与自动重启
- 梯度 NaN/Inf 检测与跳过
