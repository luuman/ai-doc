# 混合精度训练

深度学习中的数值精度直接影响训练的速度、显存占用和稳定性。从 FP32 到 FP16 再到 BF16，再到最新的 FP8，每一次精度降低都带来了显著的效率提升，但同时也引入了新的工程挑战。混合精度训练是在精度损失最小化的前提下，最大化硬件利用率的工程实践。

## 数值格式对比

### FP32（单精度浮点）

- **符号位**：1 bit
- **指数位**：8 bit（表示范围 ±1.18×10⁻³⁸ 至 ±3.4×10³⁸）
- **尾数位**：23 bit（精度约 7 位十进制有效数字）
- **大小**：4 bytes

FP32 是深度学习的传统精度，数值范围宽广，精度高，训练稳定。缺点是显存占用大、计算速度慢。

### FP16（半精度浮点）

- **符号位**：1 bit
- **指数位**：5 bit（表示范围 ±6.1×10⁻⁵ 至 ±65504）
- **尾数位**：10 bit（精度约 3 位十进制有效数字）
- **大小**：2 bytes

FP16 将显存占用减半，并在支持 Tensor Core 的 GPU（V100 及以后）上获得数倍加速。主要风险：**数值范围窄**，梯度值过小时会发生下溢（Underflow），过大时会发生溢出（Overflow 至 inf）。

### BF16（BrainFloat 16）

- **符号位**：1 bit
- **指数位**：8 bit（与 FP32 相同，范围等同 FP32）
- **尾数位**：7 bit（精度约 2 位十进制有效数字，低于 FP16）
- **大小**：2 bytes

BF16 由 Google Brain 提出，核心优势是**保留与 FP32 相同的指数范围**，从而避免了 FP16 的溢出问题。代价是精度（尾数位）低于 FP16，但在深度学习中，精度损失通常可以接受。

BF16 支持：
- NVIDIA：A100、H100（Tensor Core 原生支持）
- Google TPU：原生支持
- AMD：MI250X 及以后
- Apple Silicon：M 系列芯片

### FP8（8位浮点）

- 有两种标准格式（IEEE 754 草案）：
  - **E4M3**：4 bit 指数 + 3 bit 尾数，范围较小，适合权重
  - **E5M2**：5 bit 指数 + 2 bit 尾数，范围较大，适合梯度

NVIDIA H100 的 Transformer Engine 支持 FP8 计算，理论算力是 BF16 的 2 倍。FP8 训练需要更精密的缩放策略（per-tensor 或 per-block 缩放）。

| 格式 | 大小 | 精度 | 数值范围 | 典型场景 |
|------|------|------|---------|---------|
| FP32 | 4 B | 高 | 宽 | 主权重、优化器状态 |
| FP16 | 2 B | 中 | 窄（易溢出） | 前向/反向计算 |
| BF16 | 2 B | 低 | 宽（与FP32同） | 前向/反向计算（推荐） |
| FP8 | 1 B | 最低 | 视格式 | H100 专用矩阵乘法 |

## 混合精度训练流程

混合精度训练的核心理念：**计算用低精度（快）、存储主权重用高精度（稳）**。

标准流程（FP16 混合精度）：

1. **前向传播**：使用 FP16 权重副本进行计算，中间激活值以 FP16 存储
2. **计算 Loss**：Loss 值通常以 FP32 计算，避免精度损失
3. **Loss Scaling**：将 Loss 乘以一个缩放因子（scale factor），防止梯度下溢
4. **反向传播**：梯度以 FP16 计算
5. **梯度 Unscaling**：将梯度除以缩放因子，还原真实梯度
6. **梯度裁剪与检查**：检查梯度中是否有 inf/NaN，有则跳过本步更新
7. **优化器更新**：使用 FP32 主权重和优化器状态更新参数
8. **更新 FP16 副本**：将更新后的 FP32 主权重复制为 FP16 供下一步使用

```
FP32 主权重 → 复制为 FP16 → 前向传播（FP16）→ Loss（FP32）
    ↑                                                   ↓
FP32 优化器状态 ← 梯度更新 ← 反向传播（FP16 + Loss Scaling）
```

## Loss Scaling（损失缩放）

FP16 的最小正规数约为 6.1×10⁻⁵。深度网络中，梯度值常常远小于此阈值，导致下溢为零，模型无法更新。

Loss Scaling 的解决方案：在计算反向传播前，将 Loss 乘以一个较大的缩放因子（如 2¹⁵ = 32768）：

```python
# 手动 Loss Scaling 示例
scale = 32768.0
loss_scaled = loss * scale
loss_scaled.backward()

# 梯度还原
for param in model.parameters():
    if param.grad is not None:
        param.grad.data /= scale
        # 检查 inf/NaN
        if torch.any(torch.isinf(param.grad.data)) or torch.any(torch.isnan(param.grad.data)):
            # 跳过本步优化器更新
            break
```

### 动态 Loss Scaling

静态缩放因子不够灵活：太小则梯度仍下溢，太大则梯度溢出为 inf。

PyTorch AMP 使用**动态 Loss Scaling**：
- 如果连续 N 步（如 2000 步）没有出现 inf/NaN，将 scale 乘以 2
- 如果出现 inf/NaN，将 scale 除以 2，并跳过本步更新
- 在训练稳定性和精度之间自适应调节

## BF16 的优势

由于 BF16 与 FP32 有相同的指数范围，训练过程中几乎不会出现溢出问题，因此：

- **无需 Loss Scaling**：BF16 训练不需要复杂的 Loss Scaling 机制
- **训练稳定性更好**：消除了 FP16 训练中偶发的 inf/NaN 问题
- **代码更简洁**：BF16 训练代码比 FP16 简单得多

现代大模型训练（LLaMA 3、Mistral、Qwen 等）几乎全部采用 BF16。BF16 的精度（7 bit 尾数）对于语言模型训练已经足够。

唯一限制：BF16 只在较新的 GPU（A100+）上受硬件支持。V100 只支持 FP16，A100 同时支持 FP16 和 BF16。

## FP8 训练（H100 Transformer Engine）

NVIDIA H100 的 Transformer Engine 是 FP8 训练的硬件基础，工作原理：

1. 对每个矩阵乘法操作，动态选择 FP8 或 BF16 精度
2. 使用 per-tensor 缩放因子（amax history）跟踪激活值的数值范围
3. 自动在 E4M3（权重、激活）和 E5M2（梯度）间切换

```python
# 使用 transformer_engine 进行 FP8 训练
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,  # 前向 E4M3，反向 E5M2
    amax_history_len=16,
    amax_compute_algo="max"
)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output = transformer_layer(input)
```

FP8 训练的挑战：
- 数值精度更低，需要仔细调整缩放策略
- 部分操作（BatchNorm、LayerNorm）仍需 BF16/FP32
- 长训练中的精度累积误差需要监控

## PyTorch AMP（Automatic Mixed Precision）

PyTorch 的 `torch.cuda.amp` 模块提供了对混合精度训练的完整支持：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # 自动选择 FP16/FP32 操作
    with autocast(dtype=torch.float16):
        output = model(batch)
        loss = criterion(output, target)

    # 缩放 Loss 并反向传播
    scaler.scale(loss).backward()

    # Unscale 梯度，检查 inf/NaN，更新参数
    scaler.step(optimizer)
    scaler.update()
```

对于 BF16，只需将 `dtype` 改为 `torch.bfloat16`，并且不需要 GradScaler：

```python
with autocast(dtype=torch.bfloat16):
    output = model(batch)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
```

## 对训练速度与显存的影响

以 7B 参数模型为例，各精度方案的显存对比：

| 训练方案 | 参数显存 | 梯度显存 | 优化器状态 | 总显存 |
|---------|---------|---------|----------|-------|
| 纯 FP32 | 28 GB | 28 GB | 56 GB | ~112 GB |
| FP16 混合精度 | 14 GB (FP16) + 28 GB (FP32 主权重) | 14 GB | 56 GB | ~112 GB |
| BF16 混合精度 | 14 GB (BF16) + 28 GB (FP32 主权重) | 14 GB | 56 GB | ~112 GB |
| BF16 + ZeRO-3（64卡） | 参数切分 | 梯度切分 | 状态切分 | ~2 GB/卡 |

速度提升（相对 FP32）：
- FP16/BF16：Tensor Core 矩阵乘法加速约 2-3 倍
- FP8：相对 BF16 再提升约 1.5-2 倍（实际受限于内存带宽）

## 精度损失评估

混合精度训练的精度损失通常可以忽略不计，但以下场景需要注意：

- **极小学习率场景**：梯度值极小时，BF16 精度不足以区分细微梯度差异
- **长训练过程**：精度误差可能随训练步数累积
- **某些归一化层**：LayerNorm 对精度敏感，通常保持 FP32

评估方法：
- 在固定随机种子下，对比 FP32 和混合精度训练的 Loss 曲线
- 在相同 Benchmark 上评估最终模型性能（如 HellaSwag、MMLU）
- 监控梯度范数（Gradient Norm）的变化趋势

实践结论：对于绝大多数语言模型任务，BF16 混合精度与 FP32 训练的最终性能差异在统计误差范围内（<0.5%），而带来的训练速度提升（2-3 倍）和显存节省（约 30%）是显而易见的工程收益。
