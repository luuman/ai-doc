# LoRA 与参数高效微调

参数高效微调（Parameter-Efficient Fine-Tuning，PEFT）是在资源受限条件下定制 LLM 的核心技术体系。LoRA（Low-Rank Adaptation）是其中最广泛应用的方法，使得在消费级 GPU 上微调数十亿参数模型成为可能。

## 全量微调的资源消耗问题

在讨论 LoRA 之前，先理解全量微调（Full Fine-Tuning）的资源挑战：

### 显存构成分析（以 7B 模型为例）

全量微调需要在 GPU 显存中同时保存：

- **模型权重**：7B × 2 bytes（BF16）= 14 GB
- **梯度**：与权重相同大小 = 14 GB
- **优化器状态**（AdamW）：一阶矩 + 二阶矩 = 14 GB × 2 = 28 GB（FP32）
- **激活值（Activation）**：与 Batch Size 和序列长度相关，约 10-20 GB

**总计：约 60-80 GB**，超出单块 A100 80GB 显存的极限，必须使用多卡并行。

### 训练时间与成本

- 7B 模型全量微调 1 万条数据（3 epochs）：4 × A100，约 2-4 小时
- 70B 模型全量微调：需要 32+ 块 A100，成本高达数千至数万美元

这对大多数个人研究者和中小企业来说难以承受。

## LoRA 原理：低秩矩阵分解

**论文**：*LoRA: Low-Rank Adaptation of Large Language Models*（Hu 等，2021）

### 核心思想

大模型在微调过程中，权重矩阵的变化量（ΔW）具有**低内在秩（Intrinsic Rank）**特性——尽管权重矩阵本身是高维的（如 4096×4096），但有意义的权重更新实际上处于一个低维子空间中。

因此，可以用两个低秩矩阵的乘积来近似 ΔW：

```
W' = W + ΔW = W + B × A
```

其中：

- **W**：原始预训练权重（冻结，不更新）
- **A**：形状为 `[r, d_in]` 的低秩矩阵，随机高斯初始化
- **B**：形状为 `[d_out, r]` 的低秩矩阵，初始化为零（确保初始 ΔW=0）
- **r**：秩（rank），远小于 d_in 和 d_out（如 r=8 vs d=4096）

### 参数量节省

以 4096×4096 的注意力投影矩阵为例：

- 全量微调需要更新：4096 × 4096 = 16,777,216 个参数
- LoRA（r=8）只需要更新：(4096×8) + (8×4096) = 65,536 个参数
- **参数节省：99.6%**

### LoRA 推理时的合并

训练完成后，可以将 LoRA 权重合并回原始权重，推理时无额外开销：

```
W_merged = W + B × A × (α / r)
```

其中 `α` 是 LoRA 缩放因子（见下节），合并后的模型与原始模型架构完全相同。

## rank 与 alpha 超参

### rank（r）

- **含义**：低秩矩阵的秩，决定 LoRA 的参数量和表达能力
- **典型取值**：4、8、16、32、64
- **选择原则**：
  - r 越大，表达能力越强，参数越多，但过大时趋近于全量微调
  - 对于简单任务或风格迁移，r=4 或 r=8 通常足够
  - 对于复杂任务（如数学推理微调），r=16 或 r=32 更合适

### alpha（α）

- **含义**：LoRA 输出的缩放因子，实际缩放比例为 `α/r`
- **典型取值**：通常等于 rank 或其 2 倍（如 r=8，α=16）
- **作用**：平衡 LoRA 更新相对于原始权重的影响力

实践中常见设置：

```python
r = 16
lora_alpha = 32  # alpha/r = 2，表示 LoRA 更新被放大 2 倍
lora_dropout = 0.05
```

## 适用的模块

LoRA 可应用于 Transformer 中的线性层。选择哪些层应用 LoRA 影响效果与参数量：

### 注意力层（通常必选）

- Q（Query）投影
- K（Key）投影
- V（Value）投影
- O（Output）投影

### 前馈网络（FFN）

- Gate 投影（SwiGLU 中）
- Up 投影
- Down 投影

### 实验发现

- **只对 Q/V 应用 LoRA**：最省参数，适合存储资源极度受限的场景
- **对所有注意力层和 FFN 应用 LoRA**：效果通常最好
- **对所有线性层应用 LoRA**（包括 Embedding 层）：参数最多，效果进一步提升

## QLoRA：将 7B 微调压缩到 6-8 GB 显存

**论文**：*QLoRA: Efficient Finetuning of Quantized LLMs*（Dettmers 等，2023）

QLoRA 是 LoRA 的进一步优化，通过三项关键技术将显存需求压缩至极致：

### 1. NF4 量化（4-bit Normal Float）

- 将模型权重从 BF16（2 bytes）量化至 NF4（0.5 bytes/weight）
- NF4 是一种专为正态分布权重设计的非均匀量化格式，在 4-bit 精度下损失最小
- 7B 模型权重：从 14 GB 压缩至约 3.5 GB

### 2. 双重量化（Double Quantization）

- 对量化常数（Scale 因子）再次量化，进一步减少内存开销
- 每参数额外节省约 0.37 bit

### 3. 分页优化器（Paged Optimizer）

- 利用 NVIDIA 统一内存机制，将优化器状态在 GPU 和 CPU 内存之间动态换页
- 防止 GPU OOM（内存溢出）崩溃，实现更稳定的长序列训练

### QLoRA 实际显存需求

| 模型 | QLoRA 显存 | LoRA（BF16） 显存 | 全量微调显存 |
|------|-----------|----------------|------------|
| 7B | ~6-8 GB | ~16-20 GB | ~60-80 GB |
| 13B | ~10-12 GB | ~30-40 GB | ~120+ GB |
| 70B | ~48 GB | >200 GB | 需要集群 |

QLoRA 使得在单块 RTX 4090（24GB）上微调 7B 模型成为可能，在 A100 80GB 上可微调 70B 模型。

## PEFT 库使用（Hugging Face）

Hugging Face 的 `peft` 库封装了主流 PEFT 方法：

```python
from peft import LoraConfig, get_peft_model, TaskType

# 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 将 LoRA 应用于模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 6,738,415,616 || trainable%: 0.62%
```

## LoRA 合并（merge_adapter）

训练完成后，可以将 LoRA 权重合并至基础模型权重，得到一个可独立使用的完整模型：

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model/")
```

合并后：

- 推理时无额外的矩阵乘法开销（与原始模型完全相同）
- 可以作为独立检查点分享或部署

## 多 LoRA 切换

在服务器部署场景中，可以基于同一基础模型加载多个 LoRA Adapter，按需切换：

- 基础模型权重在 GPU 显存中常驻（最大开销）
- 不同 LoRA Adapter 按请求动态加载（开销极小）
- 实现"一个基础模型 + 多个任务专用 LoRA"的高效服务架构

`peft` 库的 `set_adapter()` 方法支持运行时切换 LoRA。

## 与全量微调的效果对比

LoRA 的效果与全量微调的差距在不断缩小：

- **简单任务**（风格迁移、格式调整）：LoRA 效果与全量微调几乎无差异
- **中等难度任务**（特定领域知识注入、指令跟随改善）：LoRA（r=64）达到全量微调约 95% 的效果
- **高难度任务**（大幅改变推理方式、深度领域专家化）：全量微调仍有明显优势
- **QLoRA 的损失**：量化引入额外 1-3% 的性能损失（可通过提高 rank 部分弥补）

对于大多数实际应用场景（企业私有数据微调、特定风格定制），LoRA/QLoRA 已经足够，是首选方案。
