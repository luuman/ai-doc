# RLHF 与偏好优化

RLHF（Reinforcement Learning from Human Feedback，从人类反馈中强化学习）是使大语言模型与人类偏好对齐的核心技术。从 InstructGPT 到 Claude，RLHF 是将"有能力的语言模型"变为"有用且安全的助手"的关键桥梁。

## RLHF 三阶段流程

### 阶段一：监督微调（SFT）

使用人工标注的高质量指令-回答对，对预训练模型进行监督微调，建立基础的指令遵循能力。

这是 RLHF 的起点，SFT 模型（也称为 Policy 的初始化点）需要已经具备基本的能力，才能使后续强化学习有效。

### 阶段二：奖励模型训练（Reward Model Training）

**目标**：训练一个模型，使其能够预测人类对 LLM 输出的偏好评分。

**数据收集流程**：

1. 从 SFT 模型生成多条候选回答（通常 4-9 条），每条使用不同采样参数
2. 人类标注员对这些候选回答进行**成对比较**或**排序**（而非绝对评分）
   - 成对比较示例："回答 A 是否比回答 B 更有帮助？"
   - 排序示例："从最好到最差对以下 4 条回答排序"
3. 收集大规模偏好数据（InstructGPT 使用约 33K 比较对）

**奖励模型架构**：

- 基于 LLM（通常是 SFT 模型），在最后加一个线性层输出标量分数
- 输入：`[Prompt, Response]`，输出：标量奖励值（越高越好）
- 使用 Bradley-Terry 模型（对数似然偏好损失）训练：

```
L = -log(σ(r(x, y_w) - r(x, y_l)))
```

其中 `y_w` 是偏好（Chosen）回答，`y_l` 是非偏好（Rejected）回答，`r(x, y)` 是奖励分数。

### 阶段三：PPO 强化学习（Policy Optimization）

使用强化学习算法（PPO，Proximal Policy Optimization）优化 SFT 模型，使其生成能获得更高奖励模型评分的回答。

**PPO 在 LLM 中的目标函数**：

```
L = E[r_θ(x, y) - β × KL(π_θ(y|x) || π_ref(y|x))]
```

- `r_θ(x, y)`：奖励模型给出的分数
- `KL(...)`：当前策略与参考策略（SFT 模型）之间的 KL 散度惩罚
- `β`：KL 惩罚系数，防止模型过度偏离 SFT 基线（防止"奖励黑客"行为）

**奖励黑客（Reward Hacking）**：

- 模型学会生成高奖励分数但实际无用的回答（如"当然！" + 重复用户问题）
- KL 惩罚限制了模型偏离 SFT 初始点的程度，缓解但不能完全解决此问题

## RLHF 的不稳定性问题

RLHF 的 PPO 阶段工程挑战极大：

- **需要同时维护多个模型**：Policy（被优化的模型）、Reward Model、Reference Model（SFT 模型，用于 KL 计算）、Value Model（PPO 的 Critic），显存需求约为 SFT 的 4 倍
- **PPO 对超参极其敏感**：KL 系数、学习率、Clip 参数稍有不当即导致训练崩溃
- **奖励模型漂移**：长时间训练后奖励模型不再准确反映人类偏好（Distribution Shift）
- **训练过程不稳定**：损失曲线不规则，需要大量调试经验

这些问题促使研究者寻找更简单的替代方案。

## DPO（Direct Preference Optimization）：简化方案

**论文**：*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*（Rafailov 等，2023）

DPO 的核心洞察：在 RLHF 的目标函数中，奖励模型可以用语言模型概率直接表示，从而完全绕过显式训练奖励模型和 PPO 强化学习。

**DPO 损失函数**：

```
L_DPO = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

**DPO 的优势**：

- 只需要偏好数据（Chosen/Rejected 对），不需要训练奖励模型
- 训练方式与 SFT 相同（监督学习），没有 PPO 的不稳定性
- 只需维护两个模型（Policy + Reference），显存需求与 SFT 相当
- 工程复杂度大幅降低

**DPO 的局限**：

- 对偏好数据质量要求更高（需要 Chosen 明显优于 Rejected）
- 在某些任务上效果略弱于精调好的 PPO
- 无法动态生成新的候选回答（属于离线偏好学习）

2023-2024 年，DPO 已成为开源 LLM 对齐训练的主流方法，LLaMA 3、Mistral、Qwen 等均在 SFT 后应用 DPO。

## DPO 变体

### SimPO（Simple Preference Optimization）

- 去除 Reference Model，用平均 Token 对数概率代替
- 进一步降低显存需求
- 增加 Length Normalization 防止模型倾向生成长回答

### IPO（Identity Preference Optimization）

- 修正 DPO 中潜在的过拟合问题（当 Chosen 与 Rejected 差异过小时）
- 使用不同的损失形式，理论上更稳定

### KTO（Kahneman-Tversky Optimization）

- 不需要成对的 Chosen/Rejected 数据，只需单条回答的好/坏标签
- 基于行为经济学中的前景理论

## 偏好数据构造

高质量偏好数据是 RLHF/DPO 的核心：

### 格式要求

每条偏好数据包含：

```json
{
  "prompt": "用户问题或系统提示",
  "chosen": "人类偏好的高质量回答",
  "rejected": "相对较差的回答"
}
```

### 数据来源

- **人工标注对比**：标注员对模型输出进行两两比较（质量最高，成本最高）
- **AI 辅助标注（RLAIF）**：用 GPT-4 等强模型评估偏好，降低成本
- **代码/数学的规则判断**：正确解法 vs 错误解法，无需人工判断
- **公开数据集**：
  - **HH-RLHF**（Anthropic）：169K 条帮助性和无害性偏好数据
  - **OpenAssistant**：多语言人工标注对话
  - **UltraFeedback**：GPT-4 自动评分的 64K 指令数据集

### 偏好差距的重要性

Chosen 与 Rejected 的质量差距越大，偏好学习效果越好。常见实践：

- 用最好的模型生成 Chosen，用较差的模型或简单贪心解码生成 Rejected
- 针对特定能力（如代码正确性、事实准确性）构造有明确对错的偏好对

## RLHF 在提升安全性与有用性的双重作用

RLHF 的标注维度通常包含两个方向：

**有用性（Helpfulness）**：

- 回答准确解决了用户的问题
- 回答信息量足够且组织清晰
- 回答风格与请求相符（技术/通俗/创意等）

**无害性（Harmlessness）**：

- 拒绝生成有害内容（暴力、色情、违法信息等）
- 在敏感话题上保持中立或适当回避
- 不对用户进行欺骗或操控

RLHF/DPO 使模型在这两个维度上的表现同时提升，但也存在内在张力：
- 过度优化安全性会导致过度拒绝（模型动辄说"我无法帮助"）
- 过度优化有用性可能降低安全护栏

Claude 3 系列相比 Claude 2 的重要改进之一，就是在保持安全性的前提下大幅减少了不必要的拒绝。
