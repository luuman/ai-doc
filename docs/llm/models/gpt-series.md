# GPT 系列模型

GPT（Generative Pre-trained Transformer）系列是 OpenAI 开发的大语言模型家族，从 2018 年的 GPT-1 到 2024 年的 o3，每一代都推动了 AI 能力的边界。本文梳理 GPT 系列的技术演进脉络与商业化战略。

## GPT-1（2018）：生成式预训练的起点

**论文**：*Improving Language Understanding by Generative Pre-Training*

GPT-1 是 OpenAI 首次将"预训练+微调"范式系统化应用于 NLP 的尝试：

- **参数量**：1.17 亿（117M）
- **架构**：12 层 Transformer Decoder，768 维隐藏层
- **训练数据**：BooksCorpus（约 5GB，8亿词）
- **核心贡献**：证明了在大量无标注文本上预训练，然后在下游任务微调，能显著优于从头训练的监督模型

GPT-1 的局限在于，每个下游任务仍需任务专用微调，尚未展现出零样本能力。

## GPT-2（2019）：零样本能力的展现

**论文**：*Language Models are Unsupervised Multitask Learners*

- **参数量**：15 亿（1.5B），相比 GPT-1 增加约 12 倍
- **训练数据**：WebText（约 40GB，来自 Reddit 高分外链页面）
- **词汇表**：50,257 个 Token（Byte-Level BPE）

GPT-2 的关键发现：在足够大的语言模型上，零样本（Zero-Shot）能力自然涌现：

- 无需微调即可完成摘要、翻译、问答
- 生成的长文本在主题一致性和流畅度上超越之前的模型

GPT-2 因此被 OpenAI 以"模型过于危险"为由，分阶段发布（先发布小版本，再发布完整版）——这也是 AI 安全叙事在产业界首次引发广泛讨论。

## GPT-3（2020）：少样本学习的突破

**论文**：*Language Models are Few-Shot Learners*

- **参数量**：1750 亿（175B），此前最大模型的约 100 倍
- **训练数据**：约 570GB 过滤后的 Common Crawl + Books + Wikipedia + WebText，共约 300B Token
- **架构**：96 层，96 个注意力头，12288 维隐藏层

GPT-3 是 LLM 发展史上的里程碑：

- **少样本（Few-Shot）学习**：在 Prompt 中给出 1-20 个示例，无需梯度更新，即可适配大量任务
- **上下文学习（In-Context Learning）**：模型从上下文中"学习"任务形式，而非通过训练
- **涌现能力首次大规模展现**：算术、代码、逻辑推理等能力从质变到量变

GPT-3 的局限：生成内容有时不遵循指令、会产生有害内容、存在虚假信息（Hallucination）。

## InstructGPT（2022）：RLHF 对齐

**论文**：*Training Language Models to Follow Instructions with Human Feedback*

InstructGPT 是将 GPT-3 从"语言模型"变为"有用助手"的关键一步：

- **方法**：在 GPT-3 上应用 RLHF（从人类反馈中强化学习）三阶段流程：监督微调（SFT）→ 奖励模型训练 → PPO 强化学习
- **数据**：OpenAI 雇佣专业标注员编写高质量指令-回答对，标注偏好
- **效果**：13 亿参数的 InstructGPT 在人类偏好评测中优于未对齐的 1750 亿 GPT-3

InstructGPT 直接催生了 ChatGPT，并确立了 RLHF 作为 LLM 对齐标准流程的地位。

## ChatGPT 与 GPT-3.5（2022-2023）

ChatGPT 于 2022 年 11 月发布，本质上是基于 GPT-3.5（GPT-3 的改进版）应用 RLHF 的对话助手：

- 5 天内突破 100 万用户，2 个月内突破 1 亿用户，成为历史上增长最快的消费级产品
- 确立了"对话式 AI 助手"作为 LLM 主流交互范式
- 推动 Google、Meta、Anthropic 等公司加速产品发布

## GPT-4（2023）：多模态与考试成绩

**技术报告**：*GPT-4 Technical Report*（未公开架构细节）

- **参数量**：未公开（据传约 1.76T，使用 MoE 架构）
- **训练数据**：未公开，截止日期约 2021 年
- **多模态**：支持图像输入（Vision 能力）

GPT-4 的标志性能力展示：

- 律师资格考试（Bar Exam）：约 90 百分位
- SAT 数学：接近满分
- GRE 各科目均超过 80 百分位
- 在多个医学、法律、工程专业考试中超过人类平均水平

技术改进：

- 更长的上下文窗口（最初 8K，后扩展至 128K）
- 更强的指令遵循和多步推理
- 减少幻觉和有害输出

## GPT-4o（2024）：实时多模态

GPT-4o（"o" 代表 omni，全能）是 OpenAI 首个原生多模态模型：

- **原生多模态**：文本、图像、音频在统一模型中处理，而非多模型拼接
- **实时语音对话**：低延迟音频对话，支持情感感知、打断对话
- **视觉理解**：识别图表、截图、实物照片并结合推理
- **成本**：相比 GPT-4 Turbo 降价 50%，速度提升约 2 倍
- **免费访问**：向 ChatGPT 免费用户开放（有限额）

## o1 / o3：推理模型的新范式（2024-2025）

OpenAI 的 o 系列代表了 LLM 的新方向——**推理时计算扩展（Test-Time Compute Scaling）**：

- **内化 CoT（Chain-of-Thought）**：模型在输出答案前，在内部执行长时间的"思维链"推理（用户不可见的推理过程）
- **计算换精度**：允许模型"想更长时间"，在数学、代码、科学推理上显著超越 GPT-4o
- **o1（2024 年 9 月）**：
  - 在竞赛数学（AIME 2024）中达到约 83% 正确率（GPT-4o 约 13%）
  - 在 Codeforces 上达到 89 百分位
- **o1 pro / o3（2024 年 12 月）**：
  - o3 在 ARC-AGI 基准上达到约 87.5%（此前最高约 53%）
  - 高计算版本在 AIME 2024 上接近满分
  - 被部分研究者认为是 AGI 能力的重要里程碑

## 参数规模演进

| 模型 | 年份 | 参数量 | 训练 Token |
|------|------|--------|-----------|
| GPT-1 | 2018 | 117M | ~1B |
| GPT-2 | 2019 | 1.5B | ~10B |
| GPT-3 | 2020 | 175B | 300B |
| GPT-3.5 | 2022 | ~175B | 未公开 |
| GPT-4 | 2023 | 未公开（~1.76T，MoE） | 未公开 |
| GPT-4o | 2024 | 未公开 | 未公开 |
| o1/o3 | 2024 | 未公开 | 未公开 |

## OpenAI 商业化战略

GPT 系列的商业化路径体现了几个核心策略：

- **API 优先**：通过 OpenAI API 向开发者提供模型访问，建立生态护城河
- **产品化**：ChatGPT Plus 订阅（20美元/月）、ChatGPT Enterprise 企业版，差异化定价
- **Microsoft 合作**：微软 100 亿美元投资，Azure OpenAI Service 独家云部署，Office 365 Copilot 集成
- **能力分层**：免费版（GPT-3.5/4o mini）→ Plus（GPT-4o）→ Pro（o1 pro/o3）多层次变现
- **API 降价**：随能效提升持续降价（GPT-3.5 API 从 2 美元/M tokens 降至 0.5 美元），扩大市场覆盖
- **生态系统**：GPT Store（自定义 GPT 市场）、Function Calling（工具调用）、Assistants API 构建完整开发者生态
