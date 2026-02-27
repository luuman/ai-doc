# LLaMA 系列与 Meta AI

LLaMA（Large Language Model Meta AI）是 Meta 发布的开源大语言模型系列，以开放权重（Open Weights）策略重塑了 LLM 生态，催生了数以百计的开源派生模型，成为开源 AI 运动最重要的里程碑之一。

## Meta 的开源战略逻辑

Meta（前 Facebook）选择开源大模型，与 OpenAI 的闭源路线形成鲜明对比。其战略逻辑包括：

- **平台竞争**：Meta 不以 AI API 为主要商业模式，开源 LLaMA 可打压竞争对手的 API 定价，降低整个行业的使用成本，从而将竞争转移到 Meta 更擅长的社交/广告生态
- **研究生态**：开放模型权重吸引学术界和开发者社区，产生大量高质量研究成果，反哺 Meta 的 AI 研究
- **人才吸引**：开源贡献提升 Meta AI Research 在技术社区的声誉
- **商业实用性**：降低内部团队使用 AI 的成本和复杂度
- **Yann LeCun 的技术信念**：Meta 首席 AI 科学家 Yann LeCun 长期倡导开放科学，认为开源是 AI 良性发展的关键

## LLaMA 1（2023 年 2 月）：学术开放

**论文**：*LLaMA: Open and Efficient Foundation Language Models*

- **模型规模**：7B、13B、33B、65B 参数
- **训练数据**：约 1.4T Token（Common Crawl、GitHub、Wikipedia、Books、ArXiv、StackExchange）
- **授权**：仅限非商业研究用途

LLaMA 1 的关键技术选择：

- 采用**Pre-normalization（RMSNorm）**而非 Post-normalization
- 使用 **SwiGLU** 激活函数（优于 ReLU 和 GeLU）
- 使用 **RoPE**（旋转位置编码）代替绝对位置编码
- 这些选择后来被几乎所有主流开源模型沿用

LLaMA 1 在发布后不久，模型权重通过 BitTorrent 泄露，实际上流向了整个社区。这次"意外"泄露大大加速了开源生态的发展。

## LLaMA 2（2023 年 7 月）：商业开放

**论文**：*Llama 2: Open Foundation and Fine-Tuned Chat Models*

- **模型规模**：7B、13B、34B（部分发布）、70B
- **训练数据**：约 2T Token，数据质量大幅提升
- **授权**：商业可用（月活跃用户超 7 亿的产品需额外申请）
- **Chat 版本**：同时发布 Llama-2-Chat，经过 RLHF 对齐，可直接用于对话

LLaMA 2 的改进：

- 上下文窗口从 2K 扩展至 **4K Token**
- 70B 模型性能接近但略低于 GPT-3.5
- Ghost Attention（GAttI）：在 RLHF 训练中保持系统提示的连贯性

## LLaMA 3（2024 年 4 月）：大幅跃升

**论文**：*The Llama 3 Herd of Models*

- **模型规模**：8B、70B、405B
- **训练数据**：超过 15T Token（约为 LLaMA 2 的 7 倍）
- **词汇表**：从 32K 扩展至 **128K Token**（大幅提升多语言覆盖率）
- **上下文**：8K Token（后续版本扩展至 128K）

关键技术升级：

- **GQA（分组查询注意力）**：应用于 8B 和 70B，减少 KV Cache，提升推理效率
- **超过 Chinchilla 最优的数据量**：用更多 Token 训练更小模型，优化推理效率
- **更高质量的指令微调数据**：超过 1000 万条精心筛选的 SFT 数据

性能：

- LLaMA 3 8B 超越 LLaMA 2 70B 的多项能力
- LLaMA 3 70B 在代码（HumanEval）、推理（MATH）、对话上接近 GPT-4

## LLaMA 3.1/3.2/3.3：快速迭代

**LLaMA 3.1（2024 年 7 月）**：

- 405B 参数旗舰模型发布，与 GPT-4o 和 Claude 3.5 Sonnet 正面竞争
- 上下文扩展至 **128K Token**
- 8B 和 70B 同步升级至 128K 上下文
- 强化工具调用（Function Calling）能力

**LLaMA 3.2（2024 年 9 月）**：

- 新增**多模态**变体：11B 和 90B 视觉语言模型（Vision LM）
- 新增**轻量级**设备端模型：1B 和 3B（面向手机端本地推理）
- 1B/3B 模型针对边缘设备（手机/笔记本）进行量化优化

**LLaMA 3.3（2024 年 12 月）**：

- 70B 指令调优版本（工具调用、代码能力进一步提升）
- 接近 LLaMA 3.1 405B 的能力，仅需 70B 的计算成本

## 开源生态影响

LLaMA 发布后，开源社区围绕其权重构建了极其丰富的生态：

### 早期派生模型（基于 LLaMA 1/2）

- **Alpaca**（Stanford）：5 万条 Self-Instruct 数据微调，成本约 600 美元，展示了廉价指令微调的可行性
- **Vicuna**（Berkeley/CMU）：用 ShareGPT 数据（真实 ChatGPT 对话）微调，接近 ChatGPT 90% 能力
- **WizardLM**（Microsoft）：Evol-Instruct 方法自动生成复杂指令数据
- **Code Llama**（Meta 自研）：专门针对代码场景微调

### 独立新架构（受 LLaMA 启发）

- **Mistral 7B**：新架构（Sliding Window Attention），以小模型在多个基准上超越 LLaMA 2 13B
- **Mixtral 8x7B**：Mistral 的 MoE 架构版本

### 量化与本地部署

- **llama.cpp**（Georgi Gerganov）：纯 C++ 实现，支持 CPU 推理，使 LLaMA 在普通笔记本上可运行
- **Ollama**：封装 llama.cpp，一行命令运行本地 LLM
- **LM Studio**：图形化本地 LLM 管理工具

## 本地部署可行性

LLaMA 系列开创了消费级硬件运行 LLM 的可能性：

| 模型 | 量化版本 | 推荐显存/内存 | 典型硬件 |
|------|---------|------------|--------|
| LLaMA 3 8B | Q4_K_M | 6-8 GB | RTX 3060 / M2 MacBook |
| LLaMA 3 8B | FP16 | 16 GB | RTX 4080 / M2 Pro |
| LLaMA 3 70B | Q4_K_M | 40-48 GB | 双 RTX 4090 / Mac Studio |
| LLaMA 3.1 405B | Q4 | ~200 GB | 8×A100 集群 |

量化（Quantization）是本地部署的关键技术：通过将 FP16（16位浮点）权重压缩至 4bit 或 8bit 整数，模型体积减少 75%，推理速度在 CPU 上可接受，精度损失通常在 5% 以内。

## Meta vs OpenAI：开源对决的深远影响

LLaMA 的开源策略对整个 AI 行业产生了深刻影响：

- **定价压力**：开源模型的存在迫使 OpenAI、Anthropic 等持续降低 API 价格
- **隐私与控制**：企业可以部署本地 LLaMA 模型，数据不离开内网
- **定制化**：允许在私有数据上微调，不依赖第三方 API
- **研究加速**：数百个研究团队基于 LLaMA 发表论文，推动对 LLM 机制的理解
- **监管讨论**：欧盟 AI 法规是否对开源模型同等约束成为政策争议热点
