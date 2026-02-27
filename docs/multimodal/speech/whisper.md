# Whisper 详解

Whisper 是 OpenAI 于 2022 年 9 月发布的开源语音识别模型，凭借在 68 万小时多语言语料上的大规模训练，实现了接近商业水准的多语言 ASR 效果，成为当前开源社区使用最广泛的语音识别系统。

## Whisper 背景

在 Whisper 发布之前，开源 ASR 系统要么性能较差，要么在特定语言上效果不佳，要么需要复杂的工程化部署。商业 ASR API（Azure、Google）质量更高，但需要付费且数据上传到云端。

Whisper 的出现填补了这一空白：
- **完全开源**：模型权重和代码以 MIT 许可证开放
- **强多语言能力**：99 种语言，尤其是英文达到人类水准
- **简单部署**：几行 Python 代码即可本地推理
- **端到端**：直接输出最终文字，无需复杂后处理

## 模型规模系列

Whisper 提供了从超轻量到旗舰的多个版本，适应不同的资源约束：

| 模型 | 参数量 | 相对速度 | VRAM 需求 | 适用场景 |
|------|--------|---------|----------|---------|
| tiny | 39M | ~32x | ~1GB | 极度资源受限，嵌入式 |
| base | 74M | ~16x | ~1GB | 轻量桌面应用 |
| small | 244M | ~6x | ~2GB | 精度/速度平衡 |
| medium | 769M | ~2x | ~5GB | 高精度，可接受延迟 |
| large-v1 | 1550M | 1x | ~10GB | 高精度 |
| large-v2 | 1550M | 1x | ~10GB | large-v1 改进版 |
| large-v3 | 1550M | 1x | ~10GB | 当前最佳，中文效果最好 |
| large-v3-turbo | 809M | ~3x | ~6GB | large-v3 蒸馏版，精度接近原版但快 3 倍 |

"相对速度"以 large 模型的推理速度为基准，以实际时长与音频时长的比值衡量（在 A100 GPU 上）。

## 架构设计

Whisper 采用标准的 **Encoder-Decoder Transformer** 架构，整体与机器翻译模型类似：

### 音频预处理

```
原始音频波形（16kHz）
  → 短时傅里叶变换（STFT）
  → 梅尔频谱（80 维 Mel filterbank，25ms 窗口，10ms 步进）
  → 归一化
```

每 30 秒音频处理为一个 80×3000 的梅尔频谱张量作为模型输入。

### 编码器

- 2 层 1D 卷积（降低时序分辨率，提取局部特征）
- N 层 Transformer Encoder（large 模型：32 层，16 头注意力，1024 维）
- 输出：音频的高维语义表示序列

### 解码器

- N 层 Transformer Decoder
- 交叉注意力（Cross-Attention）关注编码器输出
- 自回归生成输出 token 序列

### 特殊 Token 与多任务能力

Whisper 的解码器通过**特殊控制 token** 实现多任务能力：

```
<|startoftranscript|>     # 开始标志
<|zh|>                     # 语言标识（中文）
<|transcribe|>             # 任务类型：转录（保留原语言）
<|translate|>              # 任务类型：翻译（翻译成英文）
<|notimestamps|>           # 不输出时间戳
<|0.00|> 你好 <|0.80|>    # 带时间戳输出
```

解码器可以输出纯文字，也可以同时输出带时间戳的分词对齐，极大方便了字幕生成场景。

## 训练数据

Whisper 的强大能力主要来源于其训练数据规模：

- **总量**：约 **68 万小时**多语言音频-文字对
- **数据来源**：互联网收集（播客、视频字幕、YouTube 等），经过质动清洗
- **语言分布**：英文约 43 万小时，其他 98 种语言约 25 万小时
- **中文数据**：约 2.3 万小时，在非英文语种中排名靠前

大规模弱监督数据（自动生成的字幕、人工转录混合）配合简单的数据清洗，是 Whisper 泛化能力强的核心原因。

## 多语言支持

Whisper 支持 **99 种语言**的识别，以及从这 99 种语言到英文的翻译任务：

- **英文**：在多个基准上达到人类水准，WER < 3%（LibriSpeech clean）
- **中文**：large-v3 在普通话上效果优秀，WER 约 8-12%（标准测试集）
- **日语/韩语/法语/德语**等主要语言：效果良好
- **低资源语言**：训练数据少的语言效果有限

**自动语言检测**：无需指定语言，Whisper 会根据前 30 秒音频自动检测语言（概率输出），再选择相应的转录路径。

## 中文效果

Whisper 在中文场景的表现：
- **普通话**：large-v3 效果优秀，在较清晰的朗读/对话语音上 CER 可低至 3-5%
- **方言**：效果较差，粤语、闽南语等识别率低
- **中英混说**：large 模型有一定能力，但专业中文 ASR 系统（如科大讯飞）更佳
- **专业术语**：可通过 Initial Prompt 输入术语提示改善识别结果

## 本地部署

### 方式一：whisper.cpp（CPU 推理）

`whisper.cpp` 是 Whisper 的 C++ 移植版本，专为 CPU 推理优化：

```bash
# 编译
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp && make

# 下载模型（量化版本更小更快）
bash models/download-ggml-model.sh base.en
bash models/download-ggml-model.sh large-v3-q5_0  # Q5 量化，大幅减小体积

# 转录
./main -m models/ggml-base.en.bin -f audio.wav

# 实时麦克风转录
./stream -m models/ggml-base.en.bin
```

whisper.cpp 支持 Metal（Apple Silicon）、CUDA、OpenCL 等多种后端加速。在 M2 芯片 Mac 上，base 模型可实时转录。

### 方式二：faster-whisper（GPU 加速）

`faster-whisper` 使用 CTranslate2 推理引擎，比原版 OpenAI Whisper Python 实现快 4 倍，显存占用降低一半：

```python
from faster_whisper import WhisperModel

# 加载 large-v3 模型，INT8 量化
model = WhisperModel("large-v3", device="cuda", compute_type="int8")

segments, info = model.transcribe("audio.mp3", language="zh")

for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
```

### 方式三：原版 OpenAI Whisper

```bash
pip install openai-whisper
whisper audio.mp3 --model large-v3 --language zh
```

## Fine-tune 方法

在特定领域（医疗、法律、客服）微调 Whisper 可显著提升专业术语识别率：

```python
# 使用 Hugging Face Transformers 微调
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# 准备数据集（音频+对应标注文字）
# 使用 WhisperProcessor 处理音频为梅尔频谱
# 配置 Seq2SeqTrainingArguments
# 使用 Seq2SeqTrainer 训练
```

微调建议：
- 数据量：50-500 小时针对性数据即可显著改善
- 学习率：1e-5 到 5e-5，需要 warmup
- 批大小：受显存限制，通常使用梯度累积

## 与商业 API 对比

| 对比维度 | Whisper large-v3 | Azure Speech | 科大讯飞 |
|---------|-----------------|-------------|---------|
| 成本 | 本地零成本（硬件成本）| 按分钟计费 | 按次计费 |
| 隐私 | 完全本地 | 数据上传云端 | 数据上传云端 |
| 中文精度 | 良好 | 优秀 | 极佳 |
| 方言支持 | 有限 | 普通话 | 多方言 |
| 实时流式 | 需要额外工程 | 原生支持 | 原生支持 |
| 定制化 | 可微调 | 自定义语音模型（付费）| 有限 |

## 最佳实践

- **音频预处理**：去噪、去混响可显著提升识别率，推荐 SoX 或 ffmpeg 预处理
- **采样率**：输入统一转为 16kHz 单声道
- **语言指定**：已知语言时显式指定 `language` 参数，避免语言检测失误带来的错误
- **Initial Prompt**：对话开始前提供背景信息（如"以下是关于 AI 技术的讨论，包含术语如 Transformer、RAG、Embedding"），改善专业词汇识别
- **VAD（语音活动检测）**：长音频处理前用 Silero VAD 等工具预先切分，跳过静音段，减少计算量
- **批处理**：多段音频并行推理，最大化 GPU 利用率
