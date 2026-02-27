# Tokenization 与词汇表

Tokenization（分词）是将原始文本转化为模型可以处理的离散符号序列的过程。看似简单的预处理步骤，实际上对模型能力、训练效率和推理成本都有深远影响。

## 为何需要分词

神经网络本质上处理数值向量，无法直接接受文本字符串作为输入。分词的作用是在文本与数值之间建立映射：

1. **文本 → Token 序列**：将字符串切分为基本单元（Token）
2. **Token → 整数 ID**：通过词汇表（Vocabulary）将每个 Token 映射为整数索引
3. **整数 ID → 嵌入向量**：经过 Embedding 层转换为稠密向量，进入 Transformer 计算

分词粒度的选择直接决定了序列长度、词汇表大小和模型对罕见词的处理能力。

## 三种分词粒度

### 字符级分词（Character-Level）

将每个字符视为一个 Token：

- 词汇表极小（约 100-300 个字符）
- 无未登录词（OOV）问题
- 序列长度极长，导致注意力计算开销巨大
- 模型需要从字符学习词语和语义，学习负担重
- 实际应用极少，偶见于字符级语言模型研究

### 词级分词（Word-Level）

以空格或标点切分，每个词对应一个 Token：

- 词汇表通常需要数十万词才能覆盖常见词汇
- 频繁遇到未登录词（OOV），通常用 `<UNK>` 替代
- 无法处理形态丰富的语言（如芬兰语、阿拉伯语）
- 无法处理新词、网络用语、拼写变体

### 子词级分词（Subword-Level）

在字符级和词级之间取平衡，将词切分为有意义的子单元：

- 高频词保持完整（如 "the"、"is"）
- 低频词或罕见词被切分为更小的单元（如 "unbelievable" → "un" + "believ" + "able"）
- 词汇表大小适中（通常 30K-100K）
- 无 OOV 问题（最坏情况退化为字符）
- 目前所有主流 LLM 均采用子词分词

## BPE（Byte Pair Encoding）算法

BPE 最初是一种数据压缩算法，被 Sennrich 等人（2016）引入 NLP 领域，并被 GPT-2/GPT-3/LLaMA 等广泛采用。

### 算法步骤

1. **初始化**：将训练语料的所有词汇切分为字符序列，统计字符对频率
2. **迭代合并**：
   - 找到出现频率最高的相邻字符对（如 `e` + `s` → `es`）
   - 将该字符对合并为一个新 Token，加入词汇表
   - 更新语料中所有该字符对的出现
3. **重复**：持续执行合并操作，直到词汇表达到目标大小（如 50,000）

### 示例

原始词汇（带频率）：`low:5, lower:2, newest:6, widest:3`

初始字符化：`l o w </w>:5, l o w e r </w>:2, n e w e s t </w>:6, w i d e s t </w>:3`

合并次数最高的字符对 `e s` → `es`，然后 `es t` → `est`，以此类推。

### GPT-2 的 Byte-Level BPE

GPT-2 采用字节级 BPE（Byte-Level BPE）：

- 将 UTF-8 字节（256个）作为基础词汇，而非字符
- 完全消除 OOV 问题（任何 Unicode 文本都可以表示）
- 避免跨语言词汇表膨胀问题

## WordPiece（BERT 使用）

WordPiece 与 BPE 类似，但合并标准不同：

- **BPE**：选择频率最高的字符对进行合并
- **WordPiece**：选择合并后使语言模型似然提升最大的字符对（即选择最能提升语料拟合度的合并）

WordPiece 的子词以 `##` 前缀标记：

```
"playing" → ["play", "##ing"]
"unbelievable" → ["un", "##believ", "##able"]
```

BERT 的词汇表大小为 30,522（英文），包含 `[CLS]`、`[SEP]`、`[MASK]`、`[PAD]` 等特殊 Token。

## SentencePiece（T5、LLaMA 使用）

SentencePiece 由 Google 开发，解决了 BPE 和 WordPiece 对预分词（Pre-tokenization）的依赖：

- **语言无关**：直接从原始文本训练，不依赖空格分词，适合中文、日文等无空格语言
- **可逆性**：分词结果可无损还原为原始文本（通过 `▁` 标记行首空格）
- 支持 BPE 和 Unigram Language Model 两种子算法
- LLaMA 系列、T5、mT5、ALBERT 均使用 SentencePiece

```
"Hello world" → ["▁Hello", "▁world"]
"你好世界" → ["▁你好", "世界"]  # 取决于词汇表训练
```

## 词汇表大小与覆盖率的权衡

| 词汇表大小 | 优点 | 缺点 |
|-----------|------|------|
| 小（~8K） | Embedding 层参数少，内存占用小 | 序列更长，计算开销大；覆盖率低 |
| 中（~32K-50K） | 平衡点，多数模型的选择 | - |
| 大（~100K+） | 覆盖率高，多语言支持好 | Embedding 层参数量大；稀有 Token 训练不足 |

- GPT-2/GPT-3：50,257 个 Token
- LLaMA 2：32,000 个 Token
- LLaMA 3：128,256 个 Token（大幅扩充以改善多语言能力）
- Qwen2：151,936 个 Token（包含大量中文 Token）

## 中文分词的特殊性

中文没有天然的词间空格，带来独特挑战：

- **汉字级分词**：每个汉字作为一个 Token，序列较长但直觉简单
- **词级中文分词**：依赖分词工具（jieba、pkuseg），引入分词误差
- **BPE/SentencePiece**：从字节或字符出发自动学习中文子词，无需预分词；高频词组（如"人工智能"）会被合并为单个 Token

中文专用模型（如 Qwen、GLM）通常训练了包含大量中文高频词组的词汇表，使中文文本的每个 Token 平均携带更多语义信息，有效压缩序列长度。

## 特殊 Token

LLM 词汇表中包含若干具有特殊语义的 Token：

| Token | 含义 | 典型使用场景 |
|-------|------|------------|
| `<BOS>` / `<s>` | Beginning of Sequence，序列开始 | 生成时作为起始输入 |
| `<EOS>` / `</s>` | End of Sequence，序列结束 | 模型生成停止信号 |
| `<PAD>` | Padding，填充 | Batch 训练时对齐序列长度 |
| `<UNK>` | Unknown，未知词 | 词级模型中处理 OOV |
| `[SEP]` | Separator，分隔符 | BERT 中分隔句对 |
| `[CLS]` | Classification，分类 | BERT 序列级任务的聚合表示 |
| `[MASK]` | Mask，掩码 | BERT MLM 训练 |
| `<|im_start|>` | 对话轮次开始 | ChatML 格式（OpenAI/Qwen） |

## Token 计算对成本的影响

LLM API 通常按 Token 数量计费（输入 + 输出分别计价）。Token 计算直接影响使用成本和上下文长度限制：

- **英文**：平均约 1 个 Token 对应 4 个字符（或约 0.75 个单词）
- **中文**：每个汉字通常对应 1-2 个 Token（取决于模型词汇表）
- **代码**：缩进、括号等符号会产生额外 Token

实际影响举例：

- 一篇 1000 字的中文文章，约占用 1000-2000 个 Token
- GPT-4 的 128K 上下文窗口，对应约 6-12 万中文汉字
- 长文档 RAG 场景中，Token 数量直接决定能放入上下文的信息量

工具推荐：OpenAI 的 `tiktoken` 库可精确计算 GPT 系列的 Token 数量；Hugging Face `tokenizers` 库支持几乎所有主流模型的分词器。
