# Embedding 模型选型

Embedding（嵌入）是将文本转换为稠密数值向量的过程，是 RAG 系统的核心组件之一。向量的质量直接决定了检索的语义理解能力。本文介绍 Embedding 模型的选型维度和主流模型的对比。

## Embedding 的基本概念

Embedding 模型将任意长度的文本映射到固定维度的向量空间中，语义相近的文本在向量空间中距离较近。

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

vector = get_embedding("深度学习是机器学习的子领域")
print(f"向量维度：{len(vector)}")  # 1536
print(f"向量前5维：{vector[:5]}")
```

## 选型关键维度

### 维度（Dimension）

向量维度越高，表达能力越强，但存储成本和计算成本也越高：
- 低维（256-512）：存储紧凑，检索快，表达能力有限
- 中维（768-1024）：均衡选择
- 高维（1536-3072）：表达能力强，适合精度要求高的场景

### 语言支持

- **单语言模型**：通常在目标语言上性能更好（如中文专用模型）
- **多语言模型**：适合跨语言检索，但可能在单语言上稍弱
- **中英双语**：BGE 系列、M3E 等在中英文均有良好表现

### 最大输入长度

- 大多数模型限制在 512 Token（约 300-400 中文字）
- 超出长度的文本会被截断，影响长文档的语义完整性
- 选择时需考虑文档块的平均长度

### 速度与成本

- API 模型：调用简单，按 Token 计费，有网络延迟
- 本地模型：一次性部署成本，后续免费，延迟低

## 主流模型对比

### OpenAI text-embedding-3-small

- **维度**：1536（可通过 `dimensions` 参数缩短至 256-1536）
- **最大长度**：8191 Token
- **定价**：$0.02 / 百万 Token
- **特点**：性价比最高，多语言效果稳定
- **适用**：大多数生产场景的首选

```python
# 使用降维节省存储（Matryoshka 表示）
response = client.embeddings.create(
    input=["文本示例"],
    model="text-embedding-3-small",
    dimensions=512  # 降维至 512，存储减少 2/3，性能轻微下降
)
```

### OpenAI text-embedding-3-large

- **维度**：3072（可缩短至 256-3072）
- **最大长度**：8191 Token
- **定价**：$0.13 / 百万 Token
- **特点**：更高精度，适合对质量要求严格的场景
- **适用**：法律、医疗等高精度需求场景

### Cohere embed-v3

- **维度**：1024
- **最大长度**：512 Token
- **特点**：支持 `input_type` 区分 `search_document` 和 `search_query`（对检索优化）
- **亮点**：专门区分文档端和查询端嵌入，检索精度高

```python
import cohere

co = cohere.Client("your-api-key")

# 文档端 Embedding
doc_embeddings = co.embed(
    texts=["文档内容..."],
    model="embed-multilingual-v3.0",
    input_type="search_document"
).embeddings

# 查询端 Embedding
query_embedding = co.embed(
    texts=["用户查询"],
    model="embed-multilingual-v3.0",
    input_type="search_query"
).embeddings
```

### BGE-M3（BAAI 开源）

- **维度**：1024
- **最大长度**：8192 Token
- **特点**：
  - 同时支持稠密检索、稀疏检索（BM25 风格）和多向量检索（ColBERT 风格）
  - 100+ 语言支持
  - 超长上下文（8K Token）
  - 可本地部署，免费使用

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

embeddings = model.encode(
    ["文档示例", "另一个文档"],
    batch_size=12,
    max_length=8192,
    return_dense=True,
    return_sparse=True,   # 同时返回稀疏向量用于混合检索
    return_colbert_vecs=True
)

dense_vecs = embeddings['dense_vecs']
sparse_vecs = embeddings['lexical_weights']
```

### M3E（Moka Massive Mixed Embedding）

- **维度**：768
- **最大长度**：512 Token
- **特点**：专门针对中文优化，中文短文本效果优秀
- **适用**：中文短文档场景，可本地部署

### bge-large-zh

- **维度**：1024
- **最大长度**：512 Token
- **特点**：BAAI 出品的中文优化版本，在中文 MTEB 榜单表现突出
- **适用**：中文知识库，对性能要求高的场景

## Matryoshka 表示学习

Matryoshka Representation Learning（MRL）允许同一个 Embedding 模型的向量在不同维度截断后仍保持良好性能：

```python
# text-embedding-3 系列支持 MRL
# 用高维向量精排，用低维向量粗筛
response = client.embeddings.create(
    input=[text],
    model="text-embedding-3-large",
    dimensions=256  # 从 3072 缩短到 256，减少 12 倍存储
)
```

实践策略：
- 存储 256 维向量做大规模 ANN 检索
- 对 Top-K 候选结果用完整 3072 维重新打分精排

## 批量 Embedding 最佳实践

```python
import asyncio
from openai import AsyncOpenAI

aclient = AsyncOpenAI()

async def batch_embed(texts: list[str], batch_size=100) -> list[list[float]]:
    """分批处理，避免超过 API 限制"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await aclient.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        batch_embeddings = [item.embedding for item in sorted(
            response.data, key=lambda x: x.index
        )]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

# 处理 10000 条文本
embeddings = asyncio.run(batch_embed(texts))
```

## 本地 Embedding vs API

| 维度 | 本地模型 | API |
|------|---------|-----|
| 初始成本 | 高（GPU 服务器） | 无 |
| 运营成本 | 固定（服务器费用） | 按用量计费 |
| 延迟 | 低（无网络） | 高（~100ms） |
| 数据隐私 | 完全本地 | 数据上传到第三方 |
| 质量上限 | 取决于开源模型 | 商业最优 |
| 维护成本 | 高 | 无 |

## 成本估算

以构建 100 万条文档的知识库为例（平均每条 200 Token）：

| 模型 | 总 Token | 成本 |
|------|---------|------|
| text-embedding-3-small | 2亿 Token | $4 |
| text-embedding-3-large | 2亿 Token | $26 |
| BGE-M3（本地） | 2亿 Token | $0（服务器费用另计） |

增量更新时，只需对新增/变更文档重新 Embedding，而非全量重算。
