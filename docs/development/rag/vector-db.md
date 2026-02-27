# 向量数据库选型

向量数据库专门设计用于存储和检索高维向量，是 RAG 系统的核心存储组件。本文对比主流向量数据库的特性，并提供选型决策框架。

## 向量数据库与传统数据库的区别

传统数据库（关系型/文档型）针对精确匹配查询优化（`WHERE id = 123`），而向量数据库的核心能力是**近似最近邻搜索（ANN）**：给定一个查询向量，在数百万条记录中快速找出向量空间中最相近的 K 条记录。

核心差异：
- **查询类型**：精确匹配 vs 语义相似度
- **索引结构**：B-Tree/Hash vs HNSW/IVF
- **距离度量**：无 vs 余弦相似度/欧氏距离/内积
- **数据类型**：结构化/半结构化 vs 高维浮点向量
- **规模能力**：行级优化 vs 十亿级向量检索

## ANN 算法

### HNSW（层次化可导航小世界图）

目前最流行的 ANN 算法，大多数生产级向量数据库使用：

- **原理**：构建多层跳表式图结构，高层稀疏用于粗定位，低层密集用于精确检索
- **优势**：查询速度快（毫秒级），召回率高（>95%），支持增量插入
- **劣势**：内存消耗大（约为向量存储的 2-3 倍），构建索引较慢
- **参数**：`M`（每层连接数）和 `ef_construction`（构建质量），影响速度/质量权衡

### IVF（倒排文件索引）

基于聚类的 ANN 算法：

- **原理**：将向量空间划分为 N 个 Voronoi 单元（聚类），查询时只搜索最近的几个单元
- **优势**：内存效率高，适合磁盘存储
- **劣势**：需要预先训练（需要大量向量），增量更新不便
- **参数**：`nlist`（聚类数）、`nprobe`（搜索单元数）

### PQ（乘积量化）

向量压缩技术，通常与 IVF 结合使用（IVF-PQ）：

- **原理**：将高维向量分段量化压缩，大幅减少存储空间
- **压缩比**：通常可压缩 8-32 倍
- **代价**：精度略有损失（约 1-3%）

## 主流产品对比

### FAISS（Facebook AI Similarity Search）

- **类型**：本地库（非服务型）
- **特点**：
  - Meta 开源，业界标准参考实现
  - 支持 CPU/GPU 加速
  - 支持多种索引（Flat/IVF/HNSW/PQ 及组合）
  - 纯内存或内存映射文件
- **适用**：离线批量处理、小型本地实验、作为其他系统的底层引擎

```python
import faiss
import numpy as np

dimension = 1536
index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW，M=32

# 批量添加向量
vectors = np.random.rand(10000, dimension).astype('float32')
index.add(vectors)

# 检索
query = np.random.rand(1, dimension).astype('float32')
distances, indices = index.search(query, k=10)  # Top-10
```

### Milvus（分布式，开源）

- **类型**：独立服务，分布式架构
- **特点**：
  - 水平扩展，支持百亿级向量
  - 支持多种索引类型（HNSW、IVF-PQ、DiskANN）
  - 多租户、Schema 管理
  - Zilliz Cloud 提供托管版本
- **适用**：大规模生产系统、需要高并发的场景

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("http://localhost:19530")

# 创建 Collection
client.create_collection(
    collection_name="documents",
    dimension=1536,
    metric_type="COSINE",
)

# 插入数据
client.insert("documents", [
    {"id": 1, "vector": [0.1] * 1536, "text": "文档内容", "source": "file.pdf"}
])

# 检索
results = client.search(
    collection_name="documents",
    data=[[0.1] * 1536],  # 查询向量
    limit=10,
    output_fields=["text", "source"],
    filter='source == "file.pdf"'  # 元数据过滤
)
```

### Qdrant（Rust 实现，高性能）

- **类型**：独立服务，Docker 部署
- **特点**：
  - Rust 实现，内存安全，性能优秀
  - 丰富的过滤器（支持嵌套 JSON 过滤）
  - 原生支持稀疏向量和混合检索
  - 支持 Payload（元数据）索引
  - Qdrant Cloud 提供托管版本

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("http://localhost:6333")

# 创建 Collection
client.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# 插入
client.upsert(
    collection_name="knowledge_base",
    points=[
        PointStruct(
            id=1,
            vector=[0.1] * 1536,
            payload={"text": "内容", "category": "tech", "date": "2024-01-01"}
        )
    ]
)

# 带过滤的检索
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="knowledge_base",
    query_vector=[0.1] * 1536,
    query_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="tech"))]
    ),
    limit=10,
)
```

### Chroma（轻量，适合开发）

- **类型**：可内嵌（进程内）或独立服务
- **特点**：
  - 零配置，嵌入模式无需启动服务
  - Python 原生，LangChain/LlamaIndex 深度集成
  - 自动 Embedding（内置 Sentence Transformers）
  - 适合原型开发和小规模应用

```python
import chromadb

# 内存模式（开发用）
client = chromadb.Client()
# 持久化模式
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.create_collection("my_docs")

collection.add(
    documents=["这是第一个文档", "这是第二个文档"],
    ids=["doc1", "doc2"],
    metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}]
)

results = collection.query(
    query_texts=["查询内容"],  # 自动 Embedding
    n_results=5
)
```

### pgvector（PostgreSQL 扩展）

- **类型**：PostgreSQL 扩展
- **特点**：
  - 在现有 PostgreSQL 中添加向量检索能力
  - 可与关系型数据联合查询（JOIN）
  - 无需引入新基础设施
  - 支持 HNSW 和 IVF 索引
  - Supabase、Neon 等云数据库原生支持

```sql
-- 启用扩展
CREATE EXTENSION vector;

-- 创建包含向量列的表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536),
    metadata JSONB
);

-- 创建 HNSW 索引
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- 向量检索
SELECT id, content, 1 - (embedding <=> '[0.1,...]'::vector) AS similarity
FROM documents
WHERE metadata->>'category' = 'tech'
ORDER BY embedding <=> '[0.1,...]'::vector
LIMIT 10;
```

### Pinecone（完全托管）

- **类型**：SaaS，完全托管
- **特点**：
  - 无需运维，开箱即用
  - 自动扩缩容，高可用 SLA
  - 支持命名空间（多租户隔离）
  - 混合检索（稠密+稀疏）
  - 按用量计费，免费层可用

## 混合搜索（Hybrid Search）

混合搜索将向量语义检索与关键词匹配结合，互补两者的优劣：

```python
# Qdrant 混合搜索示例
from qdrant_client.models import SparseVector, SparseIndexParams

# 需要同时提供稠密向量和稀疏向量（BM25）
results = client.query_points(
    collection_name="docs",
    prefetch=[
        {"query": dense_vector, "using": "dense", "limit": 20},
        {"query": SparseVector(indices=[1, 5, 9], values=[0.5, 0.3, 0.2]),
         "using": "sparse", "limit": 20},
    ],
    query={"fusion": "rrf"},  # RRF 融合排序
    limit=10
)
```

## 选型决策树

根据以下维度做决策：

- **数据规模 < 10 万条，开发/测试**：Chroma（内存模式）
- **数据规模 < 100 万条，有 PostgreSQL**：pgvector（复用基础设施）
- **数据规模 < 1000 万条，需要自托管**：Qdrant（性能好，运维简单）
- **数据规模 > 1000 万条，需要水平扩展**：Milvus
- **不想运维，预算充足**：Pinecone
- **离线批处理，GPU 可用**：FAISS
