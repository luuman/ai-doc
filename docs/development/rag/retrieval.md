# 检索与重排优化

基础的向量相似度检索在实践中往往无法达到生产级别的要求。本文介绍从稠密检索到混合检索、从基础检索到高级优化的完整技术体系。

## 稠密检索 vs 稀疏检索

### 稠密检索（Dense Retrieval）

基于 Embedding 向量的语义相似度检索：

- **原理**：将查询和文档都编码为稠密向量，计算余弦相似度
- **优势**：能理解语义，捕捉同义词、近义词、隐含含义
- **劣势**：对专业术语、缩写、代码变量名等词汇精确匹配效果差

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

query_vector = get_embedding("机器学习算法")
doc_vectors = [get_embedding(doc) for doc in documents]

similarities = cosine_similarity([query_vector], doc_vectors)[0]
top_k_indices = np.argsort(similarities)[-10:][::-1]
```

### 稀疏检索（Sparse Retrieval）

基于关键词频率的传统信息检索：

**BM25（Best Match 25）** 是最常用的稀疏检索算法，对 TF-IDF 的改进版本：

```python
from rank_bm25 import BM25Okapi
import jieba

# 中文分词
def tokenize_zh(text):
    return list(jieba.cut(text))

corpus_tokenized = [tokenize_zh(doc) for doc in documents]
bm25 = BM25Okapi(corpus_tokenized)

query_tokens = tokenize_zh("Python 异步编程 asyncio")
scores = bm25.get_scores(query_tokens)
top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
```

稀疏检索的优势：
- 对精确关键词匹配效果极好（商品型号、人名、专有名词）
- 无需 Embedding 计算，速度快
- 可解释性强（可以看到哪些词贡献了得分）

## 混合检索（Hybrid Search）

将稠密检索和稀疏检索的结果融合，取长补短：

### RRF（Reciprocal Rank Fusion）融合排序

```python
def reciprocal_rank_fusion(
    search_results: list[list[tuple]], k: int = 60
) -> list[tuple]:
    """
    融合多路检索结果
    search_results: 每路结果为 [(doc_id, score), ...] 列表
    k: RRF 参数，通常设为 60
    """
    rrf_scores = {}

    for results in search_results:
        for rank, (doc_id, score) in enumerate(results, 1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

# 使用
dense_results = dense_search(query, top_k=20)
sparse_results = bm25_search(query, top_k=20)
fused_results = reciprocal_rank_fusion([dense_results, sparse_results])
```

### Qdrant 原生混合检索

```python
from qdrant_client.models import SparseVector, FusionQuery, Prefetch

results = client.query_points(
    collection_name="documents",
    prefetch=[
        Prefetch(query=dense_query_vector, using="dense", limit=20),
        Prefetch(
            query=SparseVector(indices=[100, 500, 2000], values=[0.5, 0.3, 0.8]),
            using="sparse",
            limit=20
        ),
    ],
    query=FusionQuery(fusion="rrf"),
    limit=10
)
```

## 重排器（Reranker）

重排器是在初步检索之后，用更精确的模型对候选结果重新打分排序的组件：

- **检索阶段**：快速从大量文档中找出候选（Top-50 或 Top-100）
- **重排阶段**：用精度更高的交叉编码器（Cross-Encoder）对候选精排
- **生成阶段**：取重排后 Top-5 到 Top-10 注入 LLM

### BGE-Reranker（本地）

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# 输入为 [查询, 文档] 对
pairs = [[query, doc] for doc in candidate_docs]
scores = reranker.compute_score(pairs, normalize=True)

# 按分数排序
ranked_results = sorted(
    zip(candidate_docs, scores),
    key=lambda x: x[1],
    reverse=True
)
top_5 = [doc for doc, score in ranked_results[:5]]
```

### Cohere Rerank（API）

```python
import cohere

co = cohere.Client("your-api-key")

results = co.rerank(
    model="rerank-multilingual-v3.0",
    query=query,
    documents=candidate_docs,
    top_n=5,
    return_documents=True
)

for result in results.results:
    print(f"分数：{result.relevance_score:.4f}")
    print(f"内容：{result.document.text[:200]}")
```

重排器显著提升精度的原因：双塔模型（Bi-Encoder，用于 Embedding）分别编码查询和文档，而交叉编码器（Cross-Encoder，用于 Reranker）将查询和文档一起输入，能够捕捉更细腻的相关性信号。

## HyDE（假设文档嵌入）

HyDE 是一种反直觉但效果显著的检索优化技术：

- **问题**：用户问题（短、口语化）和文档（长、专业）的向量分布差距大，导致检索效果不佳
- **解决方案**：让 LLM 先生成一个假设性的答案，用假设答案的向量做检索

```python
from openai import OpenAI

client = OpenAI()

def hyde_retrieve(query: str, retriever, top_k: int = 5):
    # 1. 生成假设文档
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是专业的技术文档作者，请生成一段回答以下问题的专业文档段落："},
            {"role": "user", "content": query}
        ],
        max_tokens=300
    )
    hypothetical_doc = response.choices[0].message.content

    # 2. 用假设文档的向量检索
    hyp_vector = get_embedding(hypothetical_doc)
    results = retriever.retrieve_by_vector(hyp_vector, top_k=top_k)

    return results

results = hyde_retrieve("Python 中如何实现线程安全的单例模式？", retriever)
```

## 查询扩展与改写

### 查询改写

将用户的自然语言问题改写为更适合检索的形式：

```python
def rewrite_query(original_query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""将以下用户问题改写为更适合文档检索的搜索查询。
            要求：去除口语化表达，使用专业术语，突出关键词。

            原始问题：{original_query}
            改写后的查询："""
        }]
    )
    return response.choices[0].message.content.strip()
```

### MultiQueryRetriever（多查询检索）

从不同角度生成多个查询，扩大检索覆盖面：

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI(model="gpt-4o-mini")
)

# 会自动生成多个相关查询并合并结果
docs = retriever.get_relevant_documents("如何优化 Python 代码性能？")
# 内部生成如：["Python 性能优化技巧", "Python 代码加速方法", "Python 分析工具 profiler"]
```

## 上下文压缩

从检索到的大块文档中提取最相关的片段，减少注入 LLM 的噪声：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 使用 LLM 提取最相关句子
compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-4o-mini"))
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

compressed_docs = compression_retriever.get_relevant_documents(query)
# 只返回文档中真正相关的句子，而非整块
```

## GraphRAG（知识图谱增强检索）

GraphRAG 是微软提出的 RAG 增强方案，将知识图谱与向量检索结合：

核心思路：
- 从文档中提取实体和关系，构建知识图谱
- 对知识图谱进行社区检测，生成各社区的摘要（Global Search）
- 检索时结合图结构关系，找到相关实体周围的文档（Local Search）

适用场景：
- 需要跨文档推理关系（"A 和 B 有什么间接联系？"）
- 需要全局性的综述（"文档集合的整体主题是什么？"）
- 实体关系密集的领域（医疗、法律、金融）

局限性：
- 构建图谱成本高（需要多次 LLM 调用）
- 维护图谱随文档更新的一致性复杂
- 对单一事实查询的提升有限，不值得引入额外复杂度

```bash
# 使用 Microsoft GraphRAG 库
pip install graphrag
python -m graphrag.index --root ./data
python -m graphrag.query --root ./data --method global "AI 领域的主要研究方向"
```
