# LlamaIndex 框架

LlamaIndex（原名 GPT Index）是专注于数据摄入与检索增强生成（RAG）的 Python 框架。相比 LangChain 的大而全，LlamaIndex 深耕数据连接与索引构建，在 RAG 场景下提供更精细的控制能力。

## 框架定位

LlamaIndex 的核心价值在于：

- **数据连接**：统一接口摄入来自各种来源的非结构化数据
- **智能索引**：根据数据特性选择最优索引结构
- **灵活查询**：支持多种查询策略和响应合成方式
- **生产就绪**：提供评估框架和可观测性工具

相比之下，LangChain 更侧重通用链式调用和 Agent 编排，而 LlamaIndex 在 RAG 管道的每个环节都提供更细粒度的控制。

## 数据连接器

LlamaIndex 提供大量内置数据加载器（通过 `llama-hub` 生态）：

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# 本地文件（支持 PDF、Word、TXT、Markdown、CSV 等）
documents = SimpleDirectoryReader(
    input_dir="./docs",
    recursive=True,
    required_exts=[".pdf", ".md", ".txt"]
).load_data()

print(f"加载了 {len(documents)} 个文档")
```

### WebReader（网页抓取）

```python
from llama_index.readers.web import SimpleWebPageReader

loader = SimpleWebPageReader(html_to_text=True)
documents = loader.load_data(urls=[
    "https://docs.python.org/3/library/asyncio.html",
    "https://fastapi.tiangolo.com/tutorial/"
])
```

### DatabaseReader（数据库）

```python
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    scheme="postgresql",
    host="localhost",
    port=5432,
    user="user",
    password="password",
    dbname="knowledge_base"
)

documents = reader.load_data(
    query="SELECT title, content FROM articles WHERE published = true"
)
```

其他支持的数据源包括：Notion、Confluence、Google Drive、Slack、GitHub、S3、MongoDB 等。

## 索引类型

LlamaIndex 提供多种索引结构，适应不同检索场景：

### VectorStoreIndex（向量索引）

最常用的索引，适合语义相似度检索：

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 全局配置
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 构建索引（自动分块和嵌入）
index = VectorStoreIndex.from_documents(documents)

# 持久化到磁盘
index.storage_context.persist(persist_dir="./storage")

# 从磁盘加载
from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

### TreeIndex（树形索引）

将文档构建成树状摘要层次，适合大文档的全局摘要查询：

```python
from llama_index.core import TreeIndex

tree_index = TreeIndex.from_documents(documents)
query_engine = tree_index.as_query_engine(retriever_mode="root")
response = query_engine.query("这些文档的主要结论是什么？")
```

### KeywordTableIndex（关键词索引）

基于关键词提取的索引，适合精确关键词匹配场景：

```python
from llama_index.core import KeywordTableIndex

keyword_index = KeywordTableIndex.from_documents(documents)
```

## 查询引擎

### QueryEngine（单次查询）

```python
query_engine = index.as_query_engine(
    similarity_top_k=5,           # 检索前 5 个相关块
    response_mode="compact",      # compact/refine/tree_summarize/accumulate
)

response = query_engine.query("什么是 Python 的协程？")
print(response.response)

# 查看来源节点
for node in response.source_nodes:
    print(f"来源：{node.metadata['file_name']}，相似度：{node.score:.3f}")
    print(node.text[:200])
```

### ChatEngine（多轮对话）

```python
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",  # 自动将对话历史融入检索查询
    verbose=True
)

response = chat_engine.chat("Python 中如何处理异常？")
print(response.response)

response = chat_engine.chat("那上面说的 except 和 finally 有什么区别？")  # 上下文感知
print(response.response)
```

## 节点解析（NodeParser）

节点是 LlamaIndex 的基础处理单元，NodeParser 控制文档如何被分割：

```python
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownNodeParser,
    CodeSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding

# 句子级别分割（推荐默认）
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)

# Markdown 感知分割（保留标题层次）
md_splitter = MarkdownNodeParser()

# 代码感知分割
code_splitter = CodeSplitter(language="python", chunk_lines=40)

# 语义分割（基于嵌入相似度决定分割边界）
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    embed_model=OpenAIEmbedding(),
)

nodes = splitter.get_nodes_from_documents(documents)
print(f"生成了 {len(nodes)} 个节点")
```

## 响应合成器

响应合成器控制如何将检索到的多个节点合并为最终回答：

- `compact`（默认）：将节点合并填满上下文窗口后一次生成
- `refine`：逐个节点迭代改进答案，质量较高但速度慢
- `tree_summarize`：构建摘要树，适合大量节点
- `accumulate`：对每个节点分别生成答案后合并

```python
from llama_index.core.response_synthesizers import get_response_synthesizer

synthesizer = get_response_synthesizer(response_mode="refine")
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
```

## 与 LangChain 的对比

| 维度 | LlamaIndex | LangChain |
|------|-----------|-----------|
| 核心定位 | RAG 和数据摄入 | 通用 LLM 应用编排 |
| RAG 功能深度 | 深（多种索引/检索策略） | 广（集成多，但较浅） |
| Agent 支持 | 基础 | 较完善（LangGraph） |
| 学习曲线 | 中等 | 中高（抽象层更多） |
| 数据加载器 | 专注文档数据 | 较全面 |
| 适合场景 | 知识库/文档 QA | 通用 AI 应用 |

选择建议：如果核心场景是 RAG 知识库，优先考虑 LlamaIndex；如果需要复杂 Agent 或大量第三方集成，考虑 LangChain 或两者结合使用。

## LlamaCloud 托管服务

LlamaCloud 是 LlamaIndex 官方的托管数据管道服务：

- **托管解析**：云端处理 PDF、图像等复杂格式文档
- **LlamaParse**：专业的文档解析服务，处理表格、图表等复杂布局
- **自动更新**：文档变更时自动触发重新索引
- **API 接口**：通过 REST API 管理索引

```python
from llama_index.core import VectorStoreIndex
from llama_parse import LlamaParse

# 使用 LlamaParse 处理复杂 PDF
parser = LlamaParse(
    api_key="llx-...",
    result_type="markdown",   # 将 PDF 转换为结构化 Markdown
    verbose=True
)

documents = parser.load_data("complex_report.pdf")
index = VectorStoreIndex.from_documents(documents)
```
