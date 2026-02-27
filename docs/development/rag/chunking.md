# 文档分块策略

文档分块（Chunking）是 RAG 管道中最容易被忽视却至关重要的环节。块的大小、边界和内容质量直接决定检索精度，进而影响最终回答质量。本文系统介绍各种分块策略及其适用场景。

## 分块的重要性

分块决定了以下关键问题：

- **检索粒度**：块太大则噪声多，块太小则缺乏上下文
- **语义完整性**：在错误位置截断会破坏语义
- **Token 预算**：块大小影响注入到 LLM 上下文的信息量
- **检索召回率**：合理的块边界让相关信息更集中，更容易被检索到

一个经验规则：如果检索结果质量差，先检查分块策略，再考虑换检索算法。

## 固定大小分块

最简单的策略，按字符数或 Token 数固定分割：

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,      # 每块最多 1000 个字符
    chunk_overlap=200,    # 相邻块重叠 200 个字符（保留上下文连续性）
    separator="\n",       # 优先在换行处切分
)

chunks = splitter.split_text(long_document)
```

`chunk_overlap` 的作用：确保跨块边界的信息不会丢失。例如一句话跨越两块边界时，两块都包含这句话，检索时不会漏掉。

**缺点**：完全不考虑语义边界，可能在句子中间截断，破坏语义完整性。

## 递归文本分割

递归地尝试不同分隔符，优先在语义边界处切分：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    # 按优先级依次尝试以下分隔符
    separators=[
        "\n\n",   # 段落
        "\n",     # 行
        "。",     # 句子（中文）
        ".",      # 句子（英文）
        "，",     # 分句
        ",",
        " ",      # 单词
        "",       # 字符（最后手段）
    ]
)
```

这是最常用的通用分块方案，适合普通文本内容。

## 按标题分块

专为 Markdown 文档设计，按标题层次分割：

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]
)

chunks = splitter.split_text(markdown_content)

# 每个 chunk 自动包含标题元数据
for chunk in chunks:
    print(chunk.metadata)  # {"H1": "第一章", "H2": "第一节"}
    print(chunk.page_content)
```

标题元数据可用于过滤和引用，非常适合技术文档、手册类内容。

## 语义分块

基于嵌入相似度动态决定分割边界，语义变化大的地方切分：

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # 百分位数阈值
    breakpoint_threshold_amount=95,           # 相似度最低的 5% 处切分
)

chunks = splitter.create_documents([long_text])
```

工作原理：
1. 将文本按句子分割
2. 计算相邻句子的 Embedding 相似度
3. 在相似度显著下降处（语义转换点）切分

优势：切分点更符合语义边界，但需要额外的 Embedding 计算成本。

## 文档特定分块

### Markdown 感知分块

```python
from llama_index.core.node_parser import MarkdownNodeParser

parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(markdown_documents)
# 自动按标题、列表、代码块等结构元素分割
```

### 代码分块

代码需要保持函数/类的完整性：

```python
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200,
)

# 优先在函数、类定义处切分
code_chunks = python_splitter.split_text(python_code)
```

### HTML 分块

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
)

chunks = splitter.split_text(html_content)
```

## Chunk Size 的影响

| 参数 | 小块（256-512 Token） | 大块（1024-2048 Token） |
|------|----------------------|------------------------|
| 检索精度 | 高（更精确定位信息） | 低（噪声多） |
| 上下文完整性 | 低（信息可能被截断） | 高（完整段落/章节） |
| 检索召回率 | 低（相关信息可能分散在多块） | 高（完整语境提高相关度） |
| 存储开销 | 高（块数量多） | 低 |
| 适合场景 | 精确事实查询 | 概念理解、综述性问题 |

**实践建议**：大多数场景从 512 Token 块大小、50-100 Token 重叠开始，根据评估结果调整。

## Parent-Child Chunks（父子块）

一种将检索精度与上下文完整性结合的策略：

- **Child Chunks**：小块（128-256 Token），用于精确检索
- **Parent Chunks**：大块（1024-2048 Token），用于提供完整上下文给 LLM

工作流程：
1. 同时存储大块和小块（小块保存对应大块的引用）
2. 检索时用小块做语义匹配
3. 找到小块后，获取对应大块作为 LLM 上下文

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # 三层：大->中->小
)

all_nodes = parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(all_nodes)  # 最小粒度节点用于检索
```

## Small-to-Big 检索策略

与 Parent-Child 类似，但更灵活：

1. 用小 chunk 做语义检索，获得精确匹配
2. 根据检索结果的 chunk ID，获取其上下文（前后相邻 chunk，或父级 chunk）
3. 将扩展后的上下文注入 LLM

```python
# LlamaIndex 实现
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore

docstore = SimpleDocumentStore()
docstore.add_documents(all_nodes)

base_retriever = index.as_retriever(similarity_top_k=12)
retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    verbose=True,
    simple_ratio_thresh=0.4  # 40% 以上子节点被选中时，升级到父节点
)
```

## 分块质量评估

评估分块质量的实用方法：

- **人工审查**：随机抽取 20-30 个块，检查是否在语义完整处切分
- **长度分布**：分析块大小分布，过于均匀说明固定大小截断，过于悬殊需要调整
- **端对端评估**：用 RAGAS 框架对比不同分块策略的最终回答质量
- **检索召回测试**：准备已知答案的问题集，验证相关块是否被正确检索到
