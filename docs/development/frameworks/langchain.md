# LangChain 开发框架

LangChain 是目前下载量最大的 LLM 应用框架，提供构建对话应用、RAG 系统和 Agent 所需的标准化抽象与海量集成。本文介绍其核心概念、优缺点和最佳实践。

## 核心抽象

### Chain（链）

Chain 是 LangChain 的基础单元，代表一系列处理步骤的组合。传统 Chain（如 `LLMChain`）已逐步被 LCEL 取代。

### Agent

Agent 使 LLM 能够动态决策调用哪些工具，并根据工具结果继续推理，直到完成任务。

### Memory

Memory 模块负责对话历史的存储与管理，包括：
- `ConversationBufferMemory`：保存完整历史
- `ConversationSummaryMemory`：对历史进行摘要压缩
- `ConversationBufferWindowMemory`：保留最近 K 轮

### Tool（工具）

工具是 Agent 可以调用的函数，LangChain 内置了大量工具（搜索、计算、数据库查询等），也支持自定义。

### Retriever（检索器）

检索器负责从知识库中获取相关文档，屏蔽了底层向量数据库的差异。

## LCEL（LangChain Expression Language）

LCEL 是 LangChain 0.1 引入的声明式链构建语法，用管道符 `|` 组合组件：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{domain}专家。"),
    ("human", "{question}")
])

# 构建链（声明式）
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

chain = prompt | model | parser

# 调用
result = chain.invoke({
    "domain": "机器学习",
    "question": "什么是 Dropout 正则化？"
})
print(result)

# 流式输出
for chunk in chain.stream({"domain": "机器学习", "question": "解释 Attention 机制"}):
    print(chunk, end="", flush=True)

# 批量处理
results = chain.batch([
    {"domain": "物理", "question": "解释相对论"},
    {"domain": "化学", "question": "什么是共价键"},
])
```

LCEL 的优势：
- 统一的 `invoke`/`stream`/`batch`/`astream` 接口
- 自动支持异步
- 可视化追踪（配合 LangSmith）
- 组件可自由组合

## 内置集成

LangChain 提供 100+ 集成，覆盖：

- **LLM**：OpenAI、Anthropic、Google、Cohere、Hugging Face、Ollama（本地）
- **向量数据库**：Chroma、Pinecone、Milvus、Qdrant、pgvector、FAISS
- **文档加载器**：PDF、Word、Excel、Web、Notion、Confluence、GitHub
- **工具**：Google Search、Wikipedia、Wolfram Alpha、SQL 数据库
- **Embedding**：OpenAI、HuggingFace、Cohere、Bedrock

## RAG 示例

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# 1. 加载文档
loader = WebBaseLoader("https://docs.python.org/3/tutorial/")
docs = loader.load()

# 2. 分块
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# 3. 创建向量库
vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())

# 4. 构建 QA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
)

result = qa_chain.invoke({"query": "Python 中如何使用列表推导式？"})
```

## LangSmith 调试与追踪

LangSmith 是 LangChain 官方的可观测平台，对于复杂的多步骤链和 Agent 调试极为重要：

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="ls__..."
export LANGCHAIN_PROJECT="my-rag-project"
```

启用后，所有链的执行都会自动上传到 LangSmith 仪表盘，可查看：
- 每个步骤的输入输出
- Token 用量和延迟
- 错误和异常的完整调用栈
- 对比不同配置的 A/B 测试

## LangGraph 状态机 Agent

LangGraph 是 LangChain 生态中专门用于构建有状态 Agent 的库，基于有向图（DAG）：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[dict]
    next_step: str

def should_continue(state: AgentState) -> str:
    """决定下一步走哪个节点"""
    if state["next_step"] == "tool":
        return "tool_node"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tool_node", tool_execution_node)
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tool_node", "agent")
graph.set_entry_point("agent")

app = graph.compile()
```

LangGraph 适合需要循环、条件分支、人工介入（Human-in-the-loop）的复杂 Agent。

## 适用场景与争议

### 推荐使用 LangChain 的场景

- 快速原型验证，利用丰富的预置集成
- 团队已有 LangChain 经验的项目
- 需要统一管理多种 LLM 提供商
- 配合 LangSmith 建立调试/评估体系

### 常见争议

- **过度抽象**：简单任务引入了不必要的复杂度，调试困难
- **频繁 Breaking Change**：版本升级间的 API 变化较大
- **性能开销**：抽象层带来额外延迟，性能敏感场景不适合
- **与直接 SDK 对比**：对于固定调用逻辑，官方 SDK 代码更简洁易维护

### 版本迁移痛点

- LangChain 0.1 → 0.2 → 0.3 有多次大规模重构
- `langchain` 包拆分为 `langchain-core`、`langchain-community`、`langchain-openai` 等
- 建议锁定依赖版本（`langchain==0.3.x`），升级前仔细查看 Changelog

## 与直接调用 SDK 的对比

| 维度 | LangChain | 直接 SDK |
|------|-----------|---------|
| 上手速度 | 快（预置组件多） | 慢（需自行实现） |
| 调试难度 | 高（抽象层多） | 低（代码直观） |
| 灵活性 | 中（框架限制） | 高（完全控制） |
| 性能 | 中（有额外开销） | 高 |
| 适合规模 | 原型/中等 | 任意规模 |

建议：先用官方 SDK 理解底层机制，再评估是否引入 LangChain 简化集成工作。
