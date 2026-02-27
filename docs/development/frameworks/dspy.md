# DSPy 编程框架

DSPy（Declarative Self-improving Python）是斯坦福大学 NLP 组开发的 LLM 编程框架，代表了一种与 LangChain、LlamaIndex 截然不同的设计哲学：将逻辑与提示分离，通过自动优化取代手动提示工程。

## DSPy 的核心理念

传统 LLM 开发的痛点：
- 提示（Prompt）脆弱，换一个模型或版本就需要重新调整
- 手动提示工程耗时且不可复现
- 提示优化依赖直觉，缺乏系统性

DSPy 的解决思路类比神经网络：就像神经网络通过反向传播自动调整权重，DSPy 通过优化器（Teleprompter）自动调整提示和少样本示例，使模型行为满足定义的评估指标。

核心原则：
- **声明式**：描述"做什么"，而不是"怎么做"（写什么提示）
- **模块化**：将复杂程序拆解为可组合的模块
- **自动优化**：基于评估数据自动找到最优提示和示例

## 核心组件

### Signature（签名）

Signature 定义模块的输入输出类型和含义，类似函数签名：

```python
import dspy

# 简洁语法：输入 -> 输出
class Classify(dspy.Signature):
    """将用户评论分类为正面、负面或中性"""
    comment: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()

# 也可以用字符串语法（快速原型）
classify = dspy.Predict("comment -> sentiment")
```

### Module（模块）

模块是 DSPy 程序的基础构建块，封装了 LLM 调用逻辑：

```python
class RAGPipeline(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        answer = self.generate(context=context, question=question)
        return answer

# 使用
rag = RAGPipeline(num_passages=5)
result = rag(question="Python 中的 asyncio 是如何工作的？")
print(result.answer)
```

### Teleprompter（优化器）

Teleprompter 是 DSPy 的"训练器"，通过少量标注数据自动优化程序：

```python
from dspy.teleprompt import BootstrapFewShot

# 定义评估指标
def validate_answer(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# 准备少量训练样本（10-50 条即可）
trainset = [
    dspy.Example(question="Python 列表和元组的区别？", answer="列表可变，元组不可变").with_inputs("question"),
    # ... 更多示例
]

# 自动优化
optimizer = BootstrapFewShot(metric=validate_answer, max_bootstrapped_demos=4)
optimized_rag = optimizer.compile(RAGPipeline(), trainset=trainset)

# 保存优化后的程序
optimized_rag.save("optimized_rag.json")
```

## 内置模块

### Predict（基础预测）

最简单的模块，直接按 Signature 调用 LLM：

```python
predict = dspy.Predict(Classify)
result = predict(comment="这个产品真的很好用！")
print(result.sentiment)  # "positive"
```

### ChainOfThought（思维链）

自动添加推理步骤，提高复杂问题的准确率：

```python
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="如果苹果比橘子贵，橘子比香蕉贵，那么苹果和香蕉哪个更贵？")
print(result.reasoning)  # 推理过程
print(result.answer)     # 最终答案
```

### ReAct（推理+行动）

ReAct 模块实现 Reasoning + Acting 循环，支持工具调用：

```python
def search_web(query: str) -> str:
    """搜索互联网获取信息"""
    # 实际实现
    return "搜索结果..."

react = dspy.ReAct("question -> answer", tools=[search_web])
result = react(question="2024 年的诺贝尔物理学奖得主是谁？")
```

### Retrieve（检索）

与向量数据库集成的检索模块：

```python
# 配置检索后端（如 ColBERT、Weaviate 等）
dspy.settings.configure(rm=MyRetriever())

retrieve = dspy.Retrieve(k=5)
passages = retrieve("Python 异步编程").passages
```

## 优化器

### BootstrapFewShot

最基础的优化器，从训练集中自动选择最优的少样本示例：

- 运行程序生成预测，保留满足评估指标的示例
- 将这些示例作为 Few-shot 嵌入提示
- 适合小数据集（10-50 条）

### MIPROv2

多阶段优化器，同时优化提示指令和少样本示例：

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=validate_answer,
    auto="medium",           # light/medium/heavy，控制搜索力度
)
optimized = optimizer.compile(
    program,
    trainset=trainset,
    num_trials=30            # 优化迭代次数
)
```

MIPROv2 适合需要高质量优化且有足够算力预算的场景。

## 与 LangChain 的根本差异

| 维度 | DSPy | LangChain |
|------|------|-----------|
| 提示管理 | 自动优化（不写提示） | 手动编写提示模板 |
| 核心抽象 | Signature + Module | Chain + Prompt |
| 学习曲线 | 较陡（需理解新范式） | 中等 |
| 集成生态 | 较少 | 非常丰富 |
| 适合阶段 | 需要系统优化时 | 快速集成原型 |
| 可解释性 | 高（程序结构清晰） | 中等 |

## 适用场景

DSPy 特别适合以下场景：

- **信息提取**：从文档中提取结构化数据，需要高精度
- **分类任务**：文本分类、意图识别，有评估数据集可用
- **问答系统**：有标准答案可评估的 RAG 系统
- **学术研究**：需要可复现的 NLP 实验
- **跨模型迁移**：同一程序在不同 LLM 上自动重新优化

不适合的场景：
- 需要快速集成大量第三方服务（用 LangChain）
- 没有任何评估数据（优化器无法工作）
- 以创意生成为主（优化目标不明确）

## 代码示例：完整 QA 系统

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# 1. 配置 LLM
lm = dspy.LM("openai/gpt-4o", api_key="...")
dspy.configure(lm=lm)

# 2. 定义程序
class QAWithSources(dspy.Module):
    def __init__(self):
        self.generate_query = dspy.ChainOfThought("question -> search_query")
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_answer = dspy.ChainOfThought(
            "context, question -> answer, confidence"
        )

    def forward(self, question):
        query = self.generate_query(question=question).search_query
        context = self.retrieve(query).passages
        result = self.generate_answer(
            context="\n".join(context), question=question
        )
        return dspy.Prediction(
            answer=result.answer,
            confidence=result.confidence
        )

# 3. 优化
qa = QAWithSources()
optimizer = BootstrapFewShot(metric=lambda ex, pred, trace: ex.answer in pred.answer)
optimized_qa = optimizer.compile(qa, trainset=trainset)

# 4. 使用
result = optimized_qa(question="深度学习的反向传播算法是如何工作的？")
print(f"回答：{result.answer}")
print(f"置信度：{result.confidence}")
```
