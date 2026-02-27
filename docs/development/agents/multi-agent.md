# 多智能体系统

当单个 Agent 面临复杂、多维度任务时，多智能体（Multi-Agent）系统通过角色分工和并行协作，显著提升效率和质量。本文介绍多 Agent 系统的架构设计、主流框架和实践挑战。

## 多 Agent 架构优势

### 并行执行

多个 Agent 可以同时处理任务的不同部分：

- 研究员 Agent 搜集资料时，写作 Agent 可以同步起草大纲
- 数据处理 Agent 处理不同数据源时互不阻塞
- 相比串行执行，总耗时可降低 50-80%

### 专业分工

不同 Agent 专注于各自擅长的领域：

- 专业的代码 Agent 具备代码相关工具和专业提示词
- 专业的数据分析 Agent 配备统计分析工具
- 每个 Agent 的系统提示更聚焦，减少"万能但平庸"的问题

### 互相验证

多个 Agent 对同一结果进行独立审查，提高输出质量：

- 生成 Agent 输出内容
- 批评 Agent 发现错误和不足
- 优化 Agent 根据批评改进
- 终版质量显著高于单 Agent

### 任务超出单一上下文窗口

复杂任务（如分析整个代码仓库）可能超出单个 Agent 的上下文容量，通过多 Agent 并行处理各部分后汇总。

## 角色设计

### Orchestrator（编排者）

负责整体任务规划、子任务分配和结果汇总：

```python
class OrchestratorAgent:
    def __init__(self, subagents: dict):
        self.subagents = subagents  # {"researcher": ..., "writer": ..., "critic": ...}

    def run(self, goal: str) -> str:
        # 1. 规划任务
        plan = self._create_plan(goal)

        # 2. 分配子任务
        results = {}
        for step in plan:
            agent_name = step["agent"]
            task = step["task"]
            context = {k: results[k] for k in step.get("needs", []) if k in results}

            results[step["id"]] = self.subagents[agent_name].execute(task, context)

        # 3. 汇总结果
        return self._synthesize(goal, results)

    def _create_plan(self, goal: str) -> list[dict]:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": f"""你是任务编排专家。可用的 Agent 有：{list(self.subagents.keys())}
                请规划任务步骤，输出 JSON：
                {{"steps": [{{"id": "step1", "agent": "researcher", "task": "...", "needs": []}}]}}"""
            }, {
                "role": "user",
                "content": f"请规划完成以下目标的步骤：{goal}"
            }],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)["steps"]
```

### Subagent（子代理）

专注执行特定类型的任务：

```python
class ResearcherAgent:
    """专注于信息搜集和研究的 Agent"""

    SYSTEM_PROMPT = """你是一个专业的研究员。
    你的职责是搜集、整理和分析信息。
    你有以下工具可用：网络搜索、学术数据库查询、数据提取。
    输出应该是有条理的摘要，包含来源引用。"""

    def execute(self, task: str, context: dict = None) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        if context:
            messages.append({
                "role": "user",
                "content": f"上下文信息：{context}\n\n任务：{task}"
            })
        else:
            messages.append({"role": "user", "content": task})

        # 执行带工具的推理循环
        return run_agent_loop(messages, tools=self.tools)


class CriticAgent:
    """负责审查和质量控制的 Agent"""

    SYSTEM_PROMPT = """你是一个严格的内容审查专家。
    你的职责是发现内容的错误、逻辑漏洞、不准确之处。
    以结构化格式输出问题列表，按严重程度排序。"""
```

### Critic（批评者）

独立审查其他 Agent 的输出，发现问题：

```python
def critic_review_loop(
    initial_output: str,
    task: str,
    max_iterations: int = 3
) -> str:
    """生成-批评-改进循环"""
    current_output = initial_output

    for i in range(max_iterations):
        # 批评者审查
        critique = critic_agent.review(current_output, task)

        if critique["issues_count"] == 0 or critique["severity"] == "none":
            break  # 质量达标，停止迭代

        # 改进者根据批评修正
        current_output = refiner_agent.improve(current_output, critique["issues"])

    return current_output
```

## AutoGen 框架

AutoGen（微软开源）是基于对话的多 Agent 框架，Agent 之间通过消息传递协作：

```python
import autogen

# 配置 LLM
config_list = [{"model": "gpt-4o", "api_key": "your-key"}]
llm_config = {"config_list": config_list}

# 创建 Agent
assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config=llm_config,
    system_message="你是一个 Python 专家，帮助解决编程问题。"
)

code_executor = autogen.UserProxyAgent(
    name="CodeExecutor",
    human_input_mode="NEVER",     # 不需要人工输入
    code_execution_config={
        "work_dir": "/tmp/autogen",
        "use_docker": False
    },
    max_consecutive_auto_reply=10
)

# 多 Agent 对话
code_executor.initiate_chat(
    assistant,
    message="写一个 Python 脚本，分析 CSV 文件并生成可视化报告"
)
```

AutoGen 特点：
- 内置代码执行和自动错误修复
- 支持人工介入（`human_input_mode="ALWAYS"`）
- GroupChat 支持多个 Agent 同时参与对话
- 适合需要代码生成和执行验证的任务

## CrewAI 框架

CrewAI 采用角色扮演范式，每个 Agent 有明确的角色、目标和工具：

```python
from crewai import Agent, Task, Crew, Process

# 定义角色
researcher = Agent(
    role="AI 研究员",
    goal="收集和分析关于 {topic} 的最新研究进展",
    backstory="你是一个有 10 年经验的 AI 研究员，擅长文献综述",
    tools=[search_tool, arxiv_tool],
    llm="gpt-4o",
    verbose=True
)

writer = Agent(
    role="技术写作专家",
    goal="将研究内容转化为清晰易懂的技术文章",
    backstory="你是一个技术博主，擅长将复杂概念简单化",
    tools=[],
    llm="gpt-4o"
)

# 定义任务
research_task = Task(
    description="研究 {topic} 的最新进展，收集关键论文和数据",
    expected_output="一份包含 5 个关键发现的研究摘要",
    agent=researcher
)

writing_task = Task(
    description="基于研究摘要，撰写一篇 800 字的技术博客文章",
    expected_output="完整的技术博客文章，包含标题和各章节",
    agent=writer,
    context=[research_task]  # 依赖研究任务的输出
)

# 创建团队并执行
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential  # sequential 或 hierarchical
)

result = crew.kickoff(inputs={"topic": "大型语言模型的推理能力"})
```

## 通信协议

### 消息传递（Message Passing）

Agent 之间通过结构化消息通信：

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class AgentMessage:
    sender: str
    recipient: str
    content: Any
    message_type: str  # "task" / "result" / "error" / "request"
    correlation_id: str  # 用于追踪请求-响应对
```

### 共享状态（Shared State）

Agent 通过共享状态对象协调：

```python
class SharedWorkspace:
    """多 Agent 共享的工作空间"""
    def __init__(self):
        self._state = {}
        self._lock = asyncio.Lock()

    async def update(self, key: str, value: Any, agent_id: str):
        async with self._lock:
            self._state[key] = {
                "value": value,
                "updated_by": agent_id,
                "timestamp": datetime.now()
            }

    async def read(self, key: str) -> Any:
        return self._state.get(key, {}).get("value")
```

### 黑板系统（Blackboard）

一种中心化的知识共享机制，Agent 向黑板写入/读取信息，由黑板协调任务调度。

## 协作模式

- **流水线（Pipeline）**：Agent A 的输出是 Agent B 的输入，顺序处理
- **辩论（Debate）**：多个 Agent 对同一问题持不同立场，通过辩论得出最优结论
- **投票（Voting）**：多个 Agent 独立生成答案，取多数票或加权平均
- **竞标（Bidding）**：任务发布到公共池，能力匹配的 Agent 竞标承接

## 可靠性与一致性挑战

多 Agent 系统引入了单 Agent 没有的挑战：

- **消息丢失**：Agent 之间的通信可能失败
- **状态不一致**：多个 Agent 并发修改共享状态时产生竞态条件
- **责任归因**：出错时难以定位是哪个 Agent 的问题
- **级联失败**：上游 Agent 失败导致下游 Agent 获得错误输入
- **死锁**：Agent A 等待 Agent B 的结果，同时 B 也在等待 A

## 成本控制

多 Agent 的成本是单 Agent 的数倍：

- 为不同角色选择不同规格的模型（协调者用 GPT-4o，执行者用 GPT-4o mini）
- 限制最大迭代轮次和 Agent 数量
- 对 Agent 间传递的消息进行压缩，只传递关键信息
- 缓存 Agent 的中间结果，避免重复计算
- 设置全局预算上限（如最多消耗 $5 的 Token），超出后报错
