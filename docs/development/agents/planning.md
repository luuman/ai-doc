# Agent 规划与推理

规划能力是区分简单 LLM 调用和真正 Agent 的关键特征。优秀的规划使 Agent 能够分解复杂目标、应对意外情况并高效完成任务。本文介绍主流的 Agent 规划与推理方法。

## 任务分解（Task Decomposition）

任务分解是将复杂目标分解为 LLM 和工具可执行的子任务的过程：

```python
from openai import OpenAI

client = OpenAI()

def decompose_task(goal: str) -> list[dict]:
    """将复杂目标分解为子任务列表"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """你是一个任务规划专家。将用户的目标分解为具体的、可执行的子任务。
                每个子任务应该：
                1. 足够具体，可以用一个工具调用或一次 LLM 推理完成
                2. 有明确的输入和输出
                3. 包含依赖关系（哪些任务必须先完成）

                以 JSON 格式输出：{"tasks": [{"id": 1, "name": "...", "description": "...", "depends_on": []}]}"""
            },
            {"role": "user", "content": f"请分解以下任务：{goal}"}
        ],
        response_format={"type": "json_object"}
    )

    import json
    return json.loads(response.choices[0].message.content)["tasks"]

# 示例
tasks = decompose_task("撰写一份关于 2024 年 AI 发展现状的市场分析报告")
# 可能分解为：搜集数据 → 分析趋势 → 竞争格局分析 → 撰写各章节 → 整合校对
```

## ReAct 推理循环

ReAct（Reasoning + Acting）交替进行推理和行动，每次行动后都基于观察结果继续推理：

```python
def react_agent(goal: str, tools: dict, max_steps: int = 10) -> str:
    messages = [
        {
            "role": "system",
            "content": """你是一个 ReAct Agent。按照以下格式思考和行动：
            思考：分析当前状态，决定下一步
            行动：调用工具获取信息或执行操作
            观察：分析工具返回的结果
            ...（重复直到可以给出最终答案）
            最终答案：[回答]"""
        },
        {"role": "user", "content": goal}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=list(tools.values()),
        )

        message = response.choices[0].message
        messages.append(message)

        if response.choices[0].finish_reason == "stop":
            return message.content  # 达到最终答案

        if message.tool_calls:
            for tool_call in message.tool_calls:
                result = execute_tool(tool_call.function.name,
                                      json.loads(tool_call.function.arguments))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

    return "已达到最大步骤数，任务未完成"
```

## Plan-and-Execute 架构

与 ReAct 的边想边做不同，Plan-and-Execute 先制定完整计划，再逐步执行：

```python
class PlanAndExecuteAgent:
    def __init__(self):
        self.planner = self._create_planner()
        self.executor = self._create_executor()
        self.replanner = self._create_replanner()

    def run(self, goal: str) -> str:
        # 第一步：制定完整计划
        plan = self._plan(goal)
        print(f"制定计划：{plan}")

        completed_steps = []
        remaining_steps = plan.copy()

        while remaining_steps:
            current_step = remaining_steps[0]

            # 执行当前步骤
            result = self._execute_step(current_step, completed_steps)
            completed_steps.append({"step": current_step, "result": result})
            remaining_steps.pop(0)

            # 重规划：根据执行结果调整剩余计划
            if remaining_steps:
                remaining_steps = self._replan(
                    goal, completed_steps, remaining_steps, result
                )

        # 综合所有步骤结果给出最终答案
        return self._synthesize(goal, completed_steps)

    def _plan(self, goal: str) -> list[str]:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"为以下目标制定详细的执行计划，列出步骤：{goal}"
            }],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)["steps"]
```

Plan-and-Execute 的优势在于可以在执行前审查整体计划，适合需要人工审批的场景（Human-in-the-loop）。

## Tree of Thoughts（思维树）

ToT 将推理过程建模为树状搜索，探索多条推理路径并选择最优：

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ThoughtNode:
    thought: str
    score: float
    children: list['ThoughtNode']
    parent: Optional['ThoughtNode'] = None

def tree_of_thoughts(problem: str, breadth: int = 3, depth: int = 4) -> str:
    """
    广度优先展开思维树
    breadth: 每步生成多少个候选思路
    depth: 最大思考深度
    """

    def generate_thoughts(state: str, n: int) -> list[str]:
        """生成 n 个候选下一步思路"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"针对以下问题和当前推理状态，生成 {n} 个不同的下一步思路（JSON格式）：\n问题：{problem}\n当前状态：{state}"
            }],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)["thoughts"]

    def evaluate_thought(thought: str) -> float:
        """评估思路的前景分数（0-1）"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"评估以下思路解决问题的可能性，返回 0-1 的分数：\n问题：{problem}\n思路：{thought}"
            }]
        )
        return float(response.choices[0].message.content.strip())

    # BFS 展开
    current_level = [ThoughtNode(thought=problem, score=1.0, children=[])]

    for d in range(depth):
        next_level = []
        for node in current_level:
            thoughts = generate_thoughts(node.thought, breadth)
            for t in thoughts:
                score = evaluate_thought(t)
                child = ThoughtNode(thought=t, score=score, children=[], parent=node)
                node.children.append(child)
                next_level.append(child)

        # 保留评分最高的节点继续展开
        next_level.sort(key=lambda x: x.score, reverse=True)
        current_level = next_level[:breadth]

    # 返回评分最高的完整推理链
    best_node = max(current_level, key=lambda x: x.score)
    return best_node.thought
```

## Reflexion（自我反思与修正）

Reflexion 框架让 Agent 在执行失败后进行语言反思，将经验存储在记忆中，下次避免同样错误：

```python
class ReflexionAgent:
    def __init__(self):
        self.reflective_memory = []  # 反思记录

    def run_with_reflection(self, task: str, max_trials: int = 3) -> str:
        for trial in range(max_trials):
            # 将历史反思注入提示
            reflection_context = "\n".join(self.reflective_memory[-3:])

            result = self._attempt_task(task, prior_reflections=reflection_context)

            # 评估结果
            success, feedback = self._evaluate(task, result)

            if success:
                return result

            # 失败后反思
            reflection = self._reflect(task, result, feedback)
            self.reflective_memory.append(
                f"第 {trial+1} 次尝试失败。反思：{reflection}"
            )

        return "多次尝试后仍未成功完成任务"

    def _reflect(self, task: str, result: str, feedback: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content":
                f"""分析失败原因并提炼经验：
                任务：{task}
                结果：{result}
                反馈：{feedback}
                请用一句话总结下次应该如何改进。"""
            }]
        )
        return response.choices[0].message.content
```

## Subgoal 层级规划

将目标分解为多层次的子目标，形成层级结构：

```
顶层目标：完成产品市场分析报告
  ├── 子目标1：数据收集
  │     ├── 搜索行业数据
  │     ├── 整理竞品信息
  │     └── 收集用户反馈
  ├── 子目标2：数据分析
  │     ├── 趋势分析
  │     └── 竞争格局分析
  └── 子目标3：报告撰写
        ├── 撰写各章节
        └── 审校整合
```

层级规划的好处：
- 高层规划器只需关注子目标之间的依赖，不需要了解底层细节
- 每个子目标可以由专门的子 Agent 处理
- 便于人工在关键决策点介入审查

## LLM 作为规划器的局限性

LLM 规划存在固有缺陷：

- **计划不可靠**：LLM 可能生成看似合理但执行不可行的计划
- **常识错误**：对物理约束（时间、资源）的理解不准确
- **长链推理退化**：步骤越多，错误累积越显著
- **无法评估难度**：不知道某个子任务实际需要多少时间

## 与专用规划算法的结合

对于结构化领域，将 LLM 与传统规划算法结合：

- **PDDL（Planning Domain Definition Language）**：LLM 生成 PDDL 问题描述，传统规划求解器（如 Fast Downward）求解最优计划
- **MCTS（蒙特卡洛树搜索）**：用 MCTS 搜索动作序列，用 LLM 提供启发式评估函数
- **约束规划**：LLM 生成约束，专用求解器处理组合优化问题

这种混合方法结合了 LLM 的语言理解能力和传统算法的可靠性，在机器人规划、游戏 AI 等领域取得了良好效果。
