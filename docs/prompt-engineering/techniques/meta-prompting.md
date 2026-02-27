# 元提示（Meta-Prompting）

元提示（Meta-Prompting）是指用 LLM 来生成、优化或评估其他 Prompt 的技术。在这种范式中，AI 不仅执行任务，还参与"如何更好地完成任务"的设计过程。元提示是提示工程从手工艺术向系统化科学演进的重要一步。

## 元提示定义

广义的元提示包含以下几种场景：

- **Prompt 生成**：给定任务目标，让 LLM 自动生成完成该任务的最佳 Prompt
- **Prompt 优化**：给定一个 Prompt 和其在测试集上的效果，让 LLM 改进这个 Prompt
- **自我评估**：让模型评价自己的输出质量，并根据评估结果修正
- **Prompt 压缩**：让模型在保留核心语义的前提下，精简过长的 Prompt

## APE：自动提示工程师

**APE（Automatic Prompt Engineer，Zhou et al., 2022）** 是第一个系统化研究自动 Prompt 生成的工作：

### 核心流程

```
阶段1：生成候选 Prompt
  输入：少量 (输入, 输出) 示例对
  指令给 LLM："以下是某任务的输入输出示例，请推断该任务的描述，
                并生成 10 种不同的指令 Prompt 来完成该任务"
  输出：N 个候选 Prompt

阶段2：评估候选 Prompt
  对每个候选 Prompt，在验证集上运行，计算准确率

阶段3：选择最优
  选取验证集准确率最高的 Prompt 作为最终使用的 Prompt

（可选）阶段4：迭代优化
  对最优 Prompt 进行变体采样，重复阶段2-3
```

### 实践示例

```python
# 让 GPT-4 为情感分析任务生成 Prompt
meta_prompt = """
以下是一个情感分析任务的输入输出示例：

输入："这家餐厅的服务太差了，再也不来了"
输出：负面

输入："食物很美味，环境也不错，下次还会来"
输出：正面

请生成 5 种不同风格的指令，用于指导 AI 完成该情感分析任务。
每个指令应该清晰、完整，包含输出格式要求。
以 JSON 数组格式返回，每个元素是一个字符串（指令内容）。
"""
```

## 让模型生成自己的思维链

一种特别有效的元提示技巧：先让模型生成"解决该类问题的最佳推理步骤"，再将这个推理框架应用于实际问题。

```python
# 步骤1：让模型生成推理框架
framework_prompt = """
你需要分析商业案例并给出战略建议。
在开始具体分析之前，请先列出分析此类商业问题的最佳思考步骤和框架。
思考过程应该是系统化、全面的，能适用于各类商业战略问题。
"""

framework = call_llm(framework_prompt)
# 框架示例输出：
# 1. 明确问题边界和目标
# 2. 分析市场环境（PESTEL）
# 3. 评估竞争格局（波特五力）
# ...

# 步骤2：使用该框架分析实际问题
analysis_prompt = f"""
请按照以下思考框架分析给定的商业问题：

{framework}

待分析的商业问题：
[具体的商业案例]
"""
```

## Prompt 版本管理

随着产品迭代，Prompt 需要像代码一样进行版本管理：

### 版本控制原则

```yaml
# prompts/v2.1.0/customer_service.yaml
version: "2.1.0"
last_updated: "2025-02-15"
author: "AI Team"
description: "客服助手系统提示，改进了多语言支持"
changelog:
  - "v2.1.0: 增加粤语支持，优化情绪安抚策略"
  - "v2.0.0: 重写角色设定，提升专业度"
  - "v1.0.0: 初始版本"

system_prompt: |
  你是一位专业的客服助手...

test_cases:
  - input: "我的订单为什么还没到？"
    expected_keywords: ["查询", "物流", "联系"]
  - input: "我要退款！"
    expected_keywords: ["理解", "退款流程", "时间"]
```

### 工具支持

- **Git + YAML**：最轻量的方案，将 Prompt 作为代码存储在 Git 仓库
- **PromptLayer**：专业的 Prompt 管理平台，支持版本对比、A/B 测试、成本追踪
- **LangSmith**：LangChain 官方平台，支持 Prompt Hub 和实验追踪

## 基于反馈的自动优化

将模型的输出反馈用于迭代优化 Prompt，形成自动改进循环：

```python
def auto_optimize_prompt(initial_prompt: str, test_cases: list, iterations: int = 3):
    current_prompt = initial_prompt

    for i in range(iterations):
        # 在测试集上评估当前 Prompt
        results = evaluate_prompt(current_prompt, test_cases)
        failures = [r for r in results if not r["passed"]]

        if not failures:
            print(f"第 {i+1} 轮：所有测试通过！")
            break

        # 让 LLM 分析失败案例并改进 Prompt
        optimization_prompt = f"""
        当前 Prompt：
        {current_prompt}

        以下测试案例失败（模型输出不符合期望）：
        {format_failures(failures)}

        请分析失败原因，并提供一个改进后的 Prompt 版本。
        改进版本应该：
        1. 修复导致上述失败的问题
        2. 保持对其他案例的有效性
        3. 尽可能简洁

        只返回改进后的 Prompt 内容，不要任何解释。
        """

        current_prompt = call_llm(optimization_prompt)
        print(f"第 {i+1} 轮优化完成，失败案例数：{len(failures)}")

    return current_prompt
```

## DSPy：声明式优化框架

**DSPy（Declarative Self-improving Python，Stanford，2023）** 是最系统化的自动 Prompt 优化框架，代表了提示工程的未来方向：

### 核心理念

DSPy 将 LLM 应用开发分离为：
- **程序结构**（你来定义）：数据流、模块化步骤、输入输出规范
- **Prompt 细节**（DSPy 自动优化）：具体的指令措辞、示例选择、CoT 策略

```python
import dspy

# 1. 定义输入输出签名（不写具体Prompt！）
class SentimentAnalysis(dspy.Signature):
    """分析产品评论的情感倾向"""
    review: str = dspy.InputField(desc="产品评论文本")
    sentiment: str = dspy.OutputField(desc="positive/negative/neutral")
    reasoning: str = dspy.OutputField(desc="判断理由")

# 2. 定义模块（选择推理策略）
class SentimentModule(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought(SentimentAnalysis)

    def forward(self, review):
        return self.analyze(review=review)

# 3. 定义评估指标
def sentiment_metric(example, pred, trace=None):
    return example.sentiment == pred.sentiment

# 4. 自动优化（DSPy 自动搜索最佳Prompt和示例）
optimizer = dspy.BootstrapFewShot(metric=sentiment_metric)
optimized_module = optimizer.compile(
    SentimentModule(),
    trainset=training_examples  # 少量标注示例
)
```

DSPy 会自动探索不同的 Prompt 措辞、CoT 策略和示例组合，找到在验证集上表现最佳的配置。

## 提示优化的评估指标

评估 Prompt 优化效果需要量化指标：

- **任务准确率**：最直接的指标，在标注测试集上的正确率
- **一致性**：多次运行相同输入，输出的稳定性（温度为 0 时应100%一致）
- **格式遵从率**：输出符合要求格式的比例（如 JSON 解析成功率）
- **Token 效率**：完成任务所需的平均 token 数（影响成本）
- **LLM-as-Judge**：用另一个强模型评估输出质量（无需人工标注）

## 实际应用案例

### 案例：自动优化客服 Prompt

某电商平台将客服 Prompt 优化时间从每周手工调整 8 小时降至自动化批次运行 1 小时，通过以下流程：

1. 每天收集人工质检不通过的客服对话（自动失败案例）
2. 夜间批次运行 DSPy 优化器，在失败案例上自动改进 Prompt
3. 新 Prompt 在灰度流量上 A/B 测试 24 小时
4. 若指标改善则自动推送，否则报警人工审核

这种方式将客服满意度稳步提升了 12%，同时减少了人工 Prompt 调整成本。
