# 提示工程最佳实践

掌握了基础技术之后，如何在实际项目中稳定、高效、低成本地运用提示工程，需要一套完整的工程实践体系。本章汇总了来自实战经验的最佳实践，涵盖常见错误规避、测试方法、成本控制和企业级管理。

## 常见错误

### 错误一：指令模糊

这是最常见也最容易修复的问题。

```
# 坏的 Prompt
总结这篇文章。

# 好的 Prompt
用 3 个要点总结以下文章，每个要点不超过 30 字。
目标读者：无技术背景的业务人员。
格式：使用 "• " 开头的项目符号列表。
```

诊断方法：如果你的 Prompt 可以被 10 种不同方式理解，那它就太模糊了。将 Prompt 给团队中不同背景的人看，观察他们的理解是否一致。

### 错误二：上下文过长导致遗忘

LLM 存在"上下文中间遗忘（Lost in the Middle）"现象：当上下文很长时，模型对中间部分的关注度显著下降，重要信息被"淹没"。

```
# 问题示例
[非常长的背景知识] + [核心任务] + [大量示例]

# 改进策略
1. 最重要的指令放在最前或最后（强调位置）
2. 将庞大上下文分块，通过 RAG 按需检索
3. 用 XML 标签标注重要内容：<important>关键指令</important>
4. 减少无关上下文，保持 Prompt 精练
```

### 错误三：格式不一致的 Few-shot 示例

```
# 格式不一致的示例（错误）
示例1：
输入：苹果
输出：水果

示例2 - 香蕉：
结果 -> 水果

第三个例子：葡萄 = 水果

# 格式完全一致的示例（正确）
示例1：
输入：苹果
输出：水果

示例2：
输入：香蕉
输出：水果

示例3：
输入：葡萄
输出：水果
```

### 错误四：忽略边界情况

只在"典型"案例上测试 Prompt，在边缘情况下崩溃：
- 空输入或极短输入
- 输入语言与预期不符（Prompt 是中文但用户发英文）
- 包含特殊字符、代码、数学公式的输入
- 超长输入（超过 Prompt 预期长度）

### 错误五：过度依赖单次测试

在一个案例上看起来完美的 Prompt，在实际多样化输入中可能表现差。始终在代表性的测试集（至少 20-50 个案例）上评估。

## 测试与迭代策略

### 构建测试集

高质量测试集是 Prompt 迭代的基础：

```
测试集组成建议：
- 正常案例（60%）：典型的、符合预期的输入
- 边界案例（20%）：极端情况、极短/极长输入、特殊字符
- 失败案例（20%）：你已知模型容易出错的场景
```

记录每个测试案例的：
- 输入（Input）
- 期望输出或期望标准（Expected）
- 各 Prompt 版本的实际输出（Actual）
- 是否通过（Pass/Fail）

### 单变量测试原则

每次只修改 Prompt 的一处，才能确定改动是否有效：

```
# 正确的迭代方式
版本1（基准）：你是助手，请回答问题。
版本2（测试角色）：你是专业的医学顾问，请回答问题。
         ↑ 只改了角色设定，其他不变

# 错误的迭代方式
版本1：你是助手，请回答问题。
版本2：你是专业的医学顾问，请用结构化格式、加入数据引用、以中文回答问题。
         ↑ 同时改了多处，不知道哪个改动真正有效
```

### 回归测试

修改 Prompt 后，不只测试新场景，还要确保原来通过的案例没有退化。自动化回归测试流程：

```python
def regression_test(new_prompt: str, test_cases: list, old_results: dict) -> dict:
    """确保新 Prompt 不会在已通过的案例上退化"""
    new_results = {}
    regressions = []

    for case in test_cases:
        result = evaluate(new_prompt, case["input"])
        new_results[case["id"]] = result

        # 检查是否有退化
        if old_results.get(case["id"], {}).get("passed") and not result["passed"]:
            regressions.append(case["id"])

    if regressions:
        print(f"警告：以下案例发生退化：{regressions}")

    return new_results
```

## Prompt 版本管理工具

### 轻量方案：Git + Markdown/YAML

```bash
# 目录结构
prompts/
  v1.0.0/
    customer_service.md
    code_review.md
  v1.1.0/
    customer_service.md  # 更新版本
  current -> v1.1.0/    # 软链接指向当前版本
```

### 专业工具

- **PromptLayer**：Prompt 版本管理 + 日志记录 + A/B 测试 + 成本追踪
- **LangSmith**：LangChain 生态，支持 Prompt Hub、追踪、评估
- **Weights & Biases Prompts**：与 ML 实验追踪集成

## 不同模型的差异

不同模型对 Prompt 的偏好存在显著差异，从一个模型迁移到另一个模型时需要重新测试和调整：

| 特性 | Claude（Anthropic）| GPT-4（OpenAI）| Gemini（Google）|
|------|-------------------|--------------|----------------|
| 指令遵循 | 极强，对约束很敏感 | 强 | 强 |
| XML 标签 | 优先推荐 | 效果好但非必须 | 效果一般 |
| 角色扮演 | 有明确安全边界 | 较为灵活 | 有安全边界 |
| 中文能力 | 良好 | 良好 | 良好 |
| 助手预填 | 支持（高效）| 不直接支持 | 不支持 |
| JSON 输出 | 需要在 Prompt 中指定 | JSON Mode 原生支持 | 有 JSON 模式 |
| 长上下文 | Claude 3.5：200K token | GPT-4o：128K token | Gemini 1.5：1M token |

**Claude 特有技巧**：
- 使用 XML 标签组织结构（`<context>`, `<task>`, `<rules>`），Claude 对 XML 格式特别响应
- 助手预填（Prefill）让模型从指定内容开始输出

**GPT 特有技巧**：
- 使用 JSON Mode 或 Structured Outputs API
- `response_format` 参数强制 JSON 输出

## 成本控制

Prompt Token 数量直接影响 API 成本，优化思路：

### 精简 Prompt

```
# 冗余版本（~80 token）
请你帮我把下面这段话翻译成英文，需要注意翻译要流畅自然，
符合英文的表达习惯，不要直接逐字翻译，要意译，保持原文的意思。

# 精简版本（~20 token）
将以下文字翻译为流畅自然的英文：
```

### 缓存常用 Prompt

- **OpenAI Prompt Caching**：前缀相同的长系统提示自动缓存，重复调用折扣 50% 成本
- **Anthropic Prompt Caching**：同样支持，对超过 1024 token 的可缓存内容降价

```python
# Anthropic 的 Prompt Cache 用法
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": very_long_system_context,  # 长系统上下文
                "cache_control": {"type": "ephemeral"}  # 标记为可缓存
            },
            {
                "type": "text",
                "text": user_question  # 每次变化的部分不缓存
            }
        ]
    }
]
```

## 中文提示的特殊考虑

### Token 效率差异

中文每个汉字通常对应 1-2 个 token，而英文单词可能是 1-3 个 token。对于信息密度高的中文指令，token 效率实际上更高。

### 避免中英混杂指令歧义

```
# 可能产生歧义的混杂写法
请用 JSON format 返回，包含 key: "name" 和 "age"

# 更清晰的写法（保持一致）
请以 JSON 格式返回，包含 "name" 和 "age" 两个字段
```

### 数字和日期格式

中文上下文中，明确指定格式避免歧义：
```
日期格式：使用 YYYY年MM月DD日（如 2025年02月28日）
数字格式：大数字使用中文计数（如 1.2亿，而非 120000000）
```

## 企业级 Prompt 管理

大型团队使用 LLM 时，需要系统化的 Prompt 管理：

### 集中化 Prompt 仓库

```
建立单一的 Prompt 仓库（Git Repo），包含：
- 所有生产 Prompt 的版本历史
- 每个 Prompt 的测试案例和期望指标
- Prompt 的负责人（Ownership）
- 变更审批流程（Code Review）
```

### A/B 测试 Prompt

```python
import random

def get_prompt_variant(user_id: str, experiment_name: str) -> str:
    """根据用户 ID 确定性地分配 Prompt 变体"""
    # 哈希用户 ID 确保同一用户总是看到同一变体
    hash_val = hash(f"{user_id}_{experiment_name}") % 100

    if hash_val < 50:  # 50% 流量
        return PROMPT_CONTROL  # 对照组
    else:
        return PROMPT_VARIANT  # 实验组

# 记录每次调用使用的变体，用于后续效果分析
```

### 监控与告警

生产环境中 Prompt 需要持续监控：
- **输出格式错误率**：JSON 解析失败率超过阈值时告警
- **平均 Token 消耗**：异常飙升时告警（可能是 Prompt 注入攻击）
- **用户满意度**：通过点赞/点踩收集反馈，关联到 Prompt 版本
- **延迟分布**：P95/P99 响应时间监控
