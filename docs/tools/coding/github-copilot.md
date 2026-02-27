# GitHub Copilot

GitHub Copilot 是 GitHub 与 OpenAI 联合开发的 AI 编程助手，于 2021 年发布，是目前全球用户数量最多的 AI 编程工具。作为 GitHub 生态的核心产品，Copilot 与版本控制、CI/CD、代码审查的深度整合是其独特优势。

## Copilot 核心功能

### Ghost Text（自动补全）

Copilot 最基础、最广泛使用的功能：在编辑器中实时显示 AI 生成的代码建议（灰色幽灵文字），按 Tab 接受：

```python
# 输入函数签名，Copilot 自动建议实现
def binary_search(arr: list, target: int) -> int:
    # Copilot 生成：
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

Ghost Text 的质量取决于：
- 文件开头的 import 语句（告诉 Copilot 使用的框架）
- 相邻函数的代码模式（Copilot 学习并模仿）
- 注释描述（详细的注释 → 更准确的补全）
- 函数名和参数名（语义化命名 → 更好的补全）

快捷键：
- `Tab`：接受补全
- `Esc`：拒绝补全
- `Alt+]` / `Alt+[`：切换多个候选建议
- `Ctrl+Enter`：打开 Copilot Panel（显示 10 个候选补全）

### Copilot Chat（IDE 内问答）

Copilot Chat 将对话式 AI 直接嵌入 VS Code 的侧边栏，无需切换到浏览器：

```
用户：这段 SQL 查询为什么很慢？
[选中复杂的 JOIN 查询]

Copilot Chat：这个查询慢的原因可能有几个：
1. products 表和 orders 表的 JOIN 没有在 product_id 列上建索引
2. WHERE 子句中的 LIKE '%keyword%' 无法使用索引（前导通配符）
3. 子查询每次都会全表扫描...

建议的优化：[提供具体 SQL 示例]
```

Copilot Chat 支持的内置命令（斜杠命令）：
- `/explain`：解释选中代码的工作原理
- `/fix`：修复选中代码中的问题
- `/tests`：为选中代码生成单元测试
- `/doc`：生成文档注释（JSDoc、docstring）
- `/simplify`：简化复杂代码
- `/optimize`：优化性能

### @workspace（全仓库理解）

`@workspace` 是 Copilot Chat 的上下文扩展命令，允许模型理解整个代码库：

```
@workspace 项目中用户认证是如何实现的？所有相关文件列出来

@workspace 找到所有未处理异常的异步函数

@workspace 我想在不破坏现有 API 的情况下添加速率限制，应该修改哪些文件？
```

`@workspace` 会自动检索代码库中最相关的文件提供给模型，类似于 Cursor 的 `@Codebase`，但实现相对简单。

## Copilot for CLI

Copilot CLI 将 AI 助手扩展到命令行，帮助生成和解释 Shell 命令：

```bash
# 安装 GitHub CLI 和 Copilot 扩展
gh extension install github/gh-copilot

# 解释命令
gh copilot explain "git rebase -i HEAD~5"
# 输出：这个命令会打开交互式 rebase，让你修改最近5次提交...

# 建议命令
gh copilot suggest "找到过去7天内修改过的所有 Python 文件"
# 输出：find . -name "*.py" -mtime -7

# 询问 Shell 操作
gh copilot suggest "如何查找并杀死占用 8080 端口的进程"
```

## Copilot for PR（Pull Request 功能）

这是 GitHub Copilot 区别于竞品的独特优势——深度集成 GitHub 的代码审查工作流：

### 自动生成 PR 描述

在 GitHub.com 创建 PR 时，Copilot 可以根据 Diff 自动生成：
- PR 标题（简洁描述变更）
- 变更摘要（按文件或功能模块分组）
- 测试说明（如何验证变更）

```markdown
# Copilot 自动生成的 PR 描述示例

## 变更摘要

### 新增功能
- 添加了用户邮箱验证功能（`src/auth/email_verify.py`）
- 添加了邮件发送服务集成（`src/services/email.py`）

### 修改
- 更新了用户注册流程，要求邮箱验证才能完成注册
- 添加了验证邮件模板（`templates/verify_email.html`）

## 如何测试
1. 注册新用户，检查收到验证邮件
2. 点击验证链接，确认账户激活
3. 尝试过期链接，验证报错
```

### 代码审查建议

Copilot 可以对 PR 中的每个文件生成自动代码审查建议：

- 发现潜在 Bug（空指针、类型错误、边界情况）
- 指出安全问题（SQL 注入、XSS、不安全的加密）
- 建议代码改进（性能、可读性、测试覆盖）

## 企业版特性

### 代码私有不训练

GitHub Copilot Business 和 Enterprise 的核心企业承诺：
- **代码不用于训练**：企业代码不会用于训练 Copilot 模型
- **Prompts 不保留**：代码建议请求在处理后立即丢弃
- **SOC 2 Type 2 合规**：安全合规认证

### 安全策略

- **内容过滤**：过滤与已知开源代码相同的建议（降低版权风险）
- **代码引用检测**：标记与已知开源代码相似的建议并显示来源
- **组织策略**：管理员可以限制可用功能、设置允许的模型

### 知识库（Knowledge Bases，Enterprise 专属）

Enterprise 版允许将公司内部文档（Confluence、内部 Wiki）作为 Copilot Chat 的上下文，使模型了解公司特定的约定和架构：

```
# 引用公司内部知识库
@kb-company-standards 我们的 API 命名约定是什么？
@kb-company-standards 创建新微服务的标准步骤是什么？
```

## 价格

| 计划 | 价格 | 适用场景 |
|------|------|---------|
| Individual | $10/月 | 个人开发者 |
| Business | $19/用户/月 | 团队（代码不训练 + 策略管理） |
| Enterprise | $39/用户/月 | 大型企业（知识库 + 安全审查 + 自定义） |
| 学生/教师 | 免费 | 认证学生和教师 |
| 开源维护者 | 免费 | 受邀的知名开源项目维护者 |

## 与 Cursor 对比

| 维度 | GitHub Copilot | Cursor |
|------|---------------|--------|
| 安装方式 | VS Code/JetBrains 插件 | 独立 IDE |
| 学习成本 | 几乎为零 | 需学习 @引用等新功能 |
| GitHub 集成 | 深度（PR 审查、CLI） | 无原生集成 |
| Agent 能力 | Copilot Edits（中） | Agent 模式（强） |
| 代码库理解 | @workspace（中） | @Codebase（强） |
| 模型选择 | GPT-4o 等（有限选择） | Claude/GPT-4o（灵活） |
| 企业功能 | 成熟（知识库、策略） | 较少 |
| 价格 | $10-39/月 | $20-40/月 |

## VS Code / JetBrains / Neovim 插件

### VS Code

通过 VS Code 扩展市场安装：搜索"GitHub Copilot"，安装后用 GitHub 账号登录：

```
扩展 ID: GitHub.copilot
扩展 ID: GitHub.copilot-chat（Chat 功能）
```

VS Code 中的 Copilot 快捷键：
- `Ctrl+I`：内联 Chat（类似 Cursor 的 Cmd+K）
- `Ctrl+Shift+I`：打开 Copilot Chat 面板
- `Ctrl+Enter`：显示多个补全建议

### JetBrains（IntelliJ IDEA、PyCharm 等）

在 JetBrains Marketplace 搜索"GitHub Copilot"，与 VS Code 版功能基本相同，UI 与 JetBrains 风格一致。JetBrains 的 Copilot Chat 支持 `/explain`、`/fix`、`/tests` 等相同命令。

### Neovim

通过 copilot.vim 或 copilot.lua 插件支持：

```lua
-- 使用 lazy.nvim 安装
{
    "zbirenbaum/copilot.lua",
    cmd = "Copilot",
    event = "InsertEnter",
    config = function()
        require("copilot").setup({
            suggestion = {
                auto_trigger = true,
                keymap = {
                    accept = "<Tab>",
                    dismiss = "<Esc>",
                }
            }
        })
    end
}
```

Neovim 版目前只支持 Ghost Text 补全，不支持 Chat 功能（需要通过 CopilotChat.nvim 等社区插件实现）。
