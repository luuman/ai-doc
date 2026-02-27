# Cursor IDE

Cursor 是由 Anysphere 开发的 AI 增强代码编辑器，基于 VS Code 深度定制。它在保留 VS Code 完整生态（扩展、主题、快捷键）的基础上，将 AI 能力深度嵌入编码工作流的每个环节。自 2023 年推出以来，Cursor 迅速成为硅谷工程师和 AI 工程师最喜爱的开发工具之一。

## Cursor 基于 VS Code 的 AI 增强

Cursor 的技术路线是对 VS Code 进行 Fork，而非作为 VS Code 插件运行。这使得 Cursor 能够：

- 深度修改编辑器的底层行为（如 Tab 键劫持用于 AI 补全）
- 构建私有的代码索引系统（全仓库向量检索）
- 提供无法作为插件实现的 Agent 能力（多文件自主修改）

VS Code 用户可以直接将设置和扩展迁移到 Cursor：`Cursor Settings → Import VS Code Settings`，几乎零学习成本切换。

## Tab 补全（上下文感知的多行补全）

Cursor 的 Tab 补全（Ghost Text）是其最直观的功能，但与 GitHub Copilot 的单行补全不同，Cursor 实现了真正的**多行上下文感知补全**：

**跨文件上下文**：Cursor 分析当前打开的文件和整个代码库的模式，而非仅仅基于当前文件。如果你在 `user.service.ts` 中写了类似的函数，`product.service.ts` 中会自动推断出相同的模式。

**Diff 感知**：Cursor 跟踪你最近的编辑，预测下一步最可能的操作。修改了函数签名后，它会自动建议更新所有调用点。

**光标位置推理**：不仅根据光标前的内容补全，还根据光标后的代码推断意图（中间填充，Fill-In-the-Middle）。

```python
# 示例：写了函数签名，Cursor 自动补全整个实现
def validate_email(email: str) -> bool:
    # Cursor 在此生成完整实现（正则验证、边界情况处理）
    [Tab 接受补全]
```

Tab 补全的快捷键：
- `Tab`：接受完整建议
- `Ctrl/Cmd + →`：逐词接受
- `Esc`：拒绝建议

## Cmd+K 内联编辑

`Cmd+K`（Mac）或 `Ctrl+K`（Windows/Linux）是 Cursor 的内联编辑命令，可以对选中的代码片段进行自然语言指令修改：

```
# 选中一段函数，按 Cmd+K，输入：
"添加输入验证和错误处理"
"翻译注释为中文"
"将这个 for 循环改为列表推导式"
"添加类型注解"
"这段代码有什么 bug？帮我修复"
```

Cursor 会展示一个 Diff 视图，显示原始代码和修改后的代码，用户可以：
- `Ctrl+Y`：接受所有修改
- `Ctrl+N`：拒绝所有修改
- 逐行选择接受或拒绝

Cmd+K 与 Chat 的区别：Cmd+K 适合**针对特定代码段的快速修改**，Chat 适合**需要上下文讨论的复杂问题**。

## Chat 面板（@引用系统）

Cursor 的 Chat 面板（`Ctrl+L`）支持强大的 `@` 引用语法，可以精确控制提供给 AI 的上下文：

### @文件引用

```
@src/auth/jwt.service.ts 这个 JWT 验证服务有什么安全问题？
@package.json 帮我升级所有过期的依赖，保持兼容性
```

### @代码库引用（@Codebase）

```
@Codebase 项目中所有 API 端点的鉴权方式一致吗？
@Codebase 找到所有没有错误处理的异步函数
```

`@Codebase` 触发 Cursor 对代码库的向量检索，找到最相关的代码片段提供给模型。

### @Web 引用

```
@Web React 18 的 use() Hook 怎么用？帮我改造这段代码
@Web 最新的 FastAPI 最佳实践是什么？
```

`@Web` 触发实时网络搜索，获取最新信息（突破模型知识截止日期限制）。

### @文档引用

```
@Docs shadcn/ui 的 Button 组件有哪些变体？
```

可以预先索引项目依赖的文档，直接在 Chat 中引用。

### @代码引用

```
@MyClass.processData 帮我为这个方法写 JSDoc 注释
```

## Agent 模式（自主多文件修改）

Agent 模式是 Cursor 最强大的功能，允许 AI 自主完成跨多文件的复杂任务：

启动方式：在 Chat 中使用 `cursor agent` 命令，或在 Chat 设置中切换到 Agent 模式。

Agent 的能力：
- **读取文件**：自主决定需要读取哪些文件
- **修改文件**：创建、编辑、删除文件
- **执行命令**：运行终端命令（npm install、pytest 等）
- **迭代修复**：运行测试，根据失败结果自动修复，直到通过

```
# Agent 模式示例任务
"我需要为这个 Express 项目添加 Redis 缓存层。请：
1. 安装必要依赖
2. 创建 Redis 连接配置
3. 为 GET /api/users 接口添加 5 分钟缓存
4. 添加缓存清除机制（POST/PUT/DELETE 操作时）
5. 确保测试通过"

# Agent 会自主完成所有步骤，中途询问关键决策点
```

## Cursor Rules（.cursorrules 项目规范）

`.cursorrules` 文件（新版改为 `.cursor/rules/`）定义项目特定的 AI 行为规范：

```markdown
# .cursorrules

## 技术栈
- 后端：Python 3.12 + FastAPI + SQLAlchemy 2.0
- 数据库：PostgreSQL 16
- 测试：pytest + pytest-asyncio

## 代码规范
- 所有异步函数必须使用 `async/await`
- 数据库操作必须通过 Repository 模式（不允许在 Router 中直接操作 DB）
- 所有外部 API 调用必须包含超时设置（timeout=30）
- 错误处理：使用自定义异常类，统一通过 exception_handler 返回错误

## 命名约定
- 文件名：snake_case（如 user_service.py）
- 类名：PascalCase
- 函数/变量：snake_case
- 常量：UPPER_SNAKE_CASE

## 测试要求
- 每个新函数必须有对应的单元测试
- 测试文件放在 tests/ 目录，与 src/ 结构镜像
- 使用 pytest.mark.asyncio 标注异步测试
```

Cursor Rules 让 AI 了解项目的技术决策，避免每次都需要在 Chat 中重复说明约束。

## 模型选择

Cursor 支持多个模型后端，用户可以根据任务选择：

- **Claude claude-opus-4-5**：最强代码理解和生成，但速度较慢
- **Claude Sonnet**：速度和质量的平衡，日常编码推荐
- **GPT-4o**：另一个强模型选择
- **cursor-small**：Cursor 自训练的轻量补全模型，Tab 补全默认使用

模型切换：`Settings → Models` 或在 Chat 面板底部选择。

## 价格（2024）

| 计划 | 价格 | 主要内容 |
|------|------|---------|
| Free | 免费 | 每月 2000 次补全，50 次 Chat（慢速） |
| Pro | $20/月 | 无限补全，500 次 premium Chat（快速） |
| Business | $40/用户/月 | Pro + 隐私模式 + 管理控制台 + SSO |

注意：Cursor 的定价竞争激进，实际价格以官网为准。

## 与 GitHub Copilot 对比

| 维度 | Cursor | GitHub Copilot |
|------|--------|---------------|
| 整合方式 | 独立 IDE（VS Code Fork） | VS Code/JetBrains 插件 |
| 多文件编辑 | Agent 模式（强） | Workspace（弱） |
| 代码库理解 | @Codebase 向量检索 | @workspace 基本理解 |
| Chat 上下文 | @引用精确控制 | 自动上下文 |
| 内联编辑 | Cmd+K（强大） | Copilot Edit（较弱） |
| 定价 | $20/月（Pro） | $10/月（Individual） |
| 隐私 | Business 版隐私模式 | Enterprise 版隐私配置 |
| 学习曲线 | 需学习 @引用等新概念 | 零学习成本（VS Code 插件） |

选择建议：
- 如果日常在 GitHub 生态工作、需要低价入门：选 GitHub Copilot
- 如果追求最强的 AI 辅助编程体验、不介意换 IDE：选 Cursor
- 如果在企业环境、需要团队统一管理：考虑 Copilot Business 或 Cursor Business
