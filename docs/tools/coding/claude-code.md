# Claude Code

Claude Code 是 Anthropic 官方推出的命令行界面（CLI）AI 编程助手，将 Claude 模型的强大能力直接整合到终端工作流中。与基于 IDE 的工具（如 Cursor）不同，Claude Code 以终端为核心，能够执行系统命令、管理文件、操控 Git——像一位坐在你旁边的资深开发伙伴。

## Claude Code 定位

Claude Code 的核心定位是**终端原生的 AI 工程师**：

- **不是补全工具**：不做 Tab 自动补全，而是理解任务、规划步骤、执行动作
- **不是对话框**：能够直接在文件系统操作，而非只生成代码片段供用户复制粘贴
- **是 AI Agent**：给定一个任务，Claude Code 可以自主完成多步骤操作（读文件→分析→修改→测试→提交）

典型使用场景：
- "帮我重构这个模块，把所有回调函数改成 async/await"
- "找到所有 SQL 注入漏洞并修复"
- "理解这个陌生代码库的架构"
- "帮我写单元测试直到覆盖率达到 80%"

## 核心功能

### 代码生成

Claude Code 可以根据自然语言描述生成完整的代码文件：

```bash
# 交互式会话
claude

# 在会话中
> 帮我用 FastAPI 创建一个 JWT 认证的用户管理 API，包含注册、登录、获取用户信息接口

# Claude Code 会：
# 1. 规划需要创建的文件结构
# 2. 生成 main.py、auth.py、models.py 等文件
# 3. 安装必要依赖（pip install fastapi python-jose[cryptography]）
# 4. 提供启动命令
```

### 代码调试

```bash
> 这个函数运行时报 KeyError，帮我找原因

# Claude Code 会：
# 1. 读取相关文件
# 2. 分析错误堆栈
# 3. 定位问题根因
# 4. 提出修复方案并实施
```

### 代码重构

```bash
> 把 src/utils/ 目录下所有函数的命名从驼峰改为下划线风格，并更新所有引用

# Claude Code 自主完成：跨文件搜索→重命名→更新引用→验证不破坏测试
```

### 代码解释

```bash
# 非交互模式，快速解释
claude -p "解释这段代码的作用" < complex_algorithm.py
```

### 终端操作

Claude Code 可以执行系统命令（需用户确认）：

```bash
# Claude Code 可以执行的操作示例
git log --oneline -20     # 查看提交历史
npm test                  # 运行测试
docker ps                 # 查看容器
grep -r "TODO" src/       # 搜索代码
```

## 与 IDE 集成

### VS Code 集成

在 VS Code 终端中使用 Claude Code，可以实现与编辑器的无缝协作：

```bash
# 在 VS Code 终端中启动
claude

# 打开当前工作区的文件进行分析
> 分析当前项目的依赖关系图
```

VS Code 扩展（社区开发）可以在侧边栏启动 Claude Code 会话，直接对编辑器中选中的代码进行操作。

### JetBrains 集成

通过 JetBrains 内置终端运行 Claude Code，可以结合 IntelliJ 的代码导航能力：
- 在 IDE 中定位到复杂代码段
- 切换到终端，用 Claude Code 分析和修改
- 切换回 IDE 查看修改效果

## MCP（Model Context Protocol）生态系统

MCP 是 Anthropic 提出的开放标准，允许 Claude Code 通过标准化接口调用外部工具和数据源：

```json
// ~/.claude/claude_desktop_config.json（MCP 配置）
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_TOKEN": "ghp_xxx"}
        },
        "postgres": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
        }
    }
}
```

官方 MCP Server 生态：
- **Filesystem**：文件系统操作（读写、搜索）
- **GitHub**：PR、Issue、代码搜索
- **PostgreSQL / SQLite**：数据库查询
- **Brave Search**：实时网络搜索
- **Puppeteer**：浏览器自动化
- **Slack / Linear / Jira**：项目管理工具集成

社区 MCP Server 已超过 1000 个，覆盖各种 API 和工具。

## Hooks 机制（自定义工作流）

Claude Code 的 hooks 机制允许在特定事件发生时执行自定义脚本，实现自动化工作流：

```json
// .claude/settings.json
{
    "hooks": {
        "pre-tool-call": [
            {
                "command": "echo '准备执行工具调用' >> .claude/audit.log"
            }
        ],
        "post-tool-call": [
            {
                "command": "/path/to/speak-pipe",
                "input": "工具调用已完成"
            }
        ],
        "stop": [
            {
                "command": "git diff --stat"
            }
        ]
    }
}
```

Hooks 事件类型：
- `pre-tool-call`：工具调用前触发（可用于审计、确认）
- `post-tool-call`：工具调用后触发（可用于 TTS 通知、日志）
- `stop`：会话结束时触发（可用于自动提交、清理）

## CLAUDE.md 项目配置文件

每个项目根目录下的 `CLAUDE.md` 文件为 Claude Code 提供项目特定的上下文和指导：

```markdown
# CLAUDE.md

## 项目概述
这是一个 FastAPI 后端服务，使用 PostgreSQL + Redis，部署在 K8s 上。

## 开发命令
- 启动：`uvicorn main:app --reload`
- 测试：`pytest tests/ -v --cov=src`
- 代码格式：`black . && isort .`

## 关键约定
- 所有数据库操作必须通过 Repository 层
- API 端点返回值必须使用 Pydantic 模型
- 禁止在 main.py 中包含业务逻辑
- 提交消息格式：`type(scope): description`（中文）

## 不要修改的文件
- migrations/（数据库迁移文件，手动管理）
- .env.production（生产环境配置）
```

Claude Code 在每次会话开始时自动读取 CLAUDE.md，无需用户每次重复说明项目背景。

## Slash 命令

Claude Code 内置的斜杠命令提供快捷操作：

- `/compact`：压缩当前会话历史，节省 Context 窗口空间
- `/clear`：清除会话历史，开始新任务
- `/review`：对当前 Git 变更进行代码审查
- `/init`：为当前项目生成 CLAUDE.md 文件
- `/cost`：显示当前会话的 Token 消耗和成本估算
- `/model`：切换使用的模型（claude-opus-4-5 / claude-sonnet-4-5 / claude-haiku-3-5）
- `/help`：显示帮助信息

## 成本控制

Claude Code 按 Token 计费，合理使用可以控制成本：

- 使用 `/compact` 定期压缩历史，避免 Context 过长
- 对于简单任务，使用 `/model` 切换到 claude-haiku-3-5
- 使用 `--print` 模式（非交互式）处理批量任务
- 在 CLAUDE.md 中明确项目范围，避免模型读取不必要的文件

```bash
# 非交互模式，适合批处理
claude -p "为 src/utils/string_helpers.py 生成单元测试" --model claude-haiku-3-5
```

## 与 Cursor 的根本差异

| 维度 | Claude Code | Cursor |
|------|------------|--------|
| 界面 | 终端 CLI | GUI（VS Code 界面） |
| 工作方式 | 命令式（任务驱动） | 交互式（辅助开发） |
| 文件操作 | 直接执行（自主） | 建议修改（用户确认） |
| 代码补全 | 无 | 有（Tab 补全） |
| 适合场景 | 复杂任务、代码库重构 | 日常编码辅助 |
| 上下文 | 整个代码库 | 当前文件为主 |
| 系统操作 | 支持（终端命令） | 受限 |

## 最佳实践

- **给出清晰的任务描述**：包含目标、约束、验收标准，避免模糊的"帮我优化这段代码"
- **利用 CLAUDE.md 减少重复**：将项目规范、禁止事项、关键约定放在 CLAUDE.md 中
- **分解复杂任务**：将大任务拆分为小步骤，每步确认再继续
- **配合 Git**：让 Claude Code 在修改前先提交（或 stash），方便回滚
- **启用 hooks 自动化**：将重复性后处理（格式化、测试、提交）配置为 hooks 自动执行
- **批量处理用管道**：`find . -name "*.py" | claude -p "统计这些文件的总行数"`
