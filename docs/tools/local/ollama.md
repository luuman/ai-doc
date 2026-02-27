# Ollama

Ollama 是目前最流行的本地 LLM 运行工具，将在本地运行大语言模型的复杂过程简化为几条命令。从模型下载、格式转换、服务启动到 API 调用，Ollama 提供了完整的一站式解决方案，无需 Python 环境、CUDA 配置或任何机器学习背景知识。

## Ollama 定位

Ollama 的设计目标是**让大模型的本地运行像使用 Docker 一样简单**：

- 一条命令拉取并运行模型（`ollama run llama3.1:8b`）
- 自动检测硬件并选择最优后端（NVIDIA GPU / AMD GPU / Apple Silicon Metal / CPU）
- 提供 OpenAI 兼容的 REST API
- 管理多个模型（下载、列出、删除）

底层依赖 llama.cpp 作为推理引擎，因此支持 GGUF 格式的量化模型，能够在 CPU 上运行，并对 Apple Silicon 有优化的 Metal 加速。

## 安装（Mac/Linux/Windows）

### macOS

```bash
# 方法1：官网下载安装包（推荐）
# 访问 ollama.ai/download，下载 macOS App，拖到 Applications

# 方法2：命令行安装
curl -fsSL https://ollama.ai/install.sh | sh

# 验证安装
ollama --version
```

安装后，Ollama 以菜单栏图标的形式运行，开机自启。

### Linux

```bash
# 一键安装
curl -fsSL https://ollama.ai/install.sh | sh

# 验证服务运行状态
systemctl status ollama

# 查看日志
journalctl -u ollama -f
```

Linux 安装后 Ollama 作为 systemd 服务运行，监听 `127.0.0.1:11434`。

### Windows

访问 ollama.ai/download 下载 Windows 安装包（.exe），双击安装，会在系统托盘显示图标。

## 核心命令

### ollama pull（下载模型）

```bash
# 下载模型（自动选择最合适的量化版本）
ollama pull llama3.1:8b          # LLaMA 3.1 8B（推荐通用）
ollama pull llama3.1:70b         # 需要大内存（约 40GB）
ollama pull qwen2.5:7b           # 通义千问 2.5（中文能力强）
ollama pull qwen2.5:72b          # 最强中文版（需要 40GB+）
ollama pull deepseek-r1:8b       # DeepSeek R1 8B（推理能力强）
ollama pull phi3.5               # Microsoft Phi-3.5（小模型高性能）
ollama pull gemma2:9b            # Google Gemma 2（英文能力强）
ollama pull nomic-embed-text     # 文本嵌入模型（用于 RAG）
```

### ollama run（运行模型）

```bash
# 交互式对话
ollama run llama3.1:8b

# 直接运行单次查询（非交互）
ollama run llama3.1:8b "用中文解释什么是量子纠缠"

# 管道输入
echo "翻译成英文：你好世界" | ollama run qwen2.5:7b

# 从文件输入
ollama run llama3.1:8b < my_question.txt
```

交互式会话命令：
- `/help`：显示帮助
- `/bye`：退出会话
- `/show modelfile`：显示当前模型的 Modelfile
- 多行输入：用 `"""` 包裹，`"""` 结束后发送

### ollama list（列出模型）

```bash
ollama list
# 输出：
# NAME                    ID              SIZE    MODIFIED
# llama3.1:8b             365c0bd3c000    4.7 GB  2 minutes ago
# qwen2.5:7b              845dbda0ea48    4.7 GB  1 hour ago
# nomic-embed-text:latest 0a109f422b47    274 MB  3 hours ago
```

### ollama rm（删除模型）

```bash
ollama rm llama3.1:8b
```

### ollama ps（查看运行中的模型）

```bash
ollama ps
# NAME            ID              SIZE      PROCESSOR    UNTIL
# llama3.1:8b     365c0bd3c000    5.9 GB    GPU 100%     4 minutes from now
```

## 主流模型支持

Ollama 官方支持的主要模型家族：

### Llama（Meta）

```bash
ollama pull llama3.2:1b    # 1B，最轻量，手机/弱机
ollama pull llama3.2:3b    # 3B，平衡
ollama pull llama3.1:8b    # 8B，通用推荐
ollama pull llama3.1:70b   # 70B，高质量
ollama pull llama3.1:405b  # 405B，媲美 GPT-4
```

### Qwen（阿里通义千问）

中文能力最强的开源模型系列：

```bash
ollama pull qwen2.5:0.5b   # 0.5B，极轻量
ollama pull qwen2.5:7b     # 7B，中文场景推荐
ollama pull qwen2.5:14b    # 14B，更强推理
ollama pull qwen2.5:72b    # 72B，最强中文
ollama pull qwen2.5-coder:7b  # 代码专项
```

### Phi（Microsoft）

高性价比小模型：

```bash
ollama pull phi3.5         # 3.8B，性能超出体积
ollama pull phi4           # 14B，强推理
```

### Gemma（Google）

```bash
ollama pull gemma2:9b      # 9B
ollama pull gemma2:27b     # 27B
```

### DeepSeek

```bash
ollama pull deepseek-r1:8b   # R1 蒸馏版，强推理
ollama pull deepseek-r1:32b  # 更强
ollama pull deepseek-v3      # DeepSeek V3（需要大内存）
```

## OpenAI 兼容 API 服务

Ollama 内置 OpenAI 兼容 REST API，无需额外配置：

```bash
# 启动服务（默认监听 localhost:11434）
ollama serve

# 如果要外部访问，设置环境变量
OLLAMA_HOST=0.0.0.0 ollama serve
```

API 调用示例：

```python
from openai import OpenAI

# 将 base_url 指向 Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama 不需要真实 key，传任意字符串
)

response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "解释一下 RAG 技术"}
    ]
)

print(response.choices[0].message.content)
```

嵌入向量（Embedding）API：

```python
# 用于 RAG 的文本嵌入
response = client.embeddings.create(
    model="nomic-embed-text",
    input="这是要嵌入的文本"
)
embedding_vector = response.data[0].embedding  # 768维向量
```

## Modelfile 自定义

Modelfile 是 Ollama 模型的配置文件，类似 Dockerfile，可以基于现有模型创建自定义版本：

```dockerfile
# ~/my-assistant/Modelfile

# 基础模型
FROM qwen2.5:7b

# 系统提示（角色设定）
SYSTEM """
你是一个专业的 Python 编程助手，名叫 PyHelper。
你只回答 Python 相关的问题。
回答要简洁、代码优先。
对话使用中文。
"""

# 模型参数
PARAMETER temperature 0.3        # 降低创造性，提高确定性
PARAMETER top_p 0.9
PARAMETER num_ctx 8192           # 上下文窗口大小
PARAMETER num_predict 2048       # 最大输出 token

# 消息模板（可选，覆盖默认格式）
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}{{ .Response }}<|im_end|>
"""
```

```bash
# 从 Modelfile 创建自定义模型
ollama create py-helper -f ~/my-assistant/Modelfile

# 运行自定义模型
ollama run py-helper
```

## 性能配置

### GPU 使用配置

```bash
# 指定使用第几张 GPU（多 GPU 机器）
CUDA_VISIBLE_DEVICES=0 ollama serve

# 限制 GPU 内存使用（默认自动检测）
# Ollama 会尽量将模型层加载到 GPU，超出部分卸载到 CPU
```

### 并发配置

```bash
# 最大并行请求数（默认 1，CPU 推理时建议不超过 4）
OLLAMA_NUM_PARALLEL=2 ollama serve

# 最大加载模型数（同时在内存中保持的模型）
OLLAMA_MAX_LOADED_MODELS=2 ollama serve

# 模型空闲超时后卸载（节省内存，秒数）
OLLAMA_KEEP_ALIVE=300 ollama serve  # 5分钟后卸载
OLLAMA_KEEP_ALIVE=-1 ollama serve   # 永不卸载
```

### Flash Attention

```bash
# 启用 Flash Attention（减少显存，提升长上下文速度）
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

## 与 LM Studio 对比

| 维度 | Ollama | LM Studio |
|------|--------|----------|
| 界面 | CLI + API | GUI + API |
| 易用性 | 低（需要命令行） | 高（图形界面） |
| 适合用户 | 开发者 | 非技术用户 |
| 模型来源 | Ollama 官方 Hub | HuggingFace Hub |
| 格式支持 | GGUF（主）| GGUF（主）|
| API 兼容性 | OpenAI 完全兼容 | OpenAI 兼容 |
| 自定义 Modelfile | 支持 | 不支持 |
| 脚本集成 | 好 | 较难 |
| 性能 | 相当 | 相当 |
| 跨平台 | Mac/Linux/Windows | Mac/Windows |

## 生产场景使用

Ollama 虽然定位于本地开发，但在特定场景下也可用于生产：

- **内部工具**：企业内网部署，敏感数据不出网络
- **离线场景**：工厂、医院、军事环境等无外网场景
- **低流量 API**：对并发要求不高（< 10 QPS）的内部服务

不适合高并发生产场景（> 10 QPS 建议使用 vLLM）。如果需要在内网高并发服务，可以在 Ollama 前面加 Nginx 负载均衡，运行多个 Ollama 实例。

```bash
# 运行多个 Ollama 实例（不同端口）
OLLAMA_HOST=0.0.0.0:11434 ollama serve &
OLLAMA_HOST=0.0.0.0:11435 ollama serve &
OLLAMA_HOST=0.0.0.0:11436 ollama serve &
```

```nginx
upstream ollama {
    server 127.0.0.1:11434;
    server 127.0.0.1:11435;
    server 127.0.0.1:11436;
}
```
