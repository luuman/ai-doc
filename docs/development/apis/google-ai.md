# Google AI API

Google 提供了从免费调试工具到企业级部署的完整 AI 服务体系，核心是 Gemini 系列模型。本指南介绍 Google AI Studio 和 Gemini API 的使用方法，并涵盖企业级的 Vertex AI 平台。

## Google AI Studio

Google AI Studio（[aistudio.google.com](https://aistudio.google.com)）是官方的免费调试平台：

- **免费使用**：提供免费额度（请求次数限制），无需信用卡即可开始
- **Prompt 设计器**：可视化测试各种提示策略
- **API Key 管理**：直接生成可用于 Gemini API 的密钥
- **代码导出**：将调试好的配置直接导出为 Python/JavaScript 代码
- **模型对比**：并排测试不同 Gemini 版本的输出

适合个人开发者和学习阶段的快速实验。

## Gemini API 基础使用

### 安装 SDK

```bash
pip install google-generativeai
```

### 文本生成

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-pro")

response = model.generate_content(
    "解释量子纠缠的基本原理",
    generation_config=genai.types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=2048,
        top_p=0.95,
    )
)

print(response.text)
print(f"Token 用量：{response.usage_metadata}")
```

## 主要模型版本

| 模型 | 上下文窗口 | 特点 |
|------|-----------|------|
| gemini-2.0-flash | 1M Token | 最新，速度快，多模态 |
| gemini-1.5-pro | 2M Token | 长上下文旗舰，均衡 |
| gemini-1.5-flash | 1M Token | 速度/成本最优 |
| gemini-1.5-flash-8b | 1M Token | 超轻量，极低成本 |

## 多模态输入

Gemini 原生支持文本、图像、视频、音频和 PDF 的混合输入：

### 图像输入

```python
import PIL.Image

model = genai.GenerativeModel("gemini-1.5-pro")
image = PIL.Image.open("architecture_diagram.png")

response = model.generate_content([
    image,
    "请分析这张系统架构图，识别潜在的单点故障"
])
print(response.text)
```

### 视频输入（通过 File API）

```python
# 上传视频文件
video_file = genai.upload_file("demo.mp4", mime_type="video/mp4")

# 等待处理完成
import time
while video_file.state.name == "PROCESSING":
    time.sleep(5)
    video_file = genai.get_file(video_file.name)

response = model.generate_content([
    video_file,
    "总结这个演示视频的主要内容，列出关键步骤"
])
```

### 音频输入

```python
audio_file = genai.upload_file("meeting_recording.mp3", mime_type="audio/mp3")

response = model.generate_content([
    audio_file,
    "请转录这段录音，并提取所有行动项"
])
```

### PDF 文档处理

```python
pdf_file = genai.upload_file("research_paper.pdf", mime_type="application/pdf")

response = model.generate_content([
    pdf_file,
    "总结这篇论文的主要贡献和实验结果"
])
```

## 长上下文窗口

Gemini 1.5 Pro 支持高达 200 万 Token 的上下文窗口，这意味着可以：

- 在一次请求中处理整个代码仓库（数百个文件）
- 分析几小时的视频内容
- 一次性摄入数千页 PDF
- 跨越极长对话历史进行精确引用

长上下文使用技巧：
- 将最重要的信息放在开头和结尾（"lost in the middle" 问题）
- 使用 `grounding` 确认模型确实引用了特定位置
- 长输入会显著增加 Token 成本，评估是否真正需要

## 系统指令

```python
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    system_instruction="""
    你是一个专业的代码审查助手。
    - 只关注代码质量、安全性和性能问题
    - 以结构化列表格式输出
    - 按严重程度排序：Critical > Warning > Suggestion
    """,
)
```

## Function Calling

```python
def get_stock_price(ticker: str) -> dict:
    """获取股票价格（示例函数）"""
    return {"ticker": ticker, "price": 188.5, "currency": "USD"}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    tools=[get_stock_price],  # 直接传入 Python 函数，自动推断 Schema
)

chat = model.start_chat(enable_automatic_function_calling=True)
response = chat.send_message("苹果公司（AAPL）现在股价是多少？")
print(response.text)  # 模型自动调用函数并整合结果
```

Gemini 的 Function Calling 支持自动执行模式（`enable_automatic_function_calling=True`），无需手动处理工具调用循环。

## Grounding（Google Search 接地）

将 Gemini 与 Google 搜索集成，获取实时信息并附带引用来源：

```python
model = genai.GenerativeModel("gemini-1.5-pro")

response = model.generate_content(
    "2024 年的 AI 领域重大进展是什么？",
    tools="google_search_retrieval",  # 启用搜索接地
)

print(response.text)
# 响应包含 grounding_metadata，含有搜索来源和支撑证据
for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
    print(f"来源：{chunk.web.uri}")
```

## Vertex AI（企业级）

Vertex AI 是 Google Cloud 的企业级 ML 平台，相比 Gemini API 提供：

- **IAM 权限管理**：与 GCP 账号体系集成，精细化访问控制
- **VPC 网络隔离**：数据不经过公网，满足合规要求
- **SLA 保障**：企业级服务等级协议
- **区域选择**：可指定数据处理区域（合规需求）
- **批量预测**：大规模离线推理
- **模型调优**：支持 Fine-tuning

```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="my-gcp-project", location="us-central1")

model = GenerativeModel("gemini-1.5-pro")
response = model.generate_content("Hello, Vertex AI!")
print(response.text)
```

## 免费额度与定价

### 免费额度（AI Studio，通过 Gemini API）

- Gemini 1.5 Flash：每分钟 15 次请求，每天 1500 次（免费层）
- 免费层数据可能用于模型改进（生产环境建议升级付费）

### 付费定价（参考，以官网为准）

| 模型 | Input（每百万 Token） | Output（每百万 Token） |
|------|----------------------|----------------------|
| Gemini 1.5 Pro（<128K） | $1.25 | $5 |
| Gemini 1.5 Pro（>128K） | $2.5 | $10 |
| Gemini 1.5 Flash | $0.075 | $0.3 |
| Gemini 1.5 Flash-8B | $0.0375 | $0.15 |

Gemini 1.5 Flash-8B 是目前最便宜的多模态 API 之一，适合高频率、低复杂度任务。
