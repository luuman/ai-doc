# Agent 记忆设计

记忆是 Agent 能够在时间维度上积累知识、保持上下文连贯性的关键能力。与人类记忆类似，Agent 的记忆系统也分为不同层次和类型，各自服务于不同的时间尺度和用途。

## 记忆类型框架

参考认知科学的记忆分类，Agent 记忆可分为三个层次：

### 感知记忆（Sensory Memory）

- 时间跨度：当前输入
- 实现方式：原始输入的即时处理（图像、文本、音频）
- 对应组件：多模态输入处理层

### 短期记忆（Short-term Memory）

- 时间跨度：当前会话（分钟到小时）
- 实现方式：LLM 的上下文窗口（In-Context Storage）
- 容量：受模型上下文窗口限制（8K-200K Token）
- 特点：快速读写，但会话结束后消失

### 长期记忆（Long-term Memory）

- 时间跨度：跨会话（天到永久）
- 实现方式：外部存储（向量数据库、关系型数据库、文件系统）
- 容量：理论上无限
- 特点：需要显式的读写操作，持久化

## 短期记忆：上下文窗口管理

短期记忆的核心挑战是在有限的上下文窗口内容纳足够的信息：

### 基础对话历史管理

```python
from collections import deque
from typing import Optional

class ConversationManager:
    def __init__(self, max_tokens: int = 4000, model: str = "gpt-4o"):
        self.messages = []
        self.max_tokens = max_tokens
        self.model = model

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()

    def _trim_if_needed(self):
        """保留系统提示和最近的对话历史"""
        while self._estimate_tokens() > self.max_tokens and len(self.messages) > 2:
            # 保留第一条（系统提示），删除最早的用户/助手对
            if self.messages[0]["role"] == "system":
                self.messages.pop(1)  # 删除最早的非系统消息
            else:
                self.messages.pop(0)

    def _estimate_tokens(self) -> int:
        # 粗估：4 个字符约 1 Token
        return sum(len(m["content"]) // 4 for m in self.messages)

    def get_messages(self) -> list:
        return self.messages
```

## 对话历史压缩策略

当对话历史过长时，用 LLM 对早期历史进行摘要压缩：

```python
from openai import OpenAI

client = OpenAI()

def compress_history(messages: list, keep_recent: int = 6) -> list:
    """
    压缩对话历史：
    - 最近 keep_recent 条消息保持原样
    - 更早的历史压缩为摘要
    """
    if len(messages) <= keep_recent + 1:  # +1 for system message
        return messages

    system_msg = [m for m in messages if m["role"] == "system"]
    old_messages = messages[len(system_msg):-keep_recent]
    recent_messages = messages[-keep_recent:]

    if not old_messages:
        return messages

    # 生成早期历史摘要
    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "请将以下对话历史总结为简洁的摘要，保留关键信息和决策。"},
            {"role": "user", "content": str(old_messages)}
        ],
        max_tokens=300
    )
    summary = summary_response.choices[0].message.content

    compressed = system_msg + [
        {"role": "system", "content": f"[早期对话摘要]：{summary}"}
    ] + recent_messages

    return compressed
```

## 长期记忆实现

### 向量数据库存储

将用户信息、偏好和历史事件存储为可检索的向量记忆：

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from openai import OpenAI
import uuid
from datetime import datetime

class LongTermMemory:
    def __init__(self, collection_name: str = "agent_memory"):
        self.qdrant = QdrantClient("http://localhost:6333")
        self.openai = OpenAI()
        self.collection = collection_name

        # 初始化 Collection
        self.qdrant.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    def store(self, content: str, memory_type: str = "episode",
              metadata: dict = None):
        """存储一条记忆"""
        vector = self._embed(content)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "content": content,
                "type": memory_type,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
        )
        self.qdrant.upsert(collection_name=self.collection, points=[point])

    def retrieve(self, query: str, top_k: int = 5,
                 memory_type: str = None) -> list[dict]:
        """检索相关记忆"""
        query_vector = self._embed(query)
        filter_condition = None
        if memory_type:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            filter_condition = Filter(
                must=[FieldCondition(key="type", match=MatchValue(value=memory_type))]
            )

        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_condition
        )
        return [r.payload for r in results]

    def _embed(self, text: str) -> list[float]:
        response = self.openai.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding

# 使用
memory = LongTermMemory()
memory.store("用户偏好使用 Python 3.11 以上版本", memory_type="preference")
memory.store("上次帮助用户解决了异步代码死锁问题", memory_type="episode")

relevant = memory.retrieve("用户的编程偏好", top_k=3)
```

## 情节记忆（Episodic Memory）

情节记忆存储具体的历史事件，便于 Agent 回顾过去的交互：

```python
class EpisodicMemory:
    """存储具体事件和经验"""

    def record_episode(self, task: str, actions: list, outcome: str,
                       success: bool):
        """记录一次任务执行经历"""
        episode = {
            "task": task,
            "actions_taken": actions,
            "outcome": outcome,
            "success": success,
            "learned": self._extract_lesson(task, actions, outcome, success)
        }
        self.memory.store(
            content=f"任务：{task}\n结果：{outcome}\n经验：{episode['learned']}",
            memory_type="episode",
            metadata={"success": success}
        )

    def _extract_lesson(self, task, actions, outcome, success) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"从这次任务经验中提炼关键教训（一句话）：\n任务：{task}\n动作：{actions}\n结果：{outcome}\n是否成功：{success}"}]
        )
        return response.choices[0].message.content
```

## 语义记忆（Semantic Memory）

语义记忆存储通用知识，与具体事件无关：

```python
# 类似 RAG 的知识库
class SemanticMemory:
    """存储通用知识和概念"""

    def add_knowledge(self, fact: str, source: str = None):
        self.memory.store(
            content=fact,
            memory_type="semantic",
            metadata={"source": source}
        )

    def recall(self, concept: str) -> list[str]:
        results = self.memory.retrieve(concept, memory_type="semantic")
        return [r["content"] for r in results]
```

## MemGPT 分页记忆架构

MemGPT（现更名为 Letta）提出了一种类操作系统的分页记忆架构：

- **主上下文（Main Context）**：类似 RAM，在 LLM 上下文窗口内，快速读写
- **外部存储（External Storage）**：类似磁盘，容量无限，需要显式加载到主上下文

Agent 可以通过工具控制记忆的"换页"：
- `core_memory_append`：向主上下文的核心记忆区追加内容
- `core_memory_replace`：替换核心记忆中的内容
- `archival_memory_insert`：将信息归档到外部存储
- `archival_memory_search`：从外部存储检索并加载到上下文

```python
# 核心记忆区（始终在上下文中）
CORE_MEMORY_TEMPLATE = """
[用户基本信息]
姓名：{name}
职业：{occupation}
偏好语言：{preferred_language}

[当前任务状态]
{current_task}

[重要提醒]
{reminders}
"""
```

## 记忆更新与遗忘策略

### 主动更新

Agent 在对话结束时主动整理和更新记忆：

```python
def consolidate_memory_after_session(conversation: list, memory: LongTermMemory):
    """会话结束后提炼记忆"""
    summary_prompt = f"""
    分析以下对话，提取值得长期记忆的信息：
    1. 用户表达的偏好或习惯
    2. 重要的决策或结论
    3. 需要后续跟进的事项

    对话：{conversation}

    以 JSON 格式输出需要记忆的内容列表。
    """
    memories = extract_memories_from_llm(summary_prompt)
    for mem in memories:
        memory.store(mem["content"], memory_type=mem["type"])
```

### 遗忘策略

避免记忆无限增长：
- **基于时间衰减**：超过一定时间未访问的记忆降低权重
- **重要性过滤**：只存储评分超过阈值的记忆
- **去重合并**：相似的记忆合并为更新版本
- **容量上限**：超出容量时，删除最旧或最少访问的记忆

## 多会话记忆持久化

```python
import sqlite3
import json
from pathlib import Path

class PersistentSessionMemory:
    """跨会话的记忆持久化"""

    def __init__(self, db_path: str = "agent_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TIMESTAMP,
                summary TEXT,
                key_facts TEXT  -- JSON array
            )
        """)

    def save_session(self, session_id: str, user_id: str, summary: str,
                     key_facts: list):
        self.conn.execute("""
            INSERT OR REPLACE INTO sessions VALUES (?, ?, datetime('now'), ?, ?)
        """, (session_id, user_id, summary, json.dumps(key_facts, ensure_ascii=False)))
        self.conn.commit()

    def load_user_history(self, user_id: str, limit: int = 5) -> list:
        cursor = self.conn.execute("""
            SELECT summary, key_facts FROM sessions
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (user_id, limit))
        return [{"summary": row[0], "facts": json.loads(row[1])}
                for row in cursor.fetchall()]
```
