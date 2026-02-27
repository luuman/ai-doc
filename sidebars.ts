import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    { type: "doc", id: "index", label: "知识库总览" },

    // ── 1. AI 全景概述 ──────────────────────────────────────
    {
      type: "category",
      label: "AI 全景概述",
      collapsed: false,
      items: [
        "overview/index",
        "overview/history",
        "overview/paradigms",
        "overview/agi",
      ],
    },

    // ── 2. 基础理论 ─────────────────────────────────────────
    {
      type: "category",
      label: "基础理论",
      collapsed: true,
      items: [
        "fundamentals/index",
        {
          type: "category",
          label: "数学基础",
          items: [
            "fundamentals/math/linear-algebra",
            "fundamentals/math/probability",
            "fundamentals/math/calculus",
          ],
        },
        {
          type: "category",
          label: "机器学习",
          items: [
            "fundamentals/ml/supervised",
            "fundamentals/ml/unsupervised",
            "fundamentals/ml/reinforcement",
          ],
        },
        {
          type: "category",
          label: "深度学习",
          items: [
            "fundamentals/deep-learning/neural-networks",
            "fundamentals/deep-learning/cnn",
            "fundamentals/deep-learning/rnn",
            "fundamentals/deep-learning/transformer",
          ],
        },
      ],
    },

    // ── 3. 大语言模型 ───────────────────────────────────────
    {
      type: "category",
      label: "大语言模型（LLM）",
      collapsed: false,
      items: [
        "llm/index",
        {
          type: "category",
          label: "模型原理",
          items: [
            "llm/principles/autoregressive",
            "llm/principles/tokenization",
            "llm/principles/scaling-law",
            "llm/principles/attention",
          ],
        },
        {
          type: "category",
          label: "主流模型",
          items: [
            "llm/models/gpt-series",
            "llm/models/claude",
            "llm/models/gemini",
            "llm/models/llama",
            "llm/models/open-source",
          ],
        },
        {
          type: "category",
          label: "训练体系",
          items: [
            "llm/training/pretraining",
            "llm/training/sft",
            "llm/training/rlhf",
            "llm/training/lora",
            "llm/training/alignment",
          ],
        },
        "llm/evaluation",
      ],
    },

    // ── 4. 多模态 AI ────────────────────────────────────────
    {
      type: "category",
      label: "多模态 AI",
      collapsed: true,
      items: [
        "multimodal/index",
        {
          type: "category",
          label: "计算机视觉",
          items: [
            "multimodal/vision/image-classification",
            "multimodal/vision/object-detection",
            "multimodal/vision/segmentation",
            "multimodal/vision/generation",
          ],
        },
        {
          type: "category",
          label: "语音 AI",
          items: [
            "multimodal/speech/asr",
            "multimodal/speech/tts",
            "multimodal/speech/whisper",
          ],
        },
        {
          type: "category",
          label: "多模态模型",
          items: [
            "multimodal/models/vlm",
            "multimodal/models/clip",
            "multimodal/models/text-to-image",
            "multimodal/models/text-to-video",
          ],
        },
      ],
    },

    // ── 5. 提示工程 ─────────────────────────────────────────
    {
      type: "category",
      label: "提示工程",
      collapsed: true,
      items: [
        "prompt-engineering/index",
        "prompt-engineering/basics",
        {
          type: "category",
          label: "核心技巧",
          items: [
            "prompt-engineering/techniques/chain-of-thought",
            "prompt-engineering/techniques/few-shot",
            "prompt-engineering/techniques/role-playing",
            "prompt-engineering/techniques/structured-output",
            "prompt-engineering/techniques/meta-prompting",
          ],
        },
        "prompt-engineering/best-practices",
      ],
    },

    // ── 6. AI 应用开发 ──────────────────────────────────────
    {
      type: "category",
      label: "AI 应用开发",
      collapsed: false,
      items: [
        "development/index",
        {
          type: "category",
          label: "API 接入",
          items: [
            "development/apis/anthropic",
            "development/apis/openai",
            "development/apis/google-ai",
          ],
        },
        {
          type: "category",
          label: "开发框架",
          items: [
            "development/frameworks/langchain",
            "development/frameworks/llamaindex",
            "development/frameworks/dspy",
          ],
        },
        {
          type: "category",
          label: "RAG 检索增强",
          items: [
            "development/rag/index",
            "development/rag/embeddings",
            "development/rag/vector-db",
            "development/rag/chunking",
            "development/rag/retrieval",
          ],
        },
        {
          type: "category",
          label: "AI Agent",
          items: [
            "development/agents/index",
            "development/agents/tool-use",
            "development/agents/memory",
            "development/agents/planning",
            "development/agents/multi-agent",
          ],
        },
      ],
    },

    // ── 7. AI 工程架构 ──────────────────────────────────────
    {
      type: "category",
      label: "AI 工程架构",
      collapsed: true,
      items: [
        "engineering/index",
        {
          type: "category",
          label: "训练工程",
          items: [
            "engineering/training/data-pipeline",
            "engineering/training/distributed-training",
            "engineering/training/gpu-architecture",
            "engineering/training/mixed-precision",
          ],
        },
        {
          type: "category",
          label: "推理部署",
          items: [
            "engineering/inference/optimization",
            "engineering/inference/quantization",
            "engineering/inference/onnx-tensorrt",
            "engineering/inference/serving",
            "engineering/inference/edge-deployment",
          ],
        },
        {
          type: "category",
          label: "系统架构",
          items: [
            "engineering/system/streaming",
            "engineering/system/multi-model-routing",
            "engineering/system/cost-optimization",
            "engineering/system/rate-limiting",
            "engineering/system/monitoring",
          ],
        },
      ],
    },

    // ── 8. AI 工具生态 ──────────────────────────────────────
    {
      type: "category",
      label: "AI 工具生态",
      collapsed: true,
      items: [
        "tools/index",
        {
          type: "category",
          label: "编程辅助",
          items: [
            "tools/coding/claude-code",
            "tools/coding/cursor",
            "tools/coding/github-copilot",
            "tools/coding/comparison",
          ],
        },
        {
          type: "category",
          label: "图像生成",
          items: [
            "tools/image/stable-diffusion",
            "tools/image/midjourney",
            "tools/image/flux",
            "tools/image/comfyui",
          ],
        },
        {
          type: "category",
          label: "本地部署",
          items: [
            "tools/local/ollama",
            "tools/local/lm-studio",
            "tools/local/vllm",
          ],
        },
        {
          type: "category",
          label: "效率工具",
          items: [
            "tools/productivity/chatgpt",
            "tools/productivity/perplexity",
            "tools/productivity/notebooklm",
          ],
        },
      ],
    },

    // ── 9. AI 产品与商业 ────────────────────────────────────
    {
      type: "category",
      label: "AI 产品与商业",
      collapsed: true,
      items: [
        "product/index",
        {
          type: "category",
          label: "产品设计",
          items: [
            "product/design/structure",
            "product/design/ux",
            "product/design/human-ai-collaboration",
            "product/design/risk-control",
          ],
        },
        {
          type: "category",
          label: "商业模式",
          items: [
            "product/business/saas-ai",
            "product/business/agent-platform",
            "product/business/api-monetization",
            "product/business/cost-model",
          ],
        },
      ],
    },

    // ── 10. 行业应用 ────────────────────────────────────────
    {
      type: "category",
      label: "行业应用",
      collapsed: true,
      items: [
        "industry/index",
        "industry/finance",
        "industry/medical",
        "industry/industrial",
        "industry/smart-city",
      ],
    },

    // ── 11. 安全与伦理 ──────────────────────────────────────
    {
      type: "category",
      label: "安全与伦理",
      collapsed: true,
      items: [
        "safety/index",
        {
          type: "category",
          label: "技术安全",
          items: [
            "safety/security/adversarial",
            "safety/security/prompt-injection",
            "safety/security/jailbreak",
            "safety/security/red-teaming",
          ],
        },
        {
          type: "category",
          label: "伦理与治理",
          items: [
            "safety/ethics/bias",
            "safety/ethics/explainability",
            "safety/ethics/regulations",
            "safety/ethics/governance",
          ],
        },
      ],
    },

    // ── 12. 未来趋势 ────────────────────────────────────────
    {
      type: "category",
      label: "未来趋势",
      collapsed: true,
      items: [
        "future/index",
        "future/world-model",
        "future/autonomous-agent",
        "future/robotics",
        "future/web3",
      ],
    },
  ],
};

export default sidebars;
