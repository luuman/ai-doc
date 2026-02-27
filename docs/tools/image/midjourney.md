# Midjourney

Midjourney 是目前商业化程度最高、用户认可度最好的 AI 图像生成服务。它以出色的审美、稳定的质量和相对简单的使用方式赢得了大批专业用户。从概念设计师到营销团队，Midjourney 已成为视觉创作工作流的重要组成部分。

## MJ Discord 使用

Midjourney 最初（也是至今最稳定的）界面是通过 Discord Bot 使用：

**基础设置**：
1. 访问 midjourney.com，点击"Join the Beta"加入 Discord 服务器
2. 在订阅计划页面选择方案（Free 已停用，需付费）
3. 进入任意 `#newbies-xxx` 频道，或邀请 Midjourney Bot 到自己的服务器

**在自己的 Discord 服务器使用（推荐）**：
```
1. midjourney.com → 点击右上角 Try Now
2. 登录 Discord，进入 Midjourney App 页面
3. Add to Server → 选择你的服务器 → 授权
4. 在你的服务器任意频道使用 /imagine
```

私人服务器的优势：独立的频道、不被他人作品干扰、可专注管理自己的生成记录。

## /imagine 基础命令

所有 Midjourney 生成都从 `/imagine` 命令开始：

```
/imagine prompt: a serene japanese garden at sunset, cherry blossoms falling,
soft golden light, photorealistic, 4k, cinematic composition
```

生成后，每次出图会显示 4 张候选图，下方有操作按钮：
- **U1-U4**：Upscale（放大）选定的图
- **V1-V4**：Variation（变体）——基于选定的图生成4张相似但不同的新图
- **🔄**（刷新）：重新生成全部4张
- **❤️**：收藏到 Gallery

## Prompt 技巧

### 风格词

风格词是控制图像视觉风格的关键：

- **摄影风格**：`photorealistic`、`photography`、`shot on Canon EOS`、`film grain`
- **插画风格**：`digital illustration`、`concept art`、`artstation`
- **油画风格**：`oil painting`、`impressionism`、`painterly`
- **水彩风格**：`watercolor`、`soft edges`
- **动漫风格**：`anime style`、`studio ghibli`、`manga`

### 艺术家参考

引用知名艺术家的风格（注意版权和伦理考量）：
- `in the style of Monet`（莫奈，印象派）
- `Greg Rutkowski style`（奇幻插画）
- `Makoto Shinkai style`（新海诚，动漫背景）
- `Norman Rockwell style`（美式写实插画）

### 光线描述

光线对图像氛围影响极大：
- `golden hour`（黄金时段，日落前后的暖黄光）
- `blue hour`（蓝调时刻，日落后的冷蓝光）
- `dramatic lighting`（戏剧性打光）
- `soft diffused light`（柔和漫反射光）
- `cinematic lighting`（电影布光）
- `rim lighting`（轮廓光，背光效果）
- `studio lighting`（专业影棚灯光）

### 构图描述

- `close-up portrait`（特写人像）
- `wide angle shot`（广角）
- `bird's eye view`（鸟瞰视角）
- `low angle`（仰视）
- `rule of thirds`（三分法构图）
- `symmetrical`（对称构图）

## 参数

Midjourney 通过在 Prompt 末尾添加参数来控制生成行为：

### --ar（宽高比）

```
/imagine prompt: mountain landscape --ar 16:9    # 宽屏横版
/imagine prompt: portrait photo --ar 3:4         # 竖版人像
/imagine prompt: square logo --ar 1:1            # 正方形
/imagine prompt: wide banner --ar 7:3            # 横幅
```

常用比例：
- `1:1`：社交媒体头像
- `16:9`：桌面壁纸、视频缩略图
- `9:16`：手机壁纸、Instagram Stories
- `3:4`：人像摄影
- `4:3`：传统照片比例

### --v（版本）

```
/imagine prompt: ... --v 6.1   # 当前最新版本（推荐）
/imagine prompt: ... --v 5.2   # 较旧但有时产出更"艺术感"
```

不同版本的 Midjourney 在风格和质量上有明显差异。

### --style raw

```
/imagine prompt: portrait photo --style raw    # 关闭 MJ 自动美化
```

默认 MJ 会对图像进行"主观美化"，`--style raw` 更直接响应 Prompt，适合需要精确控制的场景。

### --q（质量）

```
/imagine prompt: ... --q 2    # 更高质量，更慢（生产用途）
/imagine prompt: ... --q .5   # 更快，质量略低（快速预览）
```

### --no（排除词）

```
/imagine prompt: forest scene --no people, buildings, text
```

明确排除不想出现的元素。

### --seed

```
/imagine prompt: cat sitting on chair --seed 12345
```

固定随机种子，确保在相同 Prompt 下可以复现相似结果。

### --chaos

```
/imagine prompt: abstract art --chaos 80    # 0-100，越高越随机/实验性
```

## Vary 和 Remix 功能

### Vary（变体）

选择一张 Upscale 后的图，点击 "Vary (Strong)" 或 "Vary (Subtle)"：

- **Vary (Subtle)**：小幅变化，保持整体构图和风格
- **Vary (Strong)**：较大变化，同主题不同细节
- **Vary (Region)**：选择特定区域重新生成（局部重绘）

### Remix 模式

开启 Remix 模式后（`/prefer remix`），每次 Variation 操作时都可以修改 Prompt，实现渐进式的图像编辑：

```
原始 Prompt → 生成一张花园图
Vary → 修改 Prompt 为"花园，秋天落叶" → 保持构图，改变季节
再 Vary → 修改为"夜晚的花园，月光" → 继续演化
```

## Upscale（放大）

生成的图像默认分辨率约 1024×1024（V6），Upscale 进一步提升：

- **Upscale (2x)**：分辨率翻倍
- **Upscale (4x)**：分辨率 4 倍（Subtle 版：细节增强；Creative 版：AI 添加细节）

**Upscale Creative** 会在放大时添加 AI 想象的细节，可能与原图有细微差异，适合追求丰富细节的场景。

## Alpha Web 界面

Midjourney 在 2024 年推出了 midjourney.com 的 Alpha Web 界面，逐步摆脱对 Discord 的依赖：

- 图像管理（Gallery 浏览、下载、分享）
- 全新的 Explore 功能（浏览社区公开作品和其 Prompt）
- 编辑器模式（类似 img2img，可以在生成图上局部修改）
- Task List（图像生成队列可视化）

Web 界面功能仍在快速迭代中，逐渐接近完整替代 Discord 的程度。

## 订阅计划

| 计划 | 价格 | 快速生成 GPU 时间 | Relaxed 生成 | 并发任务 |
|------|------|----------------|------------|--------|
| Basic | $10/月 | ~200张/月 | 无 | 3 任务 |
| Standard | $30/月 | ~900张/月 | 无限 | 3+1 任务 |
| Pro | $60/月 | ~1800张/月 | 无限 | 12 任务 |
| Mega | $120/月 | ~3600张/月 | 无限 | 12 任务 |

说明：
- **快速生成（Fast）**：优先队列，通常几秒内出图
- **放松生成（Relaxed）**：Standard+ 用户享有，在空闲 GPU 上生成，可能需要等待
- 年付享受约 20% 折扣

## 商业授权

Midjourney 的商业使用授权与订阅计划挂钩：

- **Basic/Standard/Pro/Mega**：付费订阅用户拥有商业使用权，可以用生成的图像进行商业活动（产品包装、广告、NFT 等）
- **未付费用户**（历史免费期）：仅限个人非商业使用

重要限制：
- 不能声称是人类创作的作品（需要标注 AI 生成）
- 不能用于生成针对真实人物的诽谤或误导性内容
- 建议查阅最新的 Terms of Service，版权法规仍在演进中

## 与 Stable Diffusion 的对比

| 维度 | Midjourney | Stable Diffusion |
|------|-----------|----------------|
| 易用性 | 极高（Discord 命令） | 较复杂（需要配置） |
| 图像质量 | 审美出色、商业级 | 取决于模型和参数 |
| 可控性 | 中（参数有限） | 极高（ControlNet、LoRA） |
| 隐私性 | 生成图像默认公开（Pro 可私有） | 完全本地 |
| 成本 | $10-120/月订阅 | 免费（本地）或按 GPU 时间付费 |
| 定制化 | 不可微调 | 可训练 LoRA、Dreambooth |
| 批量生成 | 有限 | 脚本批量无限制 |
| 适合场景 | 快速出高质量成片、概念探索 | 高度定制化、角色一致性、商业批量 |

**选择建议**：如果追求"快速出好图"，选 Midjourney；如果需要精确控制、角色一致性或本地隐私，选 Stable Diffusion + ComfyUI。两者不是竞争关系，很多专业创作者同时使用。
