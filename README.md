<div align="center">
  <img src="packaging/AI-Jingjing.png" width="128" alt="AI静静 Logo">
  <h1>AI知识库 · AI静静</h1>
  <p><strong>本地优先、多模态、可溯源的个人知识库桌面应用</strong></p>
  <p>简体中文 · <a href="README_EN.md">English</a></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/UI-PySide6-41CD52" alt="PySide6">
    <img src="https://img.shields.io/badge/Storage-SQLite%20FTS5-0F80CC" alt="SQLite FTS5">
    <img src="https://img.shields.io/badge/Test-pytest-2F855A" alt="pytest">
    <img src="https://img.shields.io/badge/Version-2.5.0-4C8FBF" alt="Version 2.5.0">
  </p>
</div>

AI静静把 PDF、PPT、Word、图片、音频、视频、网页和 Markdown 统一整理成一套可以搜索、连续提问、查看原始证据，并按生命周期长期维护的本地知识库。

版本变化见 [CHANGELOG.md](CHANGELOG.md)。

它是一款独立桌面软件：日常使用不需要安装 Codex、Obsidian、Python 或独立 FFmpeg。Obsidian 仅作为可选同步目标；DeepSeek、Kimi 等云模型只在用户主动生成回答、知识提炼或视觉理解时调用。

## 为什么做 AI静静

普通文件夹只能保存资料，普通聊天工具又容易丢失原始依据。AI静静同时保留三层内容：

```text
原始资料与归档
       ↓
可定位的知识块与语义索引
       ↓
带引用的回答、笔记与知识作品
```

因此，一个答案不仅告诉你结论，还可以回到对应的 PDF 页码、PPT 页、音视频时间点或原始网页。

## 核心能力

### 1. 统一多模态入库

- 一次选择或拖入多个文件。
- 支持 Markdown、纯文本、Word、PDF、PPTX、图片、音频、视频、普通网页和微信公众号文章。
- PDF 与 PPT 按页面理解，重要页面保留图像证据。
- 音视频先执行解码预检，再原子标准化为 16 kHz、单声道、PCM16 音轨并运行本地 VAD；视频还可抽取关键帧。
- 提供“中文高精度（Qwen3-ASR 1.7B）”“快速预览（Qwen3-ASR 0.6B）”“兼容模式（Whisper）”和自定义路线，可在 Qwen3-ASR、MLX Whisper、faster-whisper 之间明确选择与降级。
- Apple Silicon 可使用 MLX/Metal；NVIDIA 与 CPU 环境可使用 faster-whisper CUDA/CPU int8。界面会显示实际 Provider、模型、设备和每次回退原因，用户取消不会触发回退。
- 内置本地模型管理器可查看状态与体积、登记已有目录、复制导入、显式下载或移除应用管理的权重，并对真实模型文件执行可取消的 SHA-256 内容校验。导入和状态检查不会静默下载；安装包不包含 Qwen3-ASR、Whisper 或说话人模型权重。
- 支持全局、知识空间和单一来源三层专业词库；词库版本、Provider、模型内容哈希和实际回退路线都会写入转写事实与检查点。
- 可选区分多位说话人，保留匿名说话人、重叠讲话、人数范围和对齐结果，之后可人工命名、重新分配或合并。
- `Transcript V2` 保存原始识别、校订稿、句段/词级时间戳、说话人、模型路线、回退历史和质量状态；同时继续生成 JSON、Markdown、TXT、SRT 与 VTT 产物。
- `audio_probe`、标准化音轨、VAD、ASR、说话人分段、Transcript V2 和质量报告均有原子持久检查点；程序中断后只重跑未完成阶段，源、配置、模型或词库变化会自动失效旧检查点。
- 公开链接支持 YouTube、B 站、抖音、小红书和 X：优先取得公开字幕，无字幕时仅下载可逐块监控的 HTTP/HTTPS 或原生 DASH 媒体；直播、未完成回放和可能退回无界 FFmpeg 下载的 HLS 会在下载前拒绝，并提示先合法保存为本地文件。
- 同名 PPT、PDF、录音和视频自动归为同一个 `Source Package`。
- 自动归档原件、网页快照、转写、页面资源和解析清单。
- 摄取流程是正式 Python 服务，不调用本地 Codex CLI。

### 2. 严格入库质检

每份资料在写入知识库前都会检查：

- 是否取得真实正文或真实媒体流；
- 是否只有标题、简介、封面或平台元数据；
- PDF/PPT 页面覆盖率；
- 音轨能否解码，以及编码、采样率、声道、时长、音量、静音比例和削波风险；
- VAD 是否检测到有效语音区间，音视频是否产生真实转写或画面理解；
- OCR 行级坐标、平均/最低置信度、低分行和复杂版面降级原因；
- PP-StructureV3 表格 Markdown、公式、版面坐标和页面阅读顺序；
- 音视频时间段是否倒序、越界、异常重叠、首尾缺失或出现空段；
- 是否出现重复生成、截断、静音区间幻觉、异常语速、语言/专业术语/数字单位风险；
- 说话人数是否偏离预期，未知说话人、重叠讲话或过度碎片化是否需要人工复核；
- 原始文件校验值、正文规模和解析告警。

只开放说明和封面的受限视频链接会被明确拒绝，不会把简介伪装成视频内容入库。

音视频转写为“需要复核”或“失败”时，只保存原始资料与 Transcript V2 事实，不建立 FTS 或向量索引；人工校订并批准后才会原子补建索引。AI 知识提炼是独立派生层，强制区分已确认事实、待验证推测、争议、决策和行动项，并拒绝原始资料中不存在的页码或时间戳。

### 3. 完整深度语义精校

深度精校不是把整篇转写交给模型“重写”，而是在不可变原稿之上建立一条可恢复、可审计的派生链路：

```text
Transcript V2 原始识别（只读）
        ↓
异常扫描：重复循环 / 低置信 / 静音幻觉 / 截断 / 术语 / 数字单位
        ↓
连续且带重叠上下文的分块
        ↓
可选局部重识别：仅重跑异常时间区间，可切换 Large-v3 或 Qwen3-ASR
        ↓
结构化 LLM 跨片段精校 + 术语与实体一致性
        ↓
可选外部证据核验（必须由用户显式开启联网）
        ↓
逐条修改建议、证据、理由、置信度与接受/拒绝审计
        ↓
说话人版 Markdown / 章节 / 知识卡片 / Mermaid 知识结构图
```

- 转写路线可选择 `Whisper large-v3`、`Qwen3-ASR 1.7B` 或 `Qwen3-ASR 0.6B`。首轮识别和异常区间局部重识别可以使用不同路线，用更强模型复核疑难片段，而不是浪费算力重跑整段媒体。
- 可选说话人分离保留匿名 `S1/S2/...`、时间范围和重叠讲话；人工命名、合并或重新分配不会改写原始 ASR 事实。说话人身份是人工标注，不会仅凭声纹猜测真实姓名。
- LLM 必须返回严格结构化 JSON，并保持每个 `segment_id`、开始/结束时间和说话人映射。系统拒绝缺片段、越界定位、伪造时间戳、无法对应原文的引用和静默删除内容。
- 外部核验默认关闭。开启后只把有界查询发送给检索服务；网页文本始终作为不可信数据，模型只能引用注入结果中的证据 ID、原 URL 和逐字存在的摘录，不能执行网页中的指令。证据不足时保留原意并标记 `［待核实］`、`［术语待核实］`、`［听辨不清］` 或 `［ASR解码失败］`，不会为了流畅而补写事实。
- 每项建议保存 `before / after / reason / confidence / evidence`，可以逐条接受或拒绝。原始 `raw_text` 永不覆盖；获批内容写入独立校订层，检查点允许暂停、取消和恢复。
- 导出的 Markdown 可同时包含完整时间轴、说话人、章节、术语待核实表、全量原稿/精校稿差异审计、知识卡片和 Mermaid 图；同时生成经过片段映射、顺序、边界和覆盖校验的精校版 SRT/VTT。
- Apple Silicon 新增 `Whisper Large v3 Turbo Q4` 长音频档；Whisper 路线显式关闭 `condition_on_previous_text`，降低长中文重复解码循环。

模型权重不随安装包分发。Large-v3、Qwen3-ASR 和说话人模型需由用户在模型管理器中显式下载或登记已有本地目录；没有相应权重时不会静默联网或假装完成局部重识别。

### 4. 正式知识治理

- 原始资料自动映射为 `source`，知识工坊产物自动映射为 `output`。
- 支持来源、主题、实体、分析、决策、成果 6 类正式知识。
- 支持草稿、当前有效、需要复核、可能过期、已归档 5 种生命周期状态。
- 支持未检查、已索引、已总结、已沉淀、低价值保留 5 档成熟度。
- 支持 `supports`、`extends`、`contradicts`、`supersedes`、`opens` 关系与双向查询。
- 回答可一键“沉淀为知识”，自动生成 Markdown 笔记并关联本轮真实来源；AI 内容默认进入“需要复核”。
- 深度精校知识卡先进入“候选审核”，按标题和别名做确定性去重；可接受为待复核知识、合并到已有知识或拒绝，不会直接写成确定事实。
- 每个知识空间拥有结构化策略：候选开关、人工复核门禁、来源默认可靠性、外部核验许可、冲突处理和模型/转写路线。策略字段采用允许列表，不执行外部 `AGENTS.md` 或任意脚本。
- “知识运营中心”统一提供候选审核、来源可靠性/有效期/解析完整度、显式冲突、重复资料、黄金问题集评测和 SOP 流程库。
- SOP 以步骤、触发条件、模型边界和隐私边界保存，是可审查的流程资产，不是任意代码执行器。
- 可一键从 SQLite 事实编译便携 `LLM-Wiki`，包含 frontmatter、类型目录、标签索引、原始资料/成果状态页、关系链接和知识操作日志；SQLite 仍是唯一事实源。
- 知识体检会发现孤立知识、缺失来源、空正文、过期内容、标签不统一、别名冲突和高价值来源未编译，并给出恢复建议。
- 删除正式知识前会先原子保存 tombstone、Markdown、别名、标签和知识关系；可在“知识 → 回收站”恢复原 ID 与仍有效的关系，原始资料不会被删除。

### 5. 本地语义检索

- SQLite FTS5 全文检索；
- 本地中英文语义 Embedding；
- 向量召回与 BM25 召回融合；
- RRF 融合与本地重排；
- 按最终相关性自动排序并剔除明显无关的候选片段；
- 按知识空间、标签、资料类型或指定文档限定范围；
- 展示融合分数、语义分数和全文命中状态。

检索过程不调用大模型。首次使用会下载约 240 MB 的多语言 ONNX 模型，之后可以离线运行。

### 6. 连续对话与可信引用

- 支持同一会话内连续多轮提问；
- 对话自动持久化，可搜索、重新打开、重命名、删除和导出 Markdown；
- 云模型回答逐字流式显示，可随时停止并保留已生成内容；
- 每条回答支持复制、重新生成和“有帮助/需改进”本地反馈；
- 可在提问框直接粘贴截图、拖入图片或通过按钮一次选择最多 4 张图片；
- 发送前显示缩略图并可移除；视觉模型会联合理解文字、图片和检索证据；
- 后续问题可继续指代上一轮图片，无需重复添加；
- 自动使用近期消息与滚动摘要重写检索问题；
- 回答只能引用当前检索得到的证据；
- 引用 ID、文档 ID 和知识块 ID 会在本地数据库中校验；
- 点击回答中的引用可打开原文阅读器并定位页码或时间点；
- 没有足够证据时明确说明，而不是编造答案。

检索上下文会自动选择三种策略：普通问题使用聚焦检索；选中的小文档可使用完整上下文；超长课程或报告使用分层抽样。右侧显示“证据充分 / 部分有据 / 引用有限 / 证据不足”，并可展开查看引用覆盖、证据利用率、来源多样性和具体判断依据。这里展示的是可复核的覆盖指标，不是模型虚构的“正确率”。

所有召回内容都被封装为不可信证据数据。系统会标记疑似提示词注入，并明确禁止执行证据中的命令、角色指令或泄密请求。

回答模型可以选择 DeepSeek、Kimi 或完全离线的本地证据模型。配置 DeepSeek 后默认使用支持文字与图片理解的实验模型 `deepseek-v4-flash-vision-exp`；检索本身不调用大模型，因此日常搜索成本很低。

### 7. 原文阅读与资料管理

- 在应用内阅读 PDF 页面、图片、解析文本与音视频时间轴；
- 从回答引用或转写片段直接打开内置播放器并跳转到精确时间点，播放时同步高亮当前片段，也可调整速度或用系统应用打开原文件；
- 在转写校订器中对照只读的原始识别文字修改校订稿，填写修改原因，并命名、重新分配或合并说话人；所有变化保留审计记录，保存后重建相关索引；
- 查看每份资料的原始知识块和定位信息；
- 对证据添加批注并绑定到具体知识块；
- 重命名、停用、重新启用、重新解析或移除资料；
- 编辑知识空间和标签；
- 检查内容指纹相同的重复资料；
- 原始归档默认保留，移除索引不会销毁原文件。

### 8. 自动同步与知识工坊

- 监听一个或多个文件夹并增量入库；
- 每批导入及每个文件的阶段、进度、错误和结果都会持久化；
- 程序异常退出后，未完成任务恢复为“可继续”，失败项可单独重试；
- 文件变化后只重新解析变化项；
- 源文件消失时先停用对应知识，不静默删除；
- 可选从 Obsidian 增量同步，或把 AI静静笔记导出至 Obsidian；
- 基于选定资料生成综合报告、多资料比较、时间线、测验、复习闪卡和思维导图。

### 9. 本地数据安全

- 知识、索引、对话、答案和引用保存在本机 SQLite；
- API Key 优先保存在 macOS Keychain 或系统凭据存储；
- 备份包明确排除 API Key；
- V2 完整备份覆盖数据库、设置、笔记、原始归档、页面资源和转写，并为每个文件记录大小与 SHA-256；
- 恢复前先验证路径、压缩比、大小、哈希、SQLite 和设置，再创建当前数据安全快照并原子恢复；
- 更新清单和下载重定向只接受 HTTPS；安装包下载完成后必须通过清单中的 SHA-256 才允许打开；
- 本地文件路径、数据库和个人知识目录均由 `.gitignore` 排除。
- 本地隐私扫描可检测密钥、令牌、私钥、邮箱、手机号、本机绝对路径、敏感文件名和图片 EXIF/GPS；可选使用本地 OCR 检查图片文字。
- 扫描报告只显示脱敏路径、风险类型和行号，不回显命中的秘密或完整用户路径。
- “安全分享副本”默认不包含任何知识；只有明确勾选的正式知识、知识工坊成果和资料才会进入副本。Source Notes 和已保存回答默认排除，避免带出本机定位信息。
- 分享副本固定排除数据库、API 配置、钥匙串、缓存、对话、备份和回收站；生成前后各扫描一次，并写入逐文件 SHA-256 清单。
- 存在阻断风险或未完整检查的内容时停止生成，不提供专家绕过。PDF 与新版 Office 仍会接受深度本地扫描以生成脱敏风险报告，但原始 PDF、Office/ODF、图片及音视频容器不会被原样复制到安全副本；当前安全副本只发布通过严格文本校验的 Markdown/纯文本等资料。这样可以阻断解析器不可达对象、压缩尾部和隐藏元数据；后续只有在实现页面/像素重建净化后才会重新开放这些原始容器。应用只创建本地副本，不自动上传或发送。

## 支持的输入

| 类型 | 扩展名/形式 | 主要处理 | 可追溯位置 |
|---|---|---|---|
| Markdown / 文本 | `.md` `.txt` `.csv` `.json` `.yaml` | 标题结构、段落、代码与表格 | 标题路径、字符范围 |
| Word | `.docx` | 段落、标题、表格 | 章节与原文件 |
| PDF | `.pdf` | 逐页文本、扫描页 OCR、页面图像 | 页码 |
| PowerPoint | `.pptx` | 逐页文字、备注、图片与页面结构 | 幻灯片页码 |
| 图片 | `.png` `.jpg` `.webp` `.tiff` 等 | OCR 与可选视觉理解 | 原图与图片资源 |
| 音频 | `.mp3` `.m4a` `.wav` `.flac` 等 | 预检、PCM16 标准化、VAD、Qwen3-ASR/Whisper、可选说话人识别 | 句段/词级时间与说话人 |
| 视频 | `.mp4` `.mov` `.mkv` `.webm` 等 | FFmpeg 音轨、VAD、Qwen3-ASR/Whisper、可选说话人识别与关键帧 | 时间轴、说话人与关键帧 |
| 网页/媒体链接 | `https://...` | 正文抓取、网页快照或真实媒体流下载 | URL、快照、时间轴 |
| 微信公众号文章 | `mp.weixin.qq.com/s/...` | 专用正文与标题提取，验证页拦截 | 文章 URL、正文快照 |
| 公开视频平台 | YouTube、B 站、抖音、小红书、X | 公开字幕优先；无字幕时下载公开媒体并转写 | 原链接、字幕、时间轴 |

## 架构

```text
PySide6 桌面界面
        │
        ├── IngestionService
        │   ├── 文档/PDF/PPT/图片解析
        │   ├── OCR / FFmpeg / 音频预检与标准化 / VAD
        │   ├── ASR Router (Qwen3-ASR / MLX Whisper / faster-whisper)
        │   ├── 可选说话人识别 / Transcript V2 / 质量门禁
        │   ├── DeepCorrectionService
        │   │   ├── 异常检测 / 重叠分块 / 异常区间局部重识别
        │   │   ├── 结构化精校 / 术语实体一致性 / 保守不确定标注
        │   │   └── 证据审计 / 逐条接受拒绝 / Markdown 知识产物
        │   ├── 入库真实性与完整性质检
        │   └── Source Package 归档
        │
        ├── KnowledgeDatabase (SQLite)
        │   ├── Documents / Chunks / Source References
        │   ├── FTS5 全文索引 / Embeddings
        │   ├── Conversations / Evidence / Citations
        │   └── Governed Knowledge / Relations / Lifecycle
        │
        ├── KnowledgeRetriever
        │   ├── Semantic Vector + BM25 + RRF + Rerank
        │   └── Focused / Full Context / Hierarchical
        │
        └── KnowledgeQAEngine
            ├── 上下文问题重写
            ├── DeepSeek / Kimi / Local Extractive
            ├── 不可信证据边界与提示注入防护
            └── 证据构建、引用校验与质量解释

本地安全层还提供脱敏隐私扫描、显式选择的分享副本、二次内容扫描和 SHA-256 完整性校验；该流程不包含网络发送能力。
```

## 快速开始

要求：Python 3.11 或更高版本。

```bash
git clone git@github.com:anyuzhe/aijingjing.git
cd aijingjing

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[full]'

ai-jingjing
```

只开发桌面界面、不需要完整多媒体解析时：

```bash
pip install -e '.[desktop,semantic]'
```

可选增强组件：

```bash
# Apple Silicon 的 Qwen3-ASR 与 MLX Whisper 运行时
pip install -e '.[apple-media]'

# 可选说话人识别运行时
pip install -e '.[speaker]'

# 复杂表格、公式和多栏扫描件的 PP-StructureV3
# 先按平台安装 PaddlePaddle 3.0+：https://www.paddlepaddle.org.cn/install/quick
pip install -e '.[layout-ocr]'
```

`.[full]` 已包含公开平台连接器、faster-whisper 和轻量 Sherpa-ONNX 说话人推理运行时；源码开发时可用 `.[apple-media]` 增加 Qwen3-ASR/MLX Whisper，用 `.[speaker]` 再增加 pyannote 后端。macOS Apple Silicon 正式包可包含 MLX 推理运行时，但所有 ASR 与说话人模型权重都不随安装包分发。PaddleOCR 仍作为专业版面组件按需安装；`layout-ocr` 已包含 PP-StructureV3 的文档解析依赖，但 PaddlePaddle 需要按 CPU/CUDA 与操作系统选择官方安装包。缺失组件会在“帮助 → 系统诊断”中明确显示，不会静默伪装为成功。

`scripts/build_desktop.sh` 会检测构建机架构：macOS arm64 自动安装 `full,apple-media`，使正式包带有 MLX/Qwen 运行时；Intel macOS、Windows 与 Linux 继续使用跨平台的 `full` 依赖。脚本不会下载任何模型权重。

### 配置回答模型

启动应用后打开“设置 → 模型与隐私”，填写 DeepSeek 或 Kimi API Key。密钥不会显示在状态接口、诊断日志、SQLite 或备份中。

不配置任何 API Key 时，入库、本地搜索和离线证据回答仍然可用。

### 配置本地转写模型

打开“设置 → 多媒体解析”，选择转写方案、Provider、模型、主要语言、上下文术语、词级时间戳和可选说话人数范围；可在同一页管理三层专业词库，再点击“管理本地模型…”安装或登记权重：

- `Qwen3-ASR 1.7B`：中文会议、课程与专业内容的高精度档，约 2.2 GB；
- `Qwen3-ASR 0.6B`：速度与占用优先的快速预览档，约 0.9 GB；
- Whisper：提供 `tiny / base / small / medium / large-v3` 多档，以及 Apple Silicon 长音频推荐的 `large-v3-turbo-q4`；Apple MLX 与 faster-whisper CTranslate2 权重分别管理，绝不会混用；
- Sherpa-ONNX：轻量纯本地说话人分离；需同时登记 pyannote segmentation 与 3D-Speaker embedding 两个 ONNX 权重；
- pyannote Community-1：可选本地说话人识别模型，首次取得权重前需在 Hugging Face 接受上游条款。

模型管理器支持登记已有模型目录、复制导入，并且对具有受信 Hugging Face 仓库的条目支持用户主动下载；Sherpa 双 ONNX 组合使用导入/登记。仅查看状态、启动应用或导入音视频都不会联网下载模型；缺少所选权重时会明确提示安装、登记目录或切换路线。当前音视频入口只处理已有本地文件或已合法取得的媒体，不提供麦克风采集、实时录音或实时流式转写。

### 使用知识运营中心

在主界面左侧打开“知识”，点击“知识运营中心”；也可使用“知识库 → 知识运营中心”或 `Ctrl+Shift+K`。候选审核、来源与冲突、黄金评测、SOP 流程库和便携 Wiki 都在这里。空间级规则位于“设置 → 知识空间策略”；音视频模型仍在“设置 → 多媒体解析”，深度语义精校模型位于“设置 → 深度精校”。

## 命令行

```bash
# 启动桌面应用
knowledge desktop

# 批量多模态入库
knowledge ingest report.pdf slides.pptx recording.m4a

# 本地混合检索
knowledge search "FDE 的前线学习机制"

# 有引用的知识问答
knowledge ask "FDE 是什么？"

# 查看索引状态、诊断组件、重建索引
knowledge index-status
knowledge doctor
knowledge reindex

# 使用本地黄金集评估检索与引用质量
knowledge eval docs/golden-evaluation.example.json --top-k 10
```

搜索和问答支持 `--collection`、`--tag`、`--media-type`、`--folder` 和 `--document-id` 等范围过滤参数。

## 本地数据目录

```text
AI-Jingjing/
├── archive/       原始资料、网页快照和 Source Packages
├── notes/         Source Notes、正式知识、AI 回答与知识工坊产物
├── assets/        PDF/PPT 页面、图片与视频关键帧
├── transcripts/   Transcript V2、字幕及兼容格式转写
├── models/        显式管理的 ASR 与说话人模型权重
├── cache/models/  本地语义模型缓存
├── backups/       可验证备份
├── trash/         可恢复资料
├── knowledge.db   索引、对话、答案、证据与引用
├── providers.json 模型连接信息，不存放已迁移的明文密钥
└── settings.json  应用设置
```

- macOS：`~/Library/Application Support/AI-Jingjing`
- Windows：`%LOCALAPPDATA%\AI-Jingjing`
- Linux：`${XDG_DATA_HOME:-~/.local/share}/AI-Jingjing`

可以通过环境变量 `AI_JINGJING_DATA_DIR` 或启动参数 `--data-dir` 指定其他目录。

## 隐私边界

| 操作 | 是否调用云服务 | 发送内容 |
|---|---:|---|
| 分块、SQLite、全文检索、本地语义检索 | 否 | 无 |
| 音频预检、VAD、本地 ASR、说话人识别、播放与校订 | 否 | 无 |
| 深度精校：异常检测、局部本地重识别、分块与审计 | 否 | 无 |
| 深度精校：云 LLM 语义校正 | 是（用户主动运行） | 选中转写片段、相邻上下文、术语与必要的说话人标签；不发送原始音视频或整个知识库 |
| 深度精校：外部网页核验 | 是（另行显式开启） | 有界核验查询；返回摘录、标题与 URL 作为不可信证据保存到审计记录 |
| 本地证据回答 | 否 | 无 |
| DeepSeek/Kimi 回答 | 是 | 问题、有限对话上下文、召回的证据块，以及用户主动附加的图片 |
| DeepSeek 知识提炼 | 是 | 当前导入资料的有界提取文本 |
| DeepSeek Vision/Kimi 视觉理解 | 是 | 用户启用后由导入策略选中的有限图片 |
| 公开平台字幕/媒体获取 | 是 | 用户主动提交的公开 URL；不读取浏览器 Cookie、netrc 或代理 |
| 本地隐私扫描与安全分享副本 | 否 | 无；副本只写入用户选择的本地目录 |

模型服务不会收到整个知识数据库、归档目录或 Obsidian Vault。

## 测试

```bash
pip install pytest
pytest -q
```

2.5.0 的自动化测试覆盖 ASR Provider 路由、本地模型生命周期与内容校验、Turbo Q4、长中文解码保护、音频预检/标准化/VAD、持久检查点与崩溃恢复、Qwen3-ASR/Whisper 降级、三层专业词库、说话人识别、Transcript V2、异常检测、连续重叠分块、局部重识别、严格结构化精校、精校 SRT/VTT 完整性、候选审核、知识空间策略、来源评估、SOP、便携 Wiki、黄金集评测、外部证据校验、时间轴保真、逐条校订审计、延迟索引质量门禁、播放器、人工校订、证据分层知识提炼与索引桥接，以及分块、原子索引、数据库迁移、知识治理、OCR、公开视频安全、内容寻址归档、隐私分享、备份恢复和桌面产品行为。

仓库还提供本地黄金集评测框架，可计算 Hit Rate@K、MRR、Citation Precision 和 Citation Coverage。先把示例中的文档 ID、知识块 ID 换成自己的入库记录，再运行 `knowledge eval`；加 `--retrieval-only` 可只评估检索。

## 构建桌面应用

macOS：

```bash
./scripts/build_desktop.sh
```

Windows PowerShell：

```powershell
.\scripts\build_windows.ps1
```

正式对外发布 macOS 应用时，还需要 Apple Developer ID。仓库中的 `packaging/sign_and_notarize.sh` 支持签名与公证流程。

GitHub Actions 会在 Linux、macOS 和 Windows 上运行测试，并生成 macOS/Windows 未签名构建产物。公开发行前仍需按 `packaging/RELEASE_SECURITY.md` 完成平台签名、公证、SHA-256 清单和发布验证。

## 项目结构

```text
src/media_knowledge/
├── desktop/       PySide6 应用、模型管理、播放器、转写校订、诊断与更新
├── ingestion/     多模态提取、音频管线、ASR/说话人路由、OCR、质检与归档
├── transcripts/   Transcript V2、质量评估、深度精校、外部证据审计与持久化
├── chunking/      媒体感知分块
├── embedding/     本地语义与兼容 Embedding
├── retrieval/     向量、FTS5、融合与重排
├── qa/            连续问答、证据、提示词与引用
├── storage/       SQLite、知识治理、关系、向量和对话存储
└── sync/          Obsidian 与监听文件夹同步
```

## 使用边界

- 受登录、DRM 或反爬保护的平台链接，只有在取得真实正文或媒体流后才允许入库。
- 公共平台连接器只处理精确白名单域名，不使用 Cookie、浏览器会话、netrc 或代理，也不会绕过登录、地区或权限限制。
- AI 提炼不能替代原始证据，重要页面和原件会继续保留。
- 模型权重不随安装包提供；只有用户在本地模型管理器中明确操作时才会下载。当前版本不支持麦克风采集、实时录音或实时流式转写。
- 本项目不会绕过访问控制；请仅导入你有权处理的资料。
- 当前仓库未附带开源许可证，使用与再分发权限以仓库所有者后续声明为准。

## 参与开发

欢迎提交 Issue 和 Pull Request。修改检索、质检、引用或数据库迁移逻辑时，请同时增加回归测试，并确保 `pytest -q` 全部通过。
