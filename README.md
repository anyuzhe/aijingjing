<div align="center">
  <img src="packaging/AI-Jingjing.png" width="128" alt="AI静静 Logo">
  <h1>AI知识库 · AI静静</h1>
  <p><strong>本地优先、多模态、可溯源的个人知识库桌面应用</strong></p>
  <p>简体中文 · <a href="README_EN.md">English</a></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/UI-PySide6-41CD52" alt="PySide6">
    <img src="https://img.shields.io/badge/Storage-SQLite%20FTS5-0F80CC" alt="SQLite FTS5">
    <img src="https://img.shields.io/badge/Test-95%20passed-2F855A" alt="95 tests passed">
    <img src="https://img.shields.io/badge/Version-2.1.0-4C8FBF" alt="Version 2.1.0">
  </p>
</div>

AI静静把 PDF、PPT、Word、图片、音频、视频、网页和 Markdown 统一整理成一套可以搜索、连续提问、查看原始证据并长期维护的本地知识库。

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
- 音视频生成时间轴转写；视频还可抽取关键帧。
- 同名 PPT、PDF、录音和视频自动归为同一个 `Source Package`。
- 自动归档原件、网页快照、转写、页面资源和解析清单。
- 摄取流程是正式 Python 服务，不调用本地 Codex CLI。

### 2. 严格入库质检

每份资料在写入知识库前都会检查：

- 是否取得真实正文或真实媒体流；
- 是否只有标题、简介、封面或平台元数据；
- PDF/PPT 页面覆盖率；
- 音视频是否产生真实转写或画面理解；
- 原始文件校验值、正文规模和解析告警。

只开放说明和封面的受限视频链接会被明确拒绝，不会把简介伪装成视频内容入库。

### 3. 本地语义检索

- SQLite FTS5 全文检索；
- 本地中英文语义 Embedding；
- 向量召回与 BM25 召回融合；
- RRF 融合与本地重排；
- 按最终相关性自动排序并剔除明显无关的候选片段；
- 按知识空间、标签、资料类型或指定文档限定范围；
- 展示融合分数、语义分数和全文命中状态。

检索过程不调用大模型。首次使用会下载约 240 MB 的多语言 ONNX 模型，之后可以离线运行。

### 4. 连续对话与可信引用

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

### 5. 原文阅读与资料管理

- 在应用内阅读 PDF 页面、图片、解析文本与音视频时间轴；
- 查看每份资料的原始知识块和定位信息；
- 对证据添加批注并绑定到具体知识块；
- 重命名、停用、重新启用、重新解析或移除资料；
- 编辑知识空间和标签；
- 检查内容指纹相同的重复资料；
- 原始归档默认保留，移除索引不会销毁原文件。

### 6. 自动同步与知识工坊

- 监听一个或多个文件夹并增量入库；
- 每批导入及每个文件的阶段、进度、错误和结果都会持久化；
- 程序异常退出后，未完成任务恢复为“可继续”，失败项可单独重试；
- 文件变化后只重新解析变化项；
- 源文件消失时先停用对应知识，不静默删除；
- 可选从 Obsidian 增量同步，或把 AI静静笔记导出至 Obsidian；
- 基于选定资料生成综合报告、多资料比较、时间线、测验、复习闪卡和思维导图。

### 7. 本地数据安全

- 知识、索引、对话、答案和引用保存在本机 SQLite；
- API Key 优先保存在 macOS Keychain 或系统凭据存储；
- 备份包明确排除 API Key；
- V2 完整备份覆盖数据库、设置、笔记、原始归档、页面资源和转写，并为每个文件记录大小与 SHA-256；
- 恢复前先验证路径、压缩比、大小、哈希、SQLite 和设置，再创建当前数据安全快照并原子恢复；
- 更新清单和下载重定向只接受 HTTPS；安装包下载完成后必须通过清单中的 SHA-256 才允许打开；
- 本地文件路径、数据库和个人知识目录均由 `.gitignore` 排除。

## 支持的输入

| 类型 | 扩展名/形式 | 主要处理 | 可追溯位置 |
|---|---|---|---|
| Markdown / 文本 | `.md` `.txt` `.csv` `.json` `.yaml` | 标题结构、段落、代码与表格 | 标题路径、字符范围 |
| Word | `.docx` | 段落、标题、表格 | 章节与原文件 |
| PDF | `.pdf` | 逐页文本、扫描页 OCR、页面图像 | 页码 |
| PowerPoint | `.pptx` | 逐页文字、备注、图片与页面结构 | 幻灯片页码 |
| 图片 | `.png` `.jpg` `.webp` `.tiff` 等 | OCR 与可选视觉理解 | 原图与图片资源 |
| 音频 | `.mp3` `.m4a` `.wav` `.flac` 等 | Whisper 转写 | 开始/结束时间 |
| 视频 | `.mp4` `.mov` `.mkv` `.webm` 等 | FFmpeg、Whisper、关键帧 | 时间轴与关键帧 |
| 网页/媒体链接 | `https://...` | 正文抓取、网页快照或真实媒体流下载 | URL、快照、时间轴 |
| 微信公众号文章 | `mp.weixin.qq.com/s/...` | 专用正文与标题提取，验证页拦截 | 文章 URL、正文快照 |

## 架构

```text
PySide6 桌面界面
        │
        ├── IngestionService
        │   ├── 文档/PDF/PPT/图片解析
        │   ├── OCR / Whisper / FFmpeg
        │   ├── 入库真实性与完整性质检
        │   └── Source Package 归档
        │
        ├── KnowledgeDatabase (SQLite)
        │   ├── Documents / Chunks / Source References
        │   ├── FTS5 全文索引 / Embeddings
        │   └── Conversations / Evidence / Citations
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

### 配置回答模型

启动应用后打开“设置 → 模型与隐私”，填写 DeepSeek 或 Kimi API Key。密钥不会显示在状态接口、诊断日志、SQLite 或备份中。

不配置任何 API Key 时，入库、本地搜索和离线证据回答仍然可用。

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
├── notes/         Source Notes、AI 回答与知识工坊产物
├── assets/        PDF/PPT 页面、图片与视频关键帧
├── transcripts/   音视频转写
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
| 本地证据回答 | 否 | 无 |
| DeepSeek/Kimi 回答 | 是 | 问题、有限对话上下文、召回的证据块，以及用户主动附加的图片 |
| DeepSeek 知识提炼 | 是 | 当前导入资料的有界提取文本 |
| DeepSeek Vision/Kimi 视觉理解 | 是 | 用户启用后由导入策略选中的有限图片 |

模型服务不会收到整个知识数据库、归档目录或 Obsidian Vault。

## 测试

```bash
pip install pytest
pytest -q
```

当前版本包含 95 项自动化测试，覆盖分块、数据库迁移、模型选择、真实流式输出、图片粘贴与多模态请求、图片后续追问、自适应上下文、证据质量、提示注入防护、连续对话历史、回答反馈、持久化导入任务、引用、微信文章抓取与验证页拦截、V1/V2 备份恢复、安全更新和桌面产品行为。

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
├── desktop/       PySide6 应用、控制器、诊断与更新
├── ingestion/     多模态提取、视觉理解、质检与归档
├── chunking/      媒体感知分块
├── embedding/     本地语义与兼容 Embedding
├── retrieval/     向量、FTS5、融合与重排
├── qa/            连续问答、证据、提示词与引用
├── storage/       SQLite、向量和对话存储
└── sync/          Obsidian 与监听文件夹同步
```

## 使用边界

- 受登录、DRM 或反爬保护的平台链接，只有在取得真实正文或媒体流后才允许入库。
- AI 提炼不能替代原始证据，重要页面和原件会继续保留。
- 本项目不会绕过访问控制；请仅导入你有权处理的资料。
- 当前仓库未附带开源许可证，使用与再分发权限以仓库所有者后续声明为准。

## 参与开发

欢迎提交 Issue 和 Pull Request。修改检索、质检、引用或数据库迁移逻辑时，请同时增加回归测试，并确保 `pytest -q` 全部通过。
