# Changelog / 更新日志

## 2.4.0 — 2026-08-30

### 中文

- 新增完整深度语义精校链路：在不可变 `Transcript V2` 原稿之上执行异常扫描、连续重叠分块、跨片段语义校正、术语/实体一致化和保守不确定标注，绝不覆盖 `raw_text`。
- 新增重复循环、低置信、静音幻觉、截断、专业术语和数字单位等异常检测；只对异常时间区间执行可选局部重识别，可在 Whisper Large-v3、Qwen3-ASR 1.7B 与 0.6B 之间切换首轮和复核路线。
- 深度精校严格保留 `segment_id`、时间范围和说话人映射；拒绝缺片段、越界时间、编造定位、伪造引用和结构不完整的 LLM JSON 输出。
- 新增逐条校订审计与人工接受/拒绝：每项保存修改前后、原因、置信度、关联异常和证据；支持进度、取消、检查点恢复，并保持说话人匿名标签和人工身份命名的边界。
- 新增显式联网的外部证据核验。功能默认关闭；开启后网页内容仍视为不可信数据，只接受已注入证据 ID、原 URL 和逐字存在于摘录中的引用。证据不足时标记待核实、听辨不清或 ASR 解码失败，不为流畅度编造事实。
- 新增深度精校 Markdown 产物，可包含完整时间轴、说话人、章节、术语待核实表、校订审计、知识卡片和 Mermaid 知识结构图。
- 跨平台正式包纳入 `sherpa-onnx` 运行时，支持无门禁的双 ONNX 说话人模型登记、SHA-256 校验、纯本地推理与路线诊断；权重仍由用户显式安装。
- 导出稿新增全量“原稿 / 精校稿差异审计”，逐条列出时间、说话人、修改前后、状态、置信度、证据数和理由。
- macOS Apple Silicon 构建脚本现在安装 `full,apple-media`，把 MLX/Qwen 推理依赖纳入正式应用；Intel macOS、Windows 与 Linux 保持跨平台 `full` 依赖。构建不会下载或捆绑任何模型权重。
- 更新中英文 README，明确深度精校的数据流、模型依赖、说话人边界、外部联网行为、原稿不可变原则和云端数据最小化范围。

### English

- Added the complete semantic deep-correction pipeline above immutable `Transcript V2` facts: issue scanning, continuous overlapping chunks, cross-segment semantic correction, term/entity consistency, and conservative uncertainty labels without overwriting `raw_text`.
- Added detection for repetition loops, low confidence, silence hallucinations, truncation, technical terms, and numbers/units. Optional re-recognition reruns only suspicious intervals, with independently selectable Whisper Large-v3, Qwen3-ASR 1.7B, and Qwen3-ASR 0.6B routes for the first and review passes.
- Enforced exact preservation of `segment_id`, time ranges, and speaker mappings. Missing segments, out-of-bounds times, invented locators, fabricated citations, and malformed structured LLM JSON are rejected.
- Added per-change audit and human accept/reject controls. Every proposal retains before/after text, reason, confidence, detected issues, and evidence, with progress, cancellation, checkpoint resume, anonymous speaker labels, and explicit human identity naming.
- Added opt-in external-evidence verification, disabled by default. Web content remains untrusted data, and citations must match an injected evidence ID, original URL, and verbatim snippet quote. Insufficient support produces needs-verification, inaudible, or ASR-decode-failure markers rather than invented facts.
- Added deep-correction Markdown artifacts with complete timelines, speakers, chapters, unresolved-term tables, correction audits, knowledge cards, and Mermaid knowledge diagrams.
- Added the `sherpa-onnx` runtime to portable release builds, with dual-ONNX bundle registration, SHA-256 verification, fully local inference, and route diagnostics. Model weights remain an explicit user installation.
- Added a complete raw-versus-corrected audit table to Markdown exports, including time, speaker, before/after text, decision, confidence, evidence count, and rationale for every proposal.
- macOS Apple Silicon builds now install `full,apple-media` so official packages contain the MLX/Qwen inference dependencies. Intel macOS, Windows, and Linux retain the portable `full` extra. Builds neither download nor bundle model weights.
- Expanded both READMEs with the correction data flow, model dependencies, speaker boundaries, explicit network behavior, immutable-source policy, and cloud data-minimization boundary.

## 2.3.0 — 2026-08-30

### 中文

- 新增面向中文音频的三档本地转写方案：`Qwen3-ASR 1.7B` 中文高精度、`Qwen3-ASR 0.6B` 快速预览，以及 Whisper 兼容模式；也可自定义 Qwen3-ASR、MLX Whisper 或 faster-whisper 路线。
- 新增显式本地模型管理器，可查看安装状态与体积、登记已有目录、复制导入、由用户主动下载或移除应用管理的模型；模型状态检查和音视频导入不会触发静默联网下载。
- 本地模型完整性升级为真实文件内容 SHA-256，并把模型内容标识、三层专业词库版本及实际回退路线写入转写事实和持久检查点。
- 安装包只包含可用的推理运行时与应用代码，不携带 Qwen3-ASR、Whisper 或说话人模型权重。模型下载必须由用户在模型管理器中明确发起；受限模型仍需用户接受其上游条款。
- 音视频转写前新增解码预检和音轨诊断，记录编码、采样率、声道、时长、音量、静音比例与削波风险；随后原子标准化为 16 kHz、单声道、PCM16，并执行本地 VAD 生成语音区间。
- 新增 ASR Provider 路由与可追溯降级：Apple Silicon 可选 Qwen3-ASR/MLX Whisper，NVIDIA/CPU 可用 faster-whisper；每次尝试、设备、模型、回退原因与取消状态均显式记录，取消不会触发回退。
- 转写主事实升级为 `Transcript V2`，同时保存原始识别文字、人工校订文字、句段/词级时间戳、运行配置、语言、模型、回退历史、说话人、质量状态与来源摘要，并兼容读取 V1。
- 新增可选本地说话人识别与对齐，支持 pyannote Community-1 和 Sherpa-ONNX 路由、人数范围、匿名说话人、重叠讲话标记，以及后续人工命名、重新分配和合并。
- 转写质量门禁新增空结果、时间倒序/越界、重复循环、截断、静音幻觉、异常语速、语言/专业术语/数字单位风险、说话人数偏差、未知说话人和碎片化检查；需要复核或失败的转写不会进入高可信问答索引。
- 新增七阶段原子持久检查点与崩溃恢复；需要复核或失败的转写先保存事实但延迟 FTS/向量索引，人工批准后再原子补建。
- 新增全局、知识空间和单一来源三层专业词库；AI 知识提炼严格区分事实、推测、争议、决策和行动项，并验证所有页码/时间戳来自原始资料。
- 新增应用内音视频播放器，可从引用或转写片段跳转到精确时间点、同步高亮时间轴、切换播放速度，并在本地打开原始媒体。
- 新增转写人工校订界面：原始识别文字保持只读，校订稿、修改原因、说话人命名/归属/合并均写入审计记录；保存后原子重建说话人感知的全文与语义索引。
- 当前版本只处理用户选择或导入的现有音视频文件，不包含麦克风采集、实时录音或实时流式转写。
- 回归覆盖扩展到 300 项自动化测试和 42 项额外子测试，覆盖 ASR 路由、本地模型生命周期、音频管线、断点恢复、专业词库、说话人、Transcript V2、质量门禁、播放器、校订、证据分层整理与索引桥接。

### English

- Added three local transcription profiles for Chinese audio: `Qwen3-ASR 1.7B` for high accuracy, `Qwen3-ASR 0.6B` for fast previews, and a Whisper compatibility profile, plus custom Qwen3-ASR, MLX Whisper, and faster-whisper routes.
- Added an explicit local model manager for inspecting installation state and size, registering an existing directory, copying a local model into managed storage, downloading only after a user action, and removing app-managed models. Status checks and media ingestion never initiate a silent network download.
- Upgraded model integrity to SHA-256 over actual model bytes and retained model-content identity, scoped-glossary version, and the effective fallback route in transcript facts and persistent checkpoints.
- Application packages contain compatible inference runtimes and application code, but no Qwen3-ASR, Whisper, or diarization model weights. A download must be explicitly started in the model manager, and gated models still require acceptance of their upstream terms.
- Added audio decode preflight and diagnostics before transcription, recording codec, sample rate, channels, duration, loudness, silence ratio, and clipping risk. Audio is then atomically normalized to 16 kHz mono PCM16 and processed by a local VAD to identify speech intervals.
- Added an auditable ASR Provider router: Apple Silicon can use Qwen3-ASR or MLX Whisper, while NVIDIA/CPU systems can use faster-whisper. Every attempt, device, model, fallback reason, and cancellation outcome is recorded; cancellation never triggers fallback.
- Promoted `Transcript V2` to the canonical transcript fact format, preserving immutable raw recognition, human corrections, segment/word timestamps, run configuration, language, model, fallback history, speakers, quality state, and source digest while retaining V1 read compatibility.
- Added optional local speaker diarization and alignment through pyannote Community-1 or Sherpa-ONNX routing, speaker-count bounds, anonymized speaker IDs, overlap markers, and subsequent human rename, reassignment, and merge operations.
- Expanded the transcript quality gate with checks for empty output, reversed/out-of-range timestamps, repeated generation loops, truncation, speech hallucinated over silence, abnormal character rate, language/technical-term/number-unit risks, speaker-count mismatch, unknown speakers, and fragmentation. Review/fail transcripts are excluded from the high-trust Q&A index.
- Added seven-stage atomic checkpoints and crash recovery. Review/fail transcripts retain facts while FTS/vector indexing is deferred until a human approval atomically builds it.
- Added global, knowledge-space, and source-scoped technical glossaries. Derived synthesis separates facts, inferences, disputes, decisions, and actions and validates every page/timestamp locator against source evidence.
- Added an in-app audio/video player that seeks from citations or transcript segments to an exact timestamp, highlights the active cue, changes playback speed, and can open the original local media.
- Added an audited transcript editor: immutable raw recognition stays read-only, while corrected text, edit reasons, speaker names, assignments, and merges are recorded. Saving atomically rebuilds the speaker-aware full-text and semantic index.
- This release processes existing audio/video files selected or imported by the user. It does not provide microphone capture, live recording, or real-time streaming transcription.
- Expanded regression coverage to 300 automated tests and 42 additional subtests across ASR routing, local model lifecycle, audio preparation, crash recovery, scoped glossaries, diarization, Transcript V2, quality gating, playback, correction, evidence-layered synthesis, and indexing bridges.

## 2.2.0 — 2026-08-30

### 中文

- 新增正式知识治理：6 类知识、5 种状态、5 档成熟度和 5 类图关系，兼容旧数据库自动迁移。
- 回答可一键“沉淀为知识”，写入应用自有 Markdown 笔记并与真实来源建立 `supports` 关系。
- 新增知识体检中心，检测孤立、无来源、缺摘要、过期、标签不统一和别名冲突，并给出修复建议。
- OCR 新增行级坐标、置信度、低分行和降级记录；复杂版面可选 PP-StructureV3，并保留 RapidOCR 原始证据。
- PP-StructureV3 按原始阅读顺序保留表格 Markdown、公式和版面坐标，模型在进程内复用，避免多页文档反复加载。
- OCR 平均/最低置信度和低分行比例正式进入入库质检；没有可靠视觉兜底的极低置信结果会被拒绝。
- 音视频转写自动路由可选 Apple MLX、NVIDIA CUDA 和标准包内置 CPU int8；界面明确展示实际路线，并同时产生 JSON/MD/TXT/SRT/VTT。
- Apple Silicon 正式包现已内置 MLX/Metal 运行时；设置界面标明各档模型取舍，并可直接使用最高精度的 Whisper large-v3。
- MLX 转写改为可终止的隔离子进程，点击停止后会回收推理进程，不再等待整段音频完成。
- 新增转写完整性门禁，检查倒序、越界、异常重叠、空段和首尾覆盖。
- 新增 YouTube、B 站、抖音、小红书和 X 公开链接连接器；优先保留公开字幕，不使用 Cookie、netrc 或代理。
- 公开媒体下载增加流式 2GB 硬限制、下载前大小预检、直播/未完成回放与不可监控协议拒绝；仅允许 HTTP/HTTPS 和原生 DASH，HLS 在下载前安全拒绝，避免第三方内部回退到无界 FFmpeg 下载。
- 下载源、字幕、转写与关键帧先由受控缓存临时持有，只有质检通过才用真实 SHA-256 内容寻址并原子发布；失败、取消、重复和索引异常会精确回滚本次新建证据及派生产物，不覆盖或删除既有成功证据。
- 归档包同时使用原始证据摘要与规范化解析摘要；同一原件在 OCR/转写/解析升级后生成可追溯的新版本，`bundle.json` 不再与数据库知识错位。
- 临时下载清理改为有限重试和带 marker、设备号、inode 校验的持久登记；服务启动时自动重试，未验证路径绝不删除，清理错误会显示但不会覆盖原始导入错误。
- Embedding 在数据库提交前完成，文档、知识项、分类、分块、来源、FTS 和向量在同一个事务写入，模型失败不会留下指向已回滚文件的半成品记录。
- 新增本地隐私扫描与安全分享副本：报告脱敏、默认空选择、固定排除私密数据、前后二次扫描、原子发布和逐文件 SHA-256。
- 安全分享取消所有未检查内容的专家绕过：PDF/Office 继续深度扫描并生成脱敏报告，但原始 PDF、Office/ODF、图片和音视频容器一律不原样复制；只发布通过全文件严格文本校验的资料，阻断不可达对象、压缩尾部和隐藏元数据。
- 图片 OCR 无引擎或空结果不再误判为安全，安全分享必须进入人工复核。
- 正式知识新增可恢复回收站：删除前保存 tombstone、笔记、标签、别名和关系，可从界面恢复原 ID 与仍有效的图关系。
- 数据库级后台操作统一互斥，备份恢复、索引修复、同步、安全分享、搜索、导入和回答不再发生危险重叠。
- 修复带点文件名的转写产物覆盖，并保证公开平台字幕来源始终归档到真实存在的本地证据文件。
- 多媒体设置和系统诊断可见实际 OCR/转写引擎、设备路由和缺失组件；导入任务显示中文阶段与可恢复错误。

### English

- Added governed knowledge with six types, five lifecycle states, five maturity levels, five relation types, and automatic legacy-database migration.
- Added one-click answer promotion into owned Markdown knowledge notes with real `supports` provenance edges.
- Added an actionable knowledge health center for orphaned, source-less, incomplete, stale, inconsistent, and ambiguous items.
- Added line-level OCR geometry/confidence, optional PP-StructureV3 complex-layout analysis, and preserved RapidOCR baseline evidence.
- Preserved PP-StructureV3 reading order, Markdown tables, formulas, and geometry while reusing one guarded pipeline per process.
- Added OCR confidence and low-confidence ratios to the ingestion gate; extremely unreliable OCR without real vision evidence is rejected.
- Added explicit routing across optional Apple MLX, NVIDIA CUDA, and the standard bundle's built-in CPU int8 runtime, with JSON/MD/TXT/SRT/VTT artifacts and integrity gates.
- Official Apple Silicon bundles now include the MLX/Metal runtime, while settings explain each model tier and expose Whisper large-v3 as the highest-accuracy option.
- Isolated MLX inference in a terminable worker so cancellation reclaims the process immediately.
- Added a subtitle-first public connector for YouTube, Bilibili, Douyin, Xiaohongshu, and X without cookies, netrc, or proxies.
- Added a streaming 2 GB ceiling, preflight size checks, and rejection of live, unfinished, or unmonitorable transports. Only HTTP/HTTPS and native DASH are accepted; HLS is rejected before download to prevent internal fallback to an unbounded FFmpeg downloader.
- Kept downloaded sources, subtitles, transcripts, and keyframes in explicitly owned cache until quality acceptance, then published them with real SHA-256 content-addressed paths. Failure, cancellation, duplicate detection, and indexing errors roll back only newly created evidence and derived artifacts.
- Versioned archive packages by both raw-evidence and normalized parse digests, so parser/OCR/transcription changes preserve an auditable new bundle instead of leaving SQLite ahead of stale archived metadata.
- Added bounded cleanup retries plus a persistent marker/device/inode-verified registry. Startup retries verified leftovers, never deletes an unverified path, and reports cleanup failures without masking the original error.
- Moved embedding before commit and made documents, governed items, facets, chunks, provenance, FTS, and vectors one transaction, so provider failure cannot leave a row pointing at rolled-back evidence.
- Added redacted local privacy scanning and empty-by-default safe share copies with fixed exclusions, two scans, atomic publication, and per-file SHA-256 manifests.
- Removed every expert bypass for uninspected share content. PDF/Office containers still receive deep inspection and redacted reports, but original PDF, Office/ODF, image, audio, and video containers are never copied verbatim; only strictly validated text-like files can be published, closing unreachable-object, compressed-tail, and hidden-metadata channels.
- Empty or unavailable image OCR can no longer produce a clean safe-sharing decision.
- Added recoverable governed-knowledge trash with durable tombstones for notes, aliases, tags, and relations, plus UI restoration of the original item ID and still-valid graph edges.
- Made database-wide desktop operations mutually exclusive so restore, repair, sync, sharing, search, ingestion, and answer generation cannot overlap unsafely.
- Prevented transcript collisions for dotted basenames and guaranteed that public-platform subtitle sources resolve to existing archived evidence.
- Added multimedia settings, route-aware diagnostics, and clearer Chinese ingestion stages and recovery messages.

## 2.1.0 — 2026-08-29

### 中文

- 新增可搜索、恢复、重命名、删除和 Markdown 导出的历史对话。
- 新增真实流式回答、停止生成、复制、重新生成和回答反馈。
- 新增聚焦检索、小文档全文和长文档分层摘要三种自适应 RAG 策略。
- 用结构化证据质量替代误导性的“正确率”：展示引用覆盖、证据利用率、来源多样性和判断依据。
- 将召回内容隔离为不可信数据，并增加提示词注入检测与防护。
- 持久化导入批次和逐文件进度；异常退出后可继续，失败项可重试。
- 升级 V2 完整备份，覆盖数据库、设置、笔记、归档、资源和转写，并逐文件校验 SHA-256。
- 更新链路增加严格 SemVer、HTTPS 重定向、流式大小限制和安装包 SHA-256 校验。
- 新增本地黄金集评测命令，以及 Linux/macOS/Windows CI 和 macOS/Windows 构建任务。
- 桌面界面新增对话页、证据解释、任务恢复、键盘快捷键与无障碍名称。

### English

- Added searchable, reopenable, renameable, deletable, and Markdown-exportable conversation history.
- Added real answer streaming, stop, copy, regenerate, and local answer feedback.
- Added adaptive focused, small-document full-context, and long-document hierarchical RAG strategies.
- Replaced misleading confidence percentages with structured evidence quality and explanations.
- Isolated retrieved sources as untrusted data and added prompt-injection detection and defenses.
- Persisted import batches and per-file progress for crash recovery and failed-item retry.
- Introduced V2 full backups for the database, settings, notes, archives, assets, and transcripts with per-file SHA-256.
- Hardened updates with strict SemVer, HTTPS redirect validation, streaming size limits, and package SHA-256 verification.
- Added a local golden-set evaluation command, cross-platform CI, and macOS/Windows packaging jobs.
- Improved desktop conversation, evidence, task-recovery, keyboard, and accessibility interactions.

## 2.0.5

- Added clipboard/drag image attachments, multimodal chat payloads, and follow-up image reuse.
- Added DeepSeek Vision model selection and the winter-blue AI静静 desktop identity.
