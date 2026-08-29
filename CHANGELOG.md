# Changelog / 更新日志

## 2.2.0 — 2026-08-30

### 中文

- 新增正式知识治理：6 类知识、5 种状态、5 档成熟度和 5 类图关系，兼容旧数据库自动迁移。
- 回答可一键“沉淀为知识”，写入应用自有 Markdown 笔记并与真实来源建立 `supports` 关系。
- 新增知识体检中心，检测孤立、无来源、缺摘要、过期、标签不统一和别名冲突，并给出修复建议。
- OCR 新增行级坐标、置信度、低分行和降级记录；复杂版面可选 PP-StructureV3，并保留 RapidOCR 原始证据。
- PP-StructureV3 按原始阅读顺序保留表格 Markdown、公式和版面坐标，模型在进程内复用，避免多页文档反复加载。
- OCR 平均/最低置信度和低分行比例正式进入入库质检；没有可靠视觉兜底的极低置信结果会被拒绝。
- 音视频转写自动路由可选 Apple MLX、NVIDIA CUDA 和标准包内置 CPU int8；界面明确展示实际路线，并同时产生 JSON/MD/TXT/SRT/VTT。
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
