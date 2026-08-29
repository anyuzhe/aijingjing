# Changelog / 更新日志

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
