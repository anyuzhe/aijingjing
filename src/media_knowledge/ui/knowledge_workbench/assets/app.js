const state = {
  mode: "ask",
  bootstrap: null,
  busy: false,
  conversationId: localStorage.getItem("knowledge.conversationId"),
  evidence: [],
  selectedEvidenceId: null,
  answerMode: "knowledge",
  skillName: "",
  skillSources: [],
  filters: {
    collections: new Set(), tags: new Set(), media_types: new Set(), folders: new Set(), document_ids: new Set(),
  },
};
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const MEDIA_TYPE_LABELS = {
  pdf: "PDF",
  presentation: "演示文稿",
  ppt: "演示文稿",
  pptx: "演示文稿",
  audio: "音频",
  video: "视频",
  image: "图片",
  markdown: "Markdown",
  text: "文本",
  document: "文档",
  docx: "Word 文档",
  web: "网页",
};

function mediaTypeLabel(value) {
  return MEDIA_TYPE_LABELS[String(value || "").toLowerCase()] || value || "资料";
}

function modelLabel(value) {
  const configured = state.bootstrap?.capabilities.models?.find(
    (item) => item.id === value || (item.id !== "auto" && item.model === value)
  );
  if (configured) return configured.label;
  if (value === "saved") return "已保存回答";
  if (value === "grounded-extractive-v1") return "本地证据模型";
  if (value === "codex-auto") return "Codex 自动模型";
  if (value === "gpt-5.6-luna") return "Codex 极速模型";
  return value;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[char]);
}

function filterSection(label, type, items, formatter = (item) => item.name) {
  if (!items.length) return "";
  return `<section class="filter-section"><div class="section-label">${escapeHtml(label)}</div>${items.slice(0, 8).map((item) => `
    <label class="filter-row"><input type="checkbox" data-filter-type="${type}" data-filter-value="${escapeHtml(item.value ?? item.name)}" />
      <span>${escapeHtml(formatter(item))}</span><small>${item.count}</small>
    </label>`).join("")}</section>`;
}

function renderKnowledgeSnapshot(data) {
  $("#document-count").textContent = data.stats.documents;
  $("#index-status").textContent = `${data.stats.documents} 份文档 · ${data.stats.chunks} 个证据片段`;
  $("#filter-sections").innerHTML = [
    filterSection("收藏集", "collections", data.collections), filterSection("文件夹", "folders", data.folders),
    filterSection("标签", "tags", data.tags), filterSection("媒体类型", "media_types", data.media_types, (item) => mediaTypeLabel(item.name)),
  ].join("") || `<div class="recent-item muted">录入资料后即可使用范围筛选。</div>`;
  $("#recent-documents").innerHTML = data.recent_documents.length ? data.recent_documents.slice(0, 6).map((doc) => `
    <button class="recent-document" data-document-id="${escapeHtml(doc.id)}"><span class="media-badge">${escapeHtml(doc.media_type.slice(0, 3).toUpperCase())}</span>
      <span><strong>${escapeHtml(doc.title)}</strong><small>${escapeHtml(mediaTypeLabel(doc.media_type))}</small></span></button>`).join("") : `<div class="recent-item muted">还没有录入文档。</div>`;
  bindScopeControls();
}

function renderConversationHistory(data) {
  $("#conversation-history").innerHTML = data.conversations.length ? data.conversations.map((item) => `
    <button class="history-item" data-conversation-id="${escapeHtml(item.id)}">
      <strong>${escapeHtml(item.title || "未命名对话")}</strong>
      <small>${escapeHtml(item.preview || "")}</small>
    </button>`).join("") : `<div class="recent-item muted">还没有保存的对话。</div>`;
  $$("[data-conversation-id]").forEach((button) => button.addEventListener("click", () => loadConversation(button.dataset.conversationId)));
}

function renderSyncCapability(data) {
  const sync = data.capabilities?.obsidian_sync;
  const button = $("#sync-knowledge");
  button.disabled = !sync?.available || Boolean(sync?.running);
  button.textContent = sync?.running ? "同步中……" : "同步知识库";
  const last = sync?.last_result;
  button.title = last
    ? `上次同步：新增 ${last.created || 0}、更新 ${last.updated || 0}、删除 ${last.deleted || 0}`
    : "把 Obsidian Markdown 增量同步到知识索引";
}

async function loadBootstrap() {
  try {
    const response = await fetch("/api/bootstrap");
    if (!response.ok) throw new Error("本地索引暂不可用");
    state.bootstrap = await response.json();
    const data = state.bootstrap;
    renderKnowledgeSnapshot(data);
    renderConversationHistory(data);
    renderSyncCapability(data);
    $("#web-mode").disabled = !data.capabilities.web_search;
    const modelSelect = $("#model-select");
    modelSelect.innerHTML = (data.capabilities.models || []).map((model) =>
      `<option value="${escapeHtml(model.id)}" title="${escapeHtml(model.description)}">${escapeHtml(model.label)}</option>`
    ).join("");
    modelSelect.value = data.capabilities.default_model || data.capabilities.models?.[0]?.id || "local-extractive";
    updateModelHint();
    const ingestSkill = (data.capabilities.skills || []).find((item) => item.name === "knowledge-ingestor");
    const skillOption = $('#skill-select option[value="knowledge-ingestor"]');
    skillOption.disabled = !ingestSkill?.available;
    skillOption.textContent = ingestSkill?.available ? "知识摄取" : "知识摄取（不可用）";
    $("#skill-select").title = ingestSkill?.available ? ingestSkill.description : "本地 Skill 或 Codex 执行环境不可用";
    if (state.conversationId) await loadConversation(state.conversationId, false);
  } catch (error) {
    $("#index-status").textContent = error.message;
    $("#filter-sections").innerHTML = `<div class="recent-item muted">请启动本地索引以加载筛选项。</div>`;
  }
}

function setMode(mode) {
  if (mode === "search" && state.skillName) setSkill("");
  state.mode = mode;
  $$(".mode-button").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  $("#workspace-title").textContent = mode === "search" ? "搜索原始资料" : "向知识档案提问";
  $("#prompt-input").placeholder = mode === "search" ? "搜索原始知识……" : "向知识库提问……";
  updatePrimaryAction();
  document.body.dataset.workspaceMode = mode;
}

function updatePrimaryAction() {
  $(".send-button span:first-child").textContent = state.skillName ? "运行 Skill" : (state.mode === "search" ? "搜索" : "发送");
}

function updateModelHint() {
  const selected = state.bootstrap?.capabilities.models?.find((item) => item.id === $("#model-select").value);
  $("#model-hint").textContent = selected?.description || "选择用于当前回答的模型";
  $("#model-select").title = selected?.description || "";
}

function setSkill(name) {
  state.skillName = name || "";
  if (state.skillName) state.mode = "ask";
  $("#skill-select").value = state.skillName;
  $("#skill-context").hidden = !state.skillName;
  document.body.dataset.skillActive = state.skillName ? "true" : "false";
  $("#knowledge-mode").disabled = Boolean(state.skillName);
  $("#web-mode").disabled = Boolean(state.skillName) || !state.bootstrap?.capabilities.web_search;
  $("#deep-analysis").disabled = Boolean(state.skillName);
  $("#model-select").disabled = Boolean(state.skillName);
  if (state.skillName) {
    $$('.mode-button').forEach((button) => button.classList.toggle("active", button.dataset.mode === "ask"));
    document.body.dataset.workspaceMode = "ask";
    $("#workspace-title").textContent = "使用 Skill 整理资料";
    $("#prompt-input").placeholder = "描述要整理入库的内容，也可以粘贴 URL……";
  } else {
    state.skillSources = [];
    renderSkillFiles();
    $("#workspace-title").textContent = state.mode === "search" ? "搜索原始资料" : "向知识档案提问";
    $("#prompt-input").placeholder = state.mode === "search" ? "搜索原始知识……" : "向知识库提问……";
  }
  updateScopeSummary();
  updatePrimaryAction();
}

function renderSkillFiles() {
  $("#skill-files").innerHTML = state.skillSources.length
    ? state.skillSources.map((path, index) => `<span class="skill-file-chip" title="${escapeHtml(path)}">${escapeHtml(path.split(/[\\/]/).pop())}<button type="button" data-remove-skill-file="${index}" aria-label="移除文件">×</button></span>`).join("")
    : "未选择本地文件，可直接在任务中粘贴文字或 URL。";
}

function bindScopeControls() {
  $$("[data-filter-type]").forEach((input) => input.addEventListener("change", () => {
    const target = state.filters[input.dataset.filterType];
    input.checked ? target.add(input.dataset.filterValue) : target.delete(input.dataset.filterValue);
    updateScopeSummary();
  }));
  $$("[data-document-id]").forEach((button) => button.addEventListener("click", () => {
    const id = button.dataset.documentId;
    state.filters.document_ids.has(id) ? state.filters.document_ids.delete(id) : state.filters.document_ids.add(id);
    button.classList.toggle("selected", state.filters.document_ids.has(id)); updateScopeSummary();
  }));
  $$("[data-conversation-id]").forEach((button) => button.addEventListener("click", () => loadConversation(button.dataset.conversationId)));
}

function clearScope() {
  Object.values(state.filters).forEach((items) => items.clear());
  $$("[data-filter-type]").forEach((input) => { input.checked = false; });
  $$("[data-document-id]").forEach((button) => button.classList.remove("selected"));
  updateScopeSummary();
}

function scopePayload() {
  return Object.fromEntries(Object.entries(state.filters).map(([key, value]) => [key, [...value]]));
}

function updateScopeSummary() {
  if (state.skillName) {
    $("#scope-summary").textContent = "Skill 模式 · 可处理所选文件、URL 或粘贴内容";
    $("#all-knowledge").classList.remove("selected");
    return;
  }
  const labels = [];
  const typeLabels = { collections: "收藏集", tags: "标签", media_types: "类型", folders: "文件夹" };
  Object.entries(state.filters).forEach(([type, items]) => items.forEach((item) => {
    const value = type === "media_types" ? mediaTypeLabel(item) : item;
    labels.push(type === "document_ids" ? "已选 1 份文档" : `${typeLabels[type]}：${value}`);
  }));
  $("#scope-summary").textContent = labels.length ? `检索范围 · ${labels.join(" · ")}` : "正在检索全部知识";
  $("#all-knowledge").classList.toggle("selected", labels.length === 0);
}

function showToast(message, isError = false) {
  const toast = $("#toast"); toast.textContent = message; toast.classList.toggle("error", isError);
  toast.classList.add("visible"); clearTimeout(showToast.timer); showToast.timer = setTimeout(() => toast.classList.remove("visible"), 2600);
}

function inlineMarkdown(value) {
  const links = [];
  const withTokens = String(value).replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, (_, label, url) => {
    const token = `@@LINK${links.length}@@`; links.push({ label, url }); return token;
  });
  let html = escapeHtml(withTokens);
  html = html.replace(/\[S(\d+)\]/g, (_, number) => `<button class="citation-marker" data-citation="S${number}" title="查看证据 S${number}">[${number}]</button>`);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  links.forEach((link, index) => {
    html = html.replace(`@@LINK${index}@@`, `<a href="${escapeHtml(link.url)}" target="_blank" rel="noopener">${escapeHtml(link.label)}</a>`);
  });
  return html;
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").split("\n");
  let html = "", inCode = false, codeLanguage = "", code = [], list = [];
  const closeList = () => { if (list.length) { html += `<ul>${list.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>`; list = []; } };
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (line.startsWith("```")) {
      closeList();
      if (!inCode) { inCode = true; codeLanguage = line.slice(3).trim(); code = []; }
      else { html += `<div class="code-block"><span>${escapeHtml(codeLanguage || "代码")}</span><pre><code>${escapeHtml(code.join("\n"))}</code></pre></div>`; inCode = false; }
      continue;
    }
    if (inCode) { code.push(line); continue; }
    if (/^\s*[-*]\s+/.test(line)) { list.push(line.replace(/^\s*[-*]\s+/, "")); continue; }
    closeList();
    const next = lines[i + 1] || "";
    if (line.includes("|") && /^\s*\|?\s*:?-+/.test(next)) {
      const headers = line.replace(/^\||\|$/g, "").split("|"); i += 1; const rows = [];
      while (i + 1 < lines.length && lines[i + 1].includes("|")) { rows.push(lines[++i].replace(/^\||\|$/g, "").split("|")); }
      html += `<div class="table-wrap"><table><thead><tr>${headers.map((cell) => `<th>${inlineMarkdown(cell.trim())}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell.trim())}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    } else if (/^#{1,4}\s/.test(line)) {
      const level = line.match(/^#+/)[0].length; html += `<h${level + 1}>${inlineMarkdown(line.replace(/^#{1,4}\s+/, ""))}</h${level + 1}>`;
    } else if (line.trim()) { html += `<p>${inlineMarkdown(line)}</p>`; }
  }
  closeList();
  if (inCode) html += `<div class="code-block"><pre><code>${escapeHtml(code.join("\n"))}</code></pre></div>`;
  return html;
}

function welcomeMarkup() {
  return `<div class="welcome-card"><span class="welcome-index">01</span><h2>问问你的知识档案记住了什么。</h2>
    <p>回答始终以选定证据为依据，每一页、每张幻灯片和每个时间点都可以追溯。</p>
    <div class="prompt-grid"><button class="prompt-chip">综合所有资料，FAST-LIVO2 是否需要硬件同步？</button>
    <button class="prompt-chip">比较我保存的两种工程方案</button><button class="prompt-chip">上周整理的课程里有哪些关键结论？</button></div></div>`;
}

function bindPromptChips() {
  $$(".prompt-chip").forEach((button) => button.addEventListener("click", () => { $("#prompt-input").value = button.textContent.trim(); $("#prompt-input").focus(); }));
}

function appendUser(content) {
  const element = document.createElement("article"); element.className = "message user-message";
  element.innerHTML = `<div class="message-label">你</div><div class="user-content">${escapeHtml(content)}</div>`;
  $("#conversation").append(element); scrollConversation();
}

function appendSkillUser(content, sources = []) {
  const element = document.createElement("article"); element.className = "message user-message";
  const sourceText = sources.length ? `<small class="user-skill-sources">${sources.map((path) => escapeHtml(path.split(/[\\/]/).pop())).join(" · ")}</small>` : "";
  element.innerHTML = `<div class="message-label skill-message-label">你 · <span>$knowledge-ingestor</span></div><div class="user-content"><span class="user-skill-tag">引用 Skill</span>${escapeHtml(content)}${sourceText}</div>`;
  $("#conversation").append(element); scrollConversation();
}

function appendAssistant(answer = null) {
  const element = document.createElement("article"); element.className = "message assistant-message";
  element.innerHTML = `<div class="message-label"><span class="assistant-mark">K</span> 知识 AI</div>
    <div class="answer-status"><span class="thinking-dot"></span><span>正在阅读选定证据……</span></div><div class="markdown-body"></div><div class="answer-footer"></div>`;
  $("#conversation").append(element);
  if (answer) finalizeAssistant(element, answer); scrollConversation(); return element;
}

function appendSkillAssistant(result = null) {
  const element = document.createElement("article"); element.className = "message assistant-message skill-assistant-message";
  element.innerHTML = `<div class="message-label skill-message-label"><span class="assistant-mark">$</span> knowledge-ingestor</div>
    <div class="answer-status"><span class="thinking-dot"></span><span>正在启动知识摄取工作流……</span></div><div class="markdown-body"></div><div class="skill-result-footer"></div>`;
  $("#conversation").append(element);
  if (result) finalizeSkillAssistant(element, result);
  scrollConversation();
  return element;
}

function finalizeSkillAssistant(element, result) {
  element.querySelector(".answer-status")?.remove();
  element.querySelector(".markdown-body").innerHTML = renderMarkdown(result.markdown);
  const sync = result.sync;
  const syncText = sync?.status === "complete"
    ? `知识索引已同步：新增 ${sync.created || 0}、更新 ${sync.updated || 0}`
    : (sync?.status === "error" ? `笔记已保存，但索引同步失败：${escapeHtml(sync.error || "未知错误")}` : "结果已保存到当前对话");
  element.querySelector(".skill-result-footer").innerHTML = `<span>$knowledge-ingestor</span><span>${syncText}</span>`;
}

function finalizeAssistant(element, answer) {
  element.dataset.answerId = answer.answer_id;
  element.querySelector(".answer-status").remove();
  element.querySelector(".markdown-body").innerHTML = renderMarkdown(answer.markdown);
  element.querySelector(".answer-footer").innerHTML = `<span>${escapeHtml(modelLabel(answer.model))} · ${Math.round(answer.confidence * 100)}% 有据可查</span>
    <button class="save-answer" data-answer-id="${escapeHtml(answer.answer_id)}" ${state.bootstrap?.capabilities.obsidian ? "" : "disabled title=\"配置 Obsidian 仓库后即可保存\""}>保存到 Obsidian</button>`;
}

function scrollConversation() { const panel = $("#conversation"); panel.scrollTop = panel.scrollHeight; }

function formatTime(seconds) {
  if (seconds == null) return ""; const value = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function locationLabel(source) {
  const parts = [];
  if (source.page_number != null) parts.push(`第 ${source.page_number} 页`);
  if (source.slide_number != null) parts.push(`第 ${source.slide_number} 张幻灯片`);
  if (source.timestamp_start != null) parts.push(`${formatTime(source.timestamp_start)}${source.timestamp_end != null ? ` – ${formatTime(source.timestamp_end)}` : ""}`);
  if (source.section) parts.push(source.section); return parts.join(" · ") || "资料级来源";
}

function safeExternal(url, source) {
  if (!url) return null;
  try { const parsed = new URL(url); if (!["http:", "https:"].includes(parsed.protocol)) return null;
    if (source.timestamp_start != null) parsed.hash = `t=${Math.floor(source.timestamp_start)}`; return parsed.href;
  } catch { return null; }
}

function obsidianUri(source) {
  if (!state.bootstrap?.capabilities.obsidian || !source.obsidian_path) return null;
  const file = source.obsidian_path.replace(/\.md$/, "") + (source.section ? `#${source.section}` : "");
  return `obsidian://open?${new URLSearchParams({ vault: state.bootstrap.capabilities.obsidian_vault, file })}`;
}

function sourceActions(item) {
  const source = item.source; const actions = []; const chunkId = item.chunk_id || source.chunk_id;
  if (source.local_path && chunkId) {
    let suffix = "";
    if (source.media_type === "pdf" && source.page_number != null) suffix = `#page=${source.page_number}`;
    if (["video", "audio"].includes(source.media_type) && source.timestamp_start != null) suffix = `#t=${Math.floor(source.timestamp_start)}`;
    actions.push(`<a class="source-action primary" href="/api/source/content?chunk_id=${encodeURIComponent(chunkId)}${suffix}" target="_blank">${["video", "audio"].includes(source.media_type) ? "播放原始资料" : "查看原文"}</a>`);
    actions.push(`<button class="source-action" data-open-native="${escapeHtml(chunkId)}">在默认应用中打开</button>`);
  }
  const external = safeExternal(source.original_uri, source);
  if (external) actions.push(`<a class="source-action ${actions.length ? "" : "primary"}" href="${escapeHtml(external)}" target="_blank" rel="noopener">在线打开</a>`);
  const obsidian = obsidianUri(source);
  if (obsidian) actions.push(`<a class="source-action" href="${escapeHtml(obsidian)}">在 Obsidian 中打开</a>`);
  return actions.join("");
}

function mediaPreview(item) {
  const source = item.source; const chunkId = item.chunk_id || source.chunk_id;
  if (!source.local_path || !chunkId) return "";
  const url = `/api/source/content?chunk_id=${encodeURIComponent(chunkId)}`;
  if (source.media_type === "video") return `<video class="source-media" controls preload="metadata" src="${url}#t=${Math.floor(source.timestamp_start || 0)}"></video>`;
  if (source.media_type === "audio") return `<audio class="source-media" controls preload="metadata" src="${url}#t=${Math.floor(source.timestamp_start || 0)}"></audio>`;
  if (source.media_type === "image") return `<img class="source-media" src="${url}" alt="${escapeHtml(item.title)}" />`;
  return "";
}

function renderSources(evidence, selectedId = null) {
  state.evidence = evidence || []; state.selectedEvidenceId = selectedId || state.evidence[0]?.evidence_id || null;
  $("#evidence-count").textContent = `${state.evidence.length} 条证据`;
  if (!state.evidence.length) {
    $("#sources-content").className = "source-empty"; $("#sources-content").innerHTML = `<span class="source-empty-number">S</span><h3>证据会显示在这里</h3><p>选择搜索结果或引用，即可查看原文及其准确位置。</p>`; return;
  }
  $("#sources-content").className = "source-list";
  $("#sources-content").innerHTML = state.evidence.map((item) => {
    const active = item.evidence_id === state.selectedEvidenceId; const source = item.source;
    return `<article class="source-card ${active ? "active" : ""}" id="evidence-${item.evidence_id}" data-evidence-id="${item.evidence_id}">
      <div class="source-card-top"><span class="source-id">${item.evidence_id}</span><span class="source-type">${escapeHtml(mediaTypeLabel(source.media_type))}</span></div>
      <h3>${escapeHtml(item.title)}</h3><div class="source-location">${escapeHtml(locationLabel(source))}</div>
      ${active ? `${mediaPreview(item)}<blockquote>${escapeHtml(item.content)}</blockquote><div class="source-actions">${sourceActions(item)}</div>` : `<p>${escapeHtml(item.content.slice(0, 145))}${item.content.length > 145 ? "…" : ""}</p>`}
    </article>`;
  }).join("");
}

function selectEvidence(id) {
  if (!state.evidence.some((item) => item.evidence_id === id)) return;
  renderSources(state.evidence, id); $(".sources-panel").classList.add("open");
  requestAnimationFrame(() => document.querySelector(`#evidence-${CSS.escape(id)}`)?.scrollIntoView({ behavior: "smooth", block: "start" }));
}

async function postJson(url, payload) {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  if (!response.ok) { let error; try { error = (await response.json()).error; } catch { error = response.statusText; } throw new Error(error || "请求失败"); }
  return response;
}

async function runSearch(query) {
  setBusy(true); $("#conversation").innerHTML = `<div class="search-state"><span class="thinking-dot"></span>正在搜索原始知识……</div>`;
  try {
    const response = await postJson("/api/search", { query, filters: scopePayload(), top_k: 20 }); const data = await response.json();
    const evidence = data.results.map((item, index) => ({ evidence_id: `S${index + 1}`, content: item.content, title: item.title, score: item.score, source: item.source, chunk_id: item.chunk_id, document_id: item.document_id, source_kind: "knowledge" }));
    state.evidence = evidence; $("#conversation").innerHTML = `<div class="search-summary"><span>${data.count} 条结果</span><h2>“${escapeHtml(query)}”的原始证据</h2></div>
      <div class="search-results">${evidence.map((item) => `<button class="search-result" data-search-evidence="${item.evidence_id}"><span class="search-result-index">${item.evidence_id}</span><span><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(locationLabel(item.source))}</small><p>${escapeHtml(item.content)}</p></span></button>`).join("") || `<div class="empty-result">当前范围内没有找到匹配的原文。</div>`}</div>`;
    renderSources(evidence); $$("[data-search-evidence]").forEach((button) => button.addEventListener("click", () => selectEvidence(button.dataset.searchEvidence)));
  } finally { setBusy(false); }
}

async function runAsk(question) {
  setBusy(true); if ($(".welcome-card") || $(".search-summary")) $("#conversation").innerHTML = ""; appendUser(question); const assistant = appendAssistant();
  const requestStarted = Date.now();
  let statusMessage = "正在连接本地知识库";
  const progressTimer = window.setInterval(() => {
    const label = assistant.querySelector(".answer-status span:last-child");
    if (!label) return;
    const elapsed = Math.floor((Date.now() - requestStarted) / 1000);
    label.textContent = elapsed < 5 ? statusMessage : `${statusMessage} · 已等待 ${elapsed} 秒`;
  }, 1000);
  try {
    const response = await postJson("/api/ask/stream", { question, conversation_id: state.conversationId, mode: state.answerMode, deep_analysis: $("#deep-analysis").checked, model: $("#model-select").value, response_language: "zh-CN", filters: scopePayload() });
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = "", markdown = "", finalAnswer = null;
    while (true) {
      const { value, done } = await reader.read(); buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n"); buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue; const event = JSON.parse(line);
        if (event.type === "status") {
          statusMessage = event.message;
          assistant.querySelector(".answer-status span:last-child").textContent = statusMessage;
        }
        if (event.type === "delta") { markdown += event.text; assistant.querySelector(".markdown-body").innerHTML = renderMarkdown(markdown); scrollConversation(); }
        if (event.type === "final") finalAnswer = event.answer;
        if (event.type === "error") throw new Error(event.error);
      }
      if (done) break;
    }
    if (!finalAnswer) throw new Error("回答流在完成前意外中断");
    if (assistant.querySelector(".answer-status")) finalizeAssistant(assistant, finalAnswer);
    else { assistant.querySelector(".markdown-body").innerHTML = renderMarkdown(finalAnswer.markdown); assistant.querySelector(".answer-footer").innerHTML = `<span>${escapeHtml(modelLabel(finalAnswer.model))}</span><button class="save-answer" data-answer-id="${finalAnswer.answer_id}">保存到 Obsidian</button>`; }
    state.conversationId = finalAnswer.conversation_id; localStorage.setItem("knowledge.conversationId", state.conversationId); renderSources(finalAnswer.evidence);
    await refreshBootstrapHistory();
  } catch (error) {
    assistant.querySelector(".answer-status")?.remove(); assistant.querySelector(".markdown-body").innerHTML = `<p class="error-message">${escapeHtml(error.message)}</p>`;
  } finally { window.clearInterval(progressTimer); setBusy(false); scrollConversation(); }
}

async function runSkill(instruction) {
  setBusy(true);
  if ($(".welcome-card") || $(".search-summary")) $("#conversation").innerHTML = "";
  appendSkillUser(instruction, state.skillSources);
  const assistant = appendSkillAssistant();
  try {
    const response = await postJson("/api/skills/invoke/stream", {
      skill: state.skillName,
      instruction,
      sources: state.skillSources,
      conversation_id: state.conversationId,
    });
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = "", markdown = "", finalResult = null;
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n"); buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "status") assistant.querySelector(".answer-status span:last-child").textContent = event.message;
        if (event.type === "delta") {
          markdown += event.text;
          assistant.querySelector(".markdown-body").innerHTML = renderMarkdown(markdown);
          scrollConversation();
        }
        if (event.type === "final") finalResult = event.result;
        if (event.type === "error") throw new Error(event.error);
      }
      if (done) break;
    }
    if (!finalResult) throw new Error("Skill 运行结果在完成前意外中断");
    finalizeSkillAssistant(assistant, finalResult);
    state.conversationId = finalResult.conversation_id;
    localStorage.setItem("knowledge.conversationId", state.conversationId);
    renderSources([]);
    await refreshBootstrapHistory();
  } catch (error) {
    assistant.querySelector(".answer-status")?.remove();
    assistant.querySelector(".markdown-body").innerHTML = `<p class="error-message">${escapeHtml(error.message)}</p>`;
    assistant.querySelector(".skill-result-footer").innerHTML = `<span>$knowledge-ingestor</span><span>运行失败，未记录为已完成任务</span>`;
  } finally {
    setBusy(false); scrollConversation();
  }
}

async function pickSkillFiles() {
  const button = $("#pick-skill-files"); button.disabled = true; button.textContent = "选择中……";
  try {
    const response = await postJson("/api/skills/pick-files", {});
    const data = await response.json();
    state.skillSources = [...new Set([...state.skillSources, ...(data.files || [])])];
    renderSkillFiles();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false; button.textContent = "选择文件";
  }
}

function setBusy(value) {
  state.busy = value;
  $(".send-button").disabled = value;
  $("#prompt-input").disabled = value;
  $("#skill-select").disabled = value;
  $("#pick-skill-files").disabled = value;
}

async function loadConversation(conversationId, notify = true) {
  try {
    const response = await fetch(`/api/conversations/${encodeURIComponent(conversationId)}`); if (!response.ok) throw new Error("该对话暂不可用");
    const record = await response.json(), answers = new Map(record.answers.map((answer) => [answer.answer_message_id, answer]));
    $("#conversation").innerHTML = "";
    record.messages.forEach((message) => {
      if (message.metadata?.skill && message.role === "user") appendSkillUser(message.content, message.metadata.sources || []);
      else if (message.metadata?.skill && message.role === "assistant") appendSkillAssistant({ markdown: message.content, sources: message.metadata.sources || [], sync: message.metadata.sync });
      else if (message.role === "user") appendUser(message.content);
      else appendAssistant(answers.get(message.message_id) || { answer_id: "", markdown: message.content, model: "saved", confidence: 0, evidence: [] });
    });
    state.conversationId = conversationId; localStorage.setItem("knowledge.conversationId", conversationId);
    const latest = record.answers.at(-1); renderSources(latest?.evidence || []); if (notify) showToast("已恢复历史对话");
  } catch (error) { localStorage.removeItem("knowledge.conversationId"); state.conversationId = null; if (notify) showToast(error.message, true); }
}

async function refreshBootstrapHistory() {
  const response = await fetch("/api/bootstrap"); if (!response.ok) return; const data = await response.json(); state.bootstrap = data;
  renderKnowledgeSnapshot(data);
  renderConversationHistory(data);
  renderSyncCapability(data);
}

async function syncKnowledge() {
  const button = $("#sync-knowledge");
  button.disabled = true; button.textContent = "同步中……";
  $("#index-status").textContent = "正在同步 Obsidian 知识笔记";
  try {
    const response = await postJson("/api/obsidian/sync", {});
    const result = await response.json();
    await refreshBootstrapHistory();
    showToast(`同步完成：新增 ${result.created}、更新 ${result.updated}、删除 ${result.deleted}`);
  } catch (error) {
    showToast(error.message, true);
    await refreshBootstrapHistory();
  } finally {
    button.disabled = false; button.textContent = "同步知识库";
  }
}

async function saveAnswer(answerId) {
  try { const response = await postJson("/api/obsidian/save", { answer_id: answerId, tags: ["ai-answer"] }); const result = await response.json(); showToast(`已保存到 ${result.path}`); }
  catch (error) { showToast(error.message, true); }
}

async function openNative(chunkId) {
  try { await postJson("/api/source/open-native", { chunk_id: chunkId }); showToast("已在默认应用中打开"); }
  catch (error) { showToast(error.message, true); }
}

function newConversation() {
  state.conversationId = null; localStorage.removeItem("knowledge.conversationId"); $("#conversation").innerHTML = welcomeMarkup(); bindPromptChips(); renderSources([]); showToast("已新建对话");
}

$$('.mode-button').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
$("#clear-scope").addEventListener("click", clearScope); $("#all-knowledge").addEventListener("click", clearScope);
$("#new-conversation").addEventListener("click", newConversation); $("#source-close").addEventListener("click", () => $(".sources-panel").classList.remove("open"));
$("#sync-knowledge").addEventListener("click", syncKnowledge);
$("#skill-select").addEventListener("change", (event) => setSkill(event.target.value));
$("#model-select").addEventListener("change", updateModelHint);
$("#pick-skill-files").addEventListener("click", pickSkillFiles);
$("#remove-skill").addEventListener("click", () => setSkill(""));
$("#skill-files").addEventListener("click", (event) => {
  const remove = event.target.closest("[data-remove-skill-file]");
  if (!remove) return;
  state.skillSources.splice(Number(remove.dataset.removeSkillFile), 1);
  renderSkillFiles();
});
$("#knowledge-mode").addEventListener("click", () => { state.answerMode = "knowledge"; $("#knowledge-mode").classList.add("selected"); $("#web-mode").classList.remove("selected"); });
$("#web-mode").addEventListener("click", () => { if (!$("#web-mode").disabled) { state.answerMode = "knowledge+web"; $("#web-mode").classList.add("selected"); $("#knowledge-mode").classList.remove("selected"); } });
$("#conversation").addEventListener("click", (event) => { const citation = event.target.closest("[data-citation]"); if (citation) selectEvidence(citation.dataset.citation); const save = event.target.closest("[data-answer-id]"); if (save?.classList.contains("save-answer")) saveAnswer(save.dataset.answerId); });
$("#sources-content").addEventListener("click", (event) => { const native = event.target.closest("[data-open-native]"); if (native) { event.stopPropagation(); openNative(native.dataset.openNative); return; } const card = event.target.closest("[data-evidence-id]"); if (card) selectEvidence(card.dataset.evidenceId); });
$("#composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  const input = $("#prompt-input"), value = input.value.trim();
  if (!value) return;
  if (state.skillName) {
    const approved = window.confirm("运行 knowledge-ingestor 会按照当前配置归档资料，并可能写入 Obsidian。确定继续吗？");
    if (!approved) return;
    input.value = "";
    await runSkill(value);
    return;
  }
  input.value = "";
  state.mode === "search" ? await runSearch(value) : await runAsk(value);
});
$("#prompt-input").addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#composer").requestSubmit(); } });
document.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); $("#prompt-input").focus(); } });
bindPromptChips(); renderSkillFiles(); setMode("ask"); updateScopeSummary();
loadBootstrap();
