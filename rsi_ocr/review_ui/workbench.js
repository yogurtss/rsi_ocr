"use strict";

const q = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, character => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[character]));
const names = {paddle: "PaddleOCR‑VL", oxr: "OXR‑1.0", ovis: "OvisOCR3"};
const tasks = {text: "文字", table: "表格"};
const statuses = {pending: "未复核", approved: "Pass", rejected: "Not pass", needs_check: "待核查"};
const reasons = {
  low_target_judge: "Paddle 分数低", teacher_disagreement: "模型输出分歧",
  better_judged_candidate: "候选分数更高", numeric_conflict: "数字冲突",
  judge_precheck_reject: "评分预检未通过", empty: "Paddle 输出为空", format_invalid: "Paddle 格式无效",
};
const judgeStatuses = {scored: "已评分", precheck_reject: "预检未通过", render_error: "渲染失败",
  service_error: "服务异常", incomplete: "输出未完成", format_invalid: "格式无效", missing: "未评分"};
const metricsNames = {cer: "字符错误率", teds: "TEDS", teds_s: "结构 TEDS", numeric_match: "数字一致",
  numeric_accuracy: "数字准确率", structure: "结构 F1", content: "内容相似度", numeric: "数字一致率"};
const number = (value) => typeof value === "number" ? value.toFixed(3) : "—";
const percent = (value) => typeof value === "number" ? (value * 100).toFixed(1) + "%" : "—";
const nameOf = (alias) => names[alias] || alias;
const state = {batch: "", data: null, task: "all", rows: [], current: null, case: null,
  compare: null, difference: true, request: 0, loading: false, saving: false};

function notice(message, error = false) {
  const element = q("#notice");
  element.textContent = message;
  element.hidden = !message;
  element.classList.toggle("error", error);
  clearTimeout(notice.timer);
  if (message && !error) notice.timer = setTimeout(() => { element.hidden = true; }, 3200);
}

async function api(path, options) {
  const response = await fetch(path, {cache: "no-store", ...options});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "请求失败，请重试");
  return result;
}

const batchPath = () => "/api/batches/" + encodeURIComponent(state.batch);
const casePath = () => batchPath() + "/cases/" + encodeURIComponent(state.current);

function renderQueue() {
  const list = q("#queue-list");
  list.innerHTML = state.rows.map(row => {
    const selected = state.current === row.case_id;
    const key = row.reasons[0];
    const hint = reasons[key] || key || (row.audit_sample ? "随机抽审" : "未触发规则");
    const title = row.doc_id.replace(/^(newspaper|jiaocaineedrop|yanbaopptmerge)_/i, "");
    return `<button class="queue-item ${selected ? "selected" : ""}" data-case="${esc(row.case_id)}"
      aria-current="${selected}" title="${esc(row.doc_id)} · ${esc(row.region_id)}">
      <span class="queue-meta"><span class="task-chip">${esc(tasks[row.task] || row.task)}</span>
      <span class="row-score">${number(row.priority)}</span>
      <span class="row-status ${esc(row.review_status)}">${esc(statuses[row.review_status])}</span></span>
      <strong>${esc(title)}</strong><span class="row-detail">${esc(row.region_id)} · ${esc(hint)}</span>
    </button>`;
  }).join("") || '<p class="empty">没有符合条件的样本。</p>';
  q("#queue-count").textContent = state.rows.length + " 条";
}

const difficultyLabels = {all: "全部", hard: "难例", easy: "非难例", error: "异常"};
const difficultyOf = row => row.operational ? "error" : row.hard_case ? "hard" : "easy";
const eligible = row => difficultyOf(row) === "easy" && row.review_status === "pending" && row.valid_target;
function filterQueue() {
  const query = q("#search").value.trim().toLowerCase();
  const scope = q("#scope").value;
  const status = q("#status").value;
  const reason = q("#reason").value;
  const judge = q("#judge").value;
  const gold = q("#gold").value;
  state.rows = state.data.rows.filter(row => (
    (state.level === "all" || difficultyOf(row) === state.level) &&
    (scope === "all" || (scope === "queue" && row.queued) || (scope === "audit" && row.audit_sample)
      || (scope === "outside" && !row.queued))
    && (state.task === "all" || row.task === state.task)
    && (status === "all" || row.review_status === status)
    && (reason === "all" || row.reasons.includes(reason))
    && (judge === "all" || (row.judge_status || "missing") === judge)
    && (gold === "all" || (gold === "error" && row.target_error) || (gold === "repairable" && row.repairable))
    && (!query || [row.doc_id, row.region_id, row.case_id].join(" ").toLowerCase().includes(query))
  )).sort((a, b) => (b.priority ?? -1) - (a.priority ?? -1) || a.case_id.localeCompare(b.case_id));
  q("#difficulty").innerHTML = Object.entries(difficultyLabels).map(([key,label]) =>
    `<button data-level="${key}" aria-pressed="${state.level === key}">${label} ${state.data.rows.filter(r => key === "all" || difficultyOf(r) === key).length}</button>`).join("");
  q("#bulk").hidden = state.level !== "easy";
  const count = state.rows.filter(eligible).length;
  q("#bulk").textContent = `一键通过 ${count} 条 · 采用 Paddle`;
  q("#bulk").disabled = !count || state.saving;
  q("#undo").hidden = !sessionStorage.getItem("undo-" + state.batch);
  renderQueue();
  if (!state.rows.some(row => row.case_id === state.current)) {
    selectCase(state.rows[0]?.case_id);
  } else if (state.case) {
    updateNavigation();
  }
}

function resetOptions(selector, values, labels, placeholder) {
  const select = q(selector);
  select.replaceChildren(new Option(placeholder, "all"), ...values.map(value => new Option(labels[value] || value, value)));
}

async function loadBatch(batch) {
  const request = ++state.request;
  state.loading = true;
  state.batch = batch;
  state.current = null;
  state.case = null;
  state.data = null;
  if (!q("#settings").hidden) initSettings();
  q("#queue-list").replaceChildren();
  q("#queue-count").textContent = "";
  q("#inspection").innerHTML = '<p class="empty">正在读取批次…</p>';
  q("#overview").replaceChildren();
  try {
    const data = await api(batchPath());
    if (request !== state.request) return;
    state.data = data;
    state.loading = false;
    if (!q("#settings").hidden) initSettings();
    q("#search").value = "";
    state.level = "all";
    q("#scope").value = "all";
    q("#status").value = "all";
    taskFilter("all");
    q("#gold").value = "all";
    q("#reference-filter").hidden = !data.report;
    resetOptions("#reason", [...new Set(data.rows.flatMap(row => row.reasons))], reasons, "全部原因");
    resetOptions("#judge", [...new Set(data.rows.map(row => row.judge_status || "missing"))], judgeStatuses, "全部评分状态");
    q("#review-download").href = batchPath() + "/export/review";
    q("#training-download").hidden = !data.rows.some(row => row.exportable && row.split !== "val");
    q("#validation-download").hidden = !data.rows.some(row => row.exportable && row.split === "val");
    renderOverview();
    filterQueue();
  } catch (error) {
    if (request !== state.request) return;
    state.loading = false;
    q("#inspection").innerHTML = '<p class="empty">无法读取批次。</p>';
    notice(error.message, true);
  }
}

function taskFilter(task) {
  state.task = task;
  q("#task-filters").querySelectorAll("button").forEach(button => {
    button.setAttribute("aria-pressed", button.dataset.task === task);
  });
}

async function selectCase(id, {reveal = false} = {}) {
  if (id === state.current && state.case) return;
  const request = ++state.request;
  state.current = id;
  state.case = null;
  state.compare = null;
  state.difference = true;
  renderQueue();
  const inspector = q("#inspection");
  inspector.innerHTML = id ? '<p class="empty">正在加载样本…</p>' : '<p class="empty">请调整筛选条件以查看样本。</p>';
  inspector.removeAttribute("aria-busy");
  if (!id) return;
  inspector.setAttribute("aria-busy", "true");
  try {
    const data = await api(casePath());
    if (request !== state.request) return;
    state.case = data;
    const alternatives = Object.keys(data.models).filter(alias => alias !== data.target_alias);
    const reviewedModel = data.decision?.model_alias;
    state.compare = reviewedModel && Object.hasOwn(data.models, reviewedModel) ? reviewedModel : data.target_alias;
    if (!data.models[state.compare].label && !data.decision) {
      state.compare = alternatives.find(alias => data.models[alias].label) || data.target_alias;
    }
    const first = Object.keys(data.models).find(k=>data.models[k].label);
    if (!data.models[data.target_alias].label && first) await fetchComparison(data,first);
    if (request !== state.request) return;
    renderCase();
    const list = q("#queue-list");
    const item = list.querySelector(".selected");
    if (item) {
      const itemBox = item.getBoundingClientRect();
      const queueBox = list.getBoundingClientRect();
      if (itemBox.top < queueBox.top) list.scrollTop += itemBox.top - queueBox.top - 4;
      if (itemBox.bottom > queueBox.bottom) list.scrollTop += itemBox.bottom - queueBox.bottom + 4;
    }
    if (reveal) inspector.scrollIntoView({block: "start"});
  } catch (error) {
    if (request !== state.request) return;
    inspector.innerHTML = '<p class="empty">无法加载样本。请从队列重新选择。</p>';
    notice(error.message, true);
  } finally {
    if (request === state.request) inspector.removeAttribute("aria-busy");
  }
}

function reasonBadge(key) {
  return `<span class="reason-chip ${key === "numeric_conflict" ? "numeric" : ""}">${esc(reasons[key] || key)}</span>`;
}

function modelVerdict(alias, row) {
  const model = row.models[alias];
  const metric = row.metrics[alias];
  const recommended = Boolean(model.label && row.proposals.some(
    proposal => proposal.recommended && proposal.label === model.label));
  const verdict = metric ? (metric.correct ? "符合标注" : "与标注有误差")
    : model.status === "ok" ? (judgeStatuses[model.judge?.status] || model.judge?.status || "未评分")
      : (judgeStatuses[model.status] || model.status);
  return `<span class="model-verdict ${metric ? (metric.correct ? "correct" : "incorrect") : ""}">${esc(verdict)}
      ${recommended ? '<small class="model-recommendation">系统推荐</small>' : ""}</span>`;
}

function modelTile(alias, row) {
  const score = row.models[alias].judge?.prob_yes;
  return `<button class="model-tile" type="button" aria-pressed="${state.compare === alias}" data-model="${esc(alias)}">
    <span class="model-name">${esc(nameOf(alias))}${alias === row.target_alias ? '<small>基线</small>' : ""}</span>
    <span class="model-score">${number(score)}</span>
    ${modelVerdict(alias, row)}
    </button>`;
}

function sheetWidth(view) {
  return (view?.rows || []).reduce((n,row)=>row.reduce((m,c)=>Math.max(m,c.col+c.width),n),0);
}

// Crop spans at the viewport edges, including cells anchored above or left of it.
function tableOutput(view, highlight, top=0, left=0, height=6, width=6, selected=null) {
  const bottom = Math.min(view.rows.length,top+height), right = Math.min(sheetWidth(view),left+width);
  if (top>=bottom || left>=right) return '<p class="output-empty">此结果在该范围内没有单元格</p>';
  const lines = Array.from({length:bottom-top},()=>[]);
  for (let r=0;r<bottom;r++) {
    for (const cell of view.rows[r]) {
      if (r+cell.height<=top || cell.col+cell.width<=left || cell.col>=right) continue;
      const start = Math.max(r,top), col = Math.max(cell.col,left);
      const active = selected && selected.r>=r && selected.r<r+cell.height && selected.c>=cell.col && selected.c<cell.col+cell.width;
      lines[start-top].push([col,`<td data-row="${r}" data-col="${cell.col}" colspan="${Math.min(right,cell.col+cell.width)-col}"
        rowspan="${Math.min(bottom,r+cell.height)-start}" class="${highlight && cell.changed ? "changed" : ""} ${active?"cell-selected":""}">${esc(cell.text)}</td>`]);
    }
  }
  return '<table class="ocr-table sheet"><thead><tr><th scope="col">行 / 列</th>'
    + Array.from({length:right-left},(_,i)=>`<th scope="col">${left+i+1}</th>`).join("") + '</tr></thead><tbody>'
    + lines.map((cells,i)=>`<tr><th scope="row">${top+i+1}</th>${cells.sort((a,b)=>a[0]-b[0]).map(cell=>cell[1]).join("")}</tr>`).join("")+'</tbody></table>';
}

function output(label, view, highlight) {
  if (!label) return '<p class="output-empty">无有效转录</p>';
  if (view?.rows) return tableOutput(view, highlight);
  if (!highlight || !view?.parts) return `<div class="ocr-text">${esc(label)}</div>`;
  return '<div class="ocr-text">' + view.parts.map(([kind, text]) => {
    if (kind === "insert") return `<mark>${esc(text)}</mark>`;
    if (kind === "delete") return `<del>${esc(text)}</del>`;
    return esc(text);
  }).join("") + '</div>';
}

function metricLine(metrics, view) {
  const entries = Object.entries(metrics || {}).filter(([key, value]) => key !== "correct" && value !== null);
  if (!entries.length && view?.structure != null) entries.push(["structure", view.structure]);
  return entries.length ? `<div class="output-metrics">${entries.map(([key, value]) =>
    `<span>${esc(metricsNames[key] || key)} <b>${typeof value === "boolean" ? (value ? "一致" : "冲突") : percent(value)}</b></span>`
  ).join("")}</div>` : "";
}

function comparisonViews(row, base) {
  if (base === "reference") return row.views;
  if (base === row.target_alias) return row.baseline_views;
  return row.extraViews?.[base] || {};
}

async function fetchComparison(row, base) {
  if (base === "reference" || base === row.target_alias || row.extraViews?.[base]) return;
  const views = await api(casePath() + "/comparison/" + encodeURIComponent(base));
  (row.extraViews ||= {})[base] = views;
}

function resultCard(row, alias, view, highlight, onlyChanges = false, page = null) {
  const label = alias === "reference" ? row.reference : row.models[alias].label;
  const dimensions = view?.rows ? ` · ${view.rows.length} 行 × ${sheetWidth(view)} 列` : "";
  let body = page && view?.rows ? tableOutput(view, state.difference && highlight, page.top,
    page.left, page.height, page.width) : output(label, view, state.difference && highlight);
  if (onlyChanges && label && view?.parts) {
    const parts = view.parts.filter(([tag]) => tag !== "equal");
    body = highlight ? (parts.length ? parts.map(part => output(label,{parts:[part]},true)).join("") :
      '<p class="output-empty">与基准一致</p>') : '<p class="output-empty">当前差异基准</p>';
  }
  const counts = view?.rows ? view.rows.flat().reduce((n,c) => {
    if (c.height > 1 || c.width > 1) n.merged++;
    if (c.changed) n.changed++;
    return n;
  },{merged:0,changed:0}) : null;
  return `<section class="output-card"><h4>${esc(alias === "reference" ? "参考标注" : nameOf(alias))}${dimensions}${page ? `<small>Jev ${number(row.models[alias].judge?.prob_yes)}</small>` : ""}</h4>
    <div class="output-scroll" tabindex="0">${body}</div>
    ${page ? `<div class="output-metrics">${modelVerdict(alias,row)}</div>` : ""}
    ${counts ? `<div class="output-metrics">合并格 ${counts.merged} · 当前结果差异格 ${counts.changed}</div>` : ""}
    ${metricLine(row.metrics[alias], view)}</section>`;
}

function evidenceDetails(row) {
  const p = row.policy, w = p.weights, s = row.signals;
  const rules = [
    ["Paddle 质量分", row.models[row.target_alias].judge?.prob_yes, "≤", p.low_judge, "low_target_judge", "target_risk"],
    ["模型分歧", s.disagreement, "≥", p.disagreement, "teacher_disagreement", "disagreement"],
    ["最佳候选增益", s.judge_gain, "≥", p.judge_gain, "better_judged_candidate", "judge_gain"],
    ["数字冲突", s.numeric_conflict, "=", 1, "numeric_conflict", "numeric_conflict"]];
  const known = Object.keys(s).filter(k => s[k] != null && w[k] > 0);
  const formula = ["empty","format_invalid"].includes(row.models[row.target_alias].status) ? "输出为空或格式无效，优先级置为 1" :
    known.length ? "(" + known.map(k => number(s[k]) + " × " + w[k]).join(" + ") + ") ÷ " + number(known.reduce((n,k) => n+w[k],0)) : "无可用评分信号";
  return `<details class="inspection-details"><summary>判定明细与计算过程</summary><table class="strategy-table">
  <thead><tr><th>信号</th><th>实测</th><th>触发阈值</th><th>排序权重</th><th>结果</th></tr></thead><tbody>${rules.map(([label,val,op,limit,reason,key]) =>
    `<tr><td>${label}</td><td>${number(val)}</td><td>${op} ${number(limit)}</td><td>${w[key]}</td><td>${p.fixed_rules?.[reason] === false ? "关闭" : val == null ? "缺失" : row.reasons.includes(reason) ? "命中" : "未命中"}</td></tr>`).join("")}</tbody></table>
    <p>排序分 ${number(row.priority)} ≈ ${esc(formula)}。Paddle 风险取 1 减质量分。</p></details>`;
}

const previewKey = "glyphstack.table.visibleModels";
let preferredModels;
try {
  const stored = JSON.parse(localStorage.getItem(previewKey));
  if (Array.isArray(stored)) preferredModels = stored;
} catch {}
let previewRow, previewTop = 0, previewLeft = 0;
const previewSize = {height: 8, width: 1};

function visibleModels(row, keys) {
  const picked = preferredModels?.filter(key => keys.includes(key));
  if (picked?.length) return keys.filter(key => picked.includes(key));
  const wide = Object.values(row.baseline_views || {}).some(view => sheetWidth(view) > 6);
  return wide ? [row.models[row.target_alias]?.label ? row.target_alias :
    keys.find(key => row.models[key].label) || keys[0]] : keys;
}

function renderTablePreview(row, keys, base, views) {
  const pair = q("#model-pair");
  const visible = visibleModels(row, keys);
  if (previewRow !== row) {
    previewRow = row;
    previewLeft = previewTop = 0;
  }
  q("#visible-models").innerHTML = keys.map(key => {
    const on = visible.includes(key);
    return `<label><input type="checkbox" data-visible="${esc(key)}" ${on ? "checked" : ""}
      ${on && visible.length === 1 ? "disabled" : ""}> ${esc(nameOf(key))}</label>`;
  }).join("");
  pair.style.setProperty("--models", visible.length);
  const stacked = window.matchMedia("(max-width: 700px)").matches;
  const cardWidth = pair.clientWidth / (stacked ? 1 : visible.length);
  previewSize.width = Math.max(2, Math.min(24, Math.floor((cardWidth - 55) / 80)));
  const rows = Math.max(0, ...visible.map(key => views[key]?.rows?.length || 0));
  const cols = Math.max(0, ...visible.map(key => sheetWidth(views[key])));
  function range(axis, title, start, size, total) {
    return `<span><button type="button" data-move="${axis}" data-dir="-1" aria-label="上一组${title}" ${!start ? "disabled" : ""}>‹</button>
      ${title} ${total ? start+1 : 0}–${Math.min(total,start+size)} / ${total}
      <button type="button" data-move="${axis}" data-dir="1" aria-label="下一组${title}" ${start+size>=total ? "disabled" : ""}>›</button></span>`;
  }
  previewTop = Math.min(previewTop, Math.max(0, rows-1));
  previewLeft = Math.min(previewLeft, Math.max(0, cols-1));
  q("#preview-range").innerHTML = range("height","行",previewTop,previewSize.height,rows)
    + range("width","列",previewLeft,previewSize.width,cols);
  const page = {...previewSize, top:previewTop, left:previewLeft};
  pair.innerHTML = visible.map(key => resultCard(row,key,views[key],key !== base,false,page)).join("");
}

function renderComparison() {
  const row = state.case;
  const alias = state.compare;
  const model = row.models[alias];
  const base = q("#compare-with").value;
  const views = comparisonViews(row, base);
  const keys = [row.target_alias, ...Object.keys(row.models).filter(key => key !== row.target_alias)];
  if (row.task === "table") {
    q("#adopt-model").value = alias;
    renderTablePreview(row, keys, base, views);
  } else {
    q("#model-tiles").innerHTML = keys.map(key => modelTile(key, row)).join("");
    q("#model-pair").innerHTML = keys.map(key => resultCard(row, key, views[key], key !== base, q("#only-changes")?.checked)).join("");
  }
  q("#raw-output").textContent = model.raw || model.detail || "无原始输出";
  q("#raw-title").textContent = nameOf(alias) + " · 原始输出与请求";
  const info = {...model};
  delete info.raw;
  delete info.label;
  q("#request-info").textContent = JSON.stringify(info, null, 2);
  q("#review-choice").innerHTML = `<strong>${esc(nameOf(alias))}</strong><span>${
    model.label ? "Pass 将采用所选模型的转录" : "当前模型没有可用转录，请切换模型或点 Not pass"
  }${row.export_split ? "" : " · 仅记录评测判定"}</span>`;
  q("#pass").disabled = !model.label || state.saving;
  q("#reject").disabled = state.saving;
  q("#table-pass").disabled = q("#pass").disabled;
  q("#table-reject").disabled = q("#reject").disabled;
}

function updateNavigation() {
  const index = state.rows.findIndex(row => row.case_id === state.current);
  const prev = q("#previous");
  const next = q("#next");
  if (!prev || index < 0) return;
  prev.disabled = index === 0 || state.saving;
  next.disabled = index === state.rows.length - 1 || state.saving;
  q("#position").textContent = `${index + 1} / ${state.rows.length}`;
}

function pageFrame(url, geometry) {
  let box = "";
  if (geometry?.box) {
    const [left, top, right, bottom] = geometry.box;
    const units = [left / geometry.width, top / geometry.height,
      (right - left) / geometry.width, (bottom - top) / geometry.height].map(value => (value * 100).toFixed(5));
    box = `<span class="region-box" aria-hidden="true" style="left:${units[0]}%;top:${units[1]}%;width:${units[2]}%;height:${units[3]}%"></span>`;
  }
  const size = geometry ? `width="${Number(geometry.width)}" height="${Number(geometry.height)}"` : "";
  return `<span class="page-frame"><img src="${esc(url)}" ${size} draggable="false" alt="${box ? "已框出目标区域的原页" : "原页"}">${box}</span>`;
}

function renderCase() {
  const row = state.case;
  const listed = state.rows.find(item => item.case_id === row.case_id);
  const decision = row.decision;
  const stateLabel = decision?.status || "pending";
  const crop = casePath() + "/image/crop";
  const page = casePath() + "/image/page";
  const badges = row.reasons.map(reasonBadge).join("")
    || `<span class="reason-chip">${row.audit_sample ? "随机抽审" : "未触发难例规则"}</span>`;
  q("#inspection").innerHTML = `
    <div class="case-header">
      <div class="case-heading">
        <div class="case-eyebrow"><span class="task-chip">${esc(tasks[row.task])}</span>
          <span class="case-status ${esc(stateLabel)}">${esc(statuses[stateLabel])}</span>
          ${listed?.target_error ? '<span class="case-status incorrect">Paddle 有误</span>' : ""}
          ${listed?.repairable ? '<span class="case-status correct">有正确替代</span>' : ""}</div>
        <h2>区域 ${esc(row.region_id)}</h2>
        <p title="${esc(row.doc_id)}">${esc(row.doc_id)} · 页 ${esc(row.page_id)}</p>
      </div>
      <div class="pager">
        <button id="previous" type="button">上一条</button>
        <span id="position"></span>
        <button id="next" type="button">下一条</button>
      </div>
    </div>
    <section class="decision-summary">
      <div><h3>${listed?.queued ? (row.audit_sample ? "随机抽检入队" : "难例入队") : "未入审核队列"} · 判定依据</h3><div class="chips">${badges}
        ${row.audit_sample && row.reasons.length ? '<span class="reason-chip">同时命中随机抽审</span>' : ""}</div>${evidenceDetails(row)}</div>
      <div class="priority"><span>排序分</span><strong>${number(row.priority)}</strong></div>
    </section>
    <div class="comparison-layout">
      <section class="panel source">
        <header class="panel-heading"><h3>原页定位</h3>
          ${row.page_geometry ? `<button type="button" data-image="${esc(page)}" data-page="true">查看原页</button>` : ""}</header>
        ${row.page_geometry ? `<button type="button" class="page-preview" data-image="${esc(page)}" data-page="true">
          ${pageFrame(page, row.page_geometry)}</button>
          <p class="location-caption"><span class="${row.page_geometry.box ? "located" : ""}"></span>${
            row.page_geometry.box ? "框内为当前区域" : "该裁图的位置待核对"}</p>` : '<p class="empty">原页文件不可用。</p>'}
        ${row.has_crop ? `<button type="button" class="crop" data-image="${esc(crop)}">
          <img src="${esc(crop)}" alt="区域 ${esc(row.region_id)} 的裁图"><span>放大裁图</span></button>`
          : '<p class="empty">裁图文件不可用。</p>'}
      </section>
      <section class="panel comparison">
        <header class="panel-heading"><h3>识别对照</h3><div class="panel-actions"><span>Jev 一致性分</span>
          ${row.task === "table" ? '<button type="button" id="open-table">放大表格</button>' : ""}</div></header>
        ${row.task === "table" ? `<div class="preview-tools"><fieldset><legend>显示模型</legend><div id="visible-models"></div></fieldset>
          <label>Pass 采用<select id="adopt-model">${Object.keys(row.models).map(key =>
            `<option value="${esc(key)}">${esc(nameOf(key))} · ${number(row.models[key].judge?.prob_yes)}</option>`).join("")}</select></label></div>` :
          '<div id="model-tiles" class="model-tiles"></div>'}
        <div class="diff-controls">
          <label>差异基准<select id="compare-with">${Object.keys(row.models).filter(k=>row.models[k].label).map(k=>`<option value="${esc(k)}" ${k===row.target_alias?"selected":""}>${esc(nameOf(k))}</option>`).join("")}${row.reference?'<option value="reference">参考标注</option>':""}</select></label>
          ${row.reference ? "" : "<span>无参考标注</span>"}
          ${row.task === "text" ? '<label><input id="only-changes" type="checkbox">仅看分歧</label>' : ""}
          <label><input id="difference" type="checkbox" checked>高亮差异</label>
        </div>
        ${row.task === "table" ? '<nav id="preview-range" aria-label="表格预览范围"></nav>' : ""}
        <div id="model-pair" class="output-pair ${row.task === "table" ? "table-preview" : ""}"></div>
        ${row.reference ? `<details class="inspection-details"><summary>查看参考标注</summary><div class="output-scroll">${output(row.reference,row.reference_view,false)}</div></details>` : ""}
        <details class="inspection-details"><summary id="raw-title"></summary>
          <pre id="raw-output"></pre><pre id="request-info"></pre></details>
      </section>
    </div>
    <section class="review-dock" aria-label="人工复核">
      <div class="review-selection" id="review-choice"></div>
      <div class="review-actions">
        <button id="reject" type="button" data-review="rejected">Not pass</button>
        <button id="pass" class="primary" type="button" data-review="approved">Pass</button>
      </div>
    </section>`;
  renderComparison();
  if (decision?.timestamp) {
    const context = document.createElement("small");
    context.className = "review-timestamp";
    context.textContent = "上次复核 " + new Date(decision.timestamp).toLocaleString("zh-CN");
    q("#review-choice").append(context);
  }
  updateNavigation();
}

function changeCase(offset) {
  if (state.saving) return;
  const index = state.rows.findIndex(row => row.case_id === state.current);
  const next = index >= 0 ? state.rows[index + offset] : undefined;
  if (next) selectCase(next.case_id, {reveal: true});
}

async function saveReview(status) {
  if (state.saving) return;
  const path = casePath();
  const row = state.rows.find(item => item.case_id === state.current);
  const next = state.rows.slice(state.rows.indexOf(row) + 1).find(item => item.review_status === "pending")
    || state.rows.find(item => item.case_id !== state.current && item.review_status === "pending");
  state.saving = true;
  q("#batch").disabled = true;
  q("#pass").disabled = true;
  q("#reject").disabled = true;
  q("#table-pass").disabled = true;
  q("#table-reject").disabled = true;
  q("#table-feedback").textContent = "";
  updateNavigation();
  try {
    const decision = await api(path + "/decision", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({status, model_alias: state.compare}),
    });
    row.review_status = decision.status;
    state.case.decision = decision;
    state.saving = false;
    if (q("#table-dialog").open) q("#table-dialog").close();
    const destination = decision.sft ? " · 已存入 SFT 数据" : "";
    notice("已记录 " + statuses[decision.status] + destination);
    renderOverview();
    if (next) await selectCase(next.case_id, {reveal: true});
    else renderCase();
    filterQueue();
  } catch (error) {
    notice(error.message, true);
    q("#table-feedback").textContent = error.message;
  } finally {
    state.saving = false;
    q("#batch").disabled = false;
    if (state.case && q("#pass")) q("#pass").disabled = !state.case.models[state.compare].label;
    if (q("#reject")) q("#reject").disabled = false;
    q("#table-pass").disabled = Boolean(q("#pass")?.disabled);
    q("#table-reject").disabled = false;
    updateNavigation();
  }
}

async function downloadSft(split) {
  try {
    const response = await fetch(batchPath() + "/export/sft/" + split, {method: "POST"});
    if (!response.ok) throw new Error((await response.json()).detail);
    const objectURL = URL.createObjectURL(await response.blob());
    const trigger = document.createElement("a");
    trigger.href = objectURL;
    trigger.download = response.headers.get("content-disposition")?.match(/filename="([^"]+)"/)?.[1] || "sft.jsonl";
    trigger.click();
    setTimeout(() => URL.revokeObjectURL(objectURL), 1000);
    notice("已保存并下载 SFT 数据");
  } catch (error) {
    notice(error.message, true);
  }
}

function updateTableDialog() {
  const content = q("#table-dialog-content");
  content.classList.toggle("single-table", q("#table-layout").value === "single");
  content.style.setProperty("--table-zoom", q("#table-zoom").value);
}

function showLargeTable() {
  const select = q("#table-model");
  select.replaceChildren(...Object.keys(state.case.models).map(alias => new Option(nameOf(alias), alias)));
  select.value = state.compare;
  const base = q("#table-base");
  base.replaceChildren(...Object.keys(state.case.models).map(alias=>new Option(nameOf(alias),alias)));
  if (state.case.reference) base.add(new Option("参考标注","reference"));
  base.value = state.case.reference ? "reference" : Object.keys(state.case.models).find(k=>k!==state.compare && state.case.models[k].label) || state.compare;
  q("#table-feedback").textContent = "";
  q("#table-layout").value = innerWidth < 760 ? "single" : "pair";
  renderTablePair();
  updateTableDialog();
  q("#table-dialog").showModal();
}

const locationDialog = q("#image-dialog");
const locationContent = q("#dialog-content");
let drag = null;

function drawing(enabled) {
  locationContent.classList.toggle("drawing", enabled);
  q("#draw-location").textContent = enabled ? "取消画框" : "手动画框";
  const hint = q("#location-hint");
  hint.hidden = !enabled;
  hint.textContent = enabled ? "在原页上拖动框出目标区域" : "";
}

function locationPoint(event, bounds, geometry) {
  const axis = (offset, size, pixels) => Math.round(Math.max(0, Math.min(1, offset / size)) * pixels);
  return [axis(event.clientX - bounds.left, bounds.width, geometry.width),
    axis(event.clientY - bounds.top, bounds.height, geometry.height)];
}

function showLocationPreview(event) {
  const [x, y] = locationPoint(event, drag.bounds, drag.geometry);
  const [left, top] = drag.start;
  const box = [Math.min(x, left), Math.min(y, top), Math.max(x, left), Math.max(y, top)];
  Object.assign(drag.preview.style, {
    left: box[0] / drag.geometry.width * 100 + "%",
    top: box[1] / drag.geometry.height * 100 + "%",
    width: (box[2] - box[0]) / drag.geometry.width * 100 + "%",
    height: (box[3] - box[1]) / drag.geometry.height * 100 + "%",
  });
  return box;
}

q("#draw-location").addEventListener("click", () => drawing(!locationContent.classList.contains("drawing")));
locationDialog.addEventListener("close", () => { drag?.preview.remove(); drag = null; drawing(false); });
locationContent.addEventListener("pointerdown", event => {
  const frame = event.target.closest(".page-frame");
  if (!frame || !locationContent.classList.contains("drawing") || drag || event.button !== 0) return;
  const geometry = state.case.page_geometry;
  const bounds = frame.getBoundingClientRect();
  const preview = document.createElement("span");
  preview.className = "region-box drawing-preview";
  frame.append(preview);
  drag = {bounds, geometry, preview, start: locationPoint(event, bounds, geometry), pointer: event.pointerId};
  locationContent.setPointerCapture(event.pointerId);
  showLocationPreview(event);
});
locationContent.addEventListener("pointermove", event => {
  if (drag?.pointer === event.pointerId) showLocationPreview(event);
});
locationContent.addEventListener("pointercancel", event => {
  if (drag?.pointer !== event.pointerId) return;
  drag.preview.remove();
  drag = null;
});
locationContent.addEventListener("pointerup", async event => {
  if (drag?.pointer !== event.pointerId) return;
  const box = showLocationPreview(event);
  drag.preview.remove();
  drag = null;
  if (box[2] - box[0] < 2 || box[3] - box[1] < 2) return;
  const row = state.case;
  const path = casePath();
  drawing(false);
  q("#draw-location").disabled = true;
  try {
    row.page_geometry = await api(path + "/location", {
      method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({box}),
    });
    if (row !== state.case) return;
    const preview = q(".page-preview");
    preview.innerHTML = pageFrame(preview.dataset.image, row.page_geometry);
    q(".source .location-caption").innerHTML = '<span class="located"></span>框内为当前区域';
    if (locationDialog.open && locationDialog.classList.contains("full-page")) {
      q("#dialog-title").textContent = "原页 · 手动定位";
      locationContent.innerHTML = preview.innerHTML;
    }
    notice("已保存原页位置");
  } catch (error) {
    const hint = q("#location-hint");
    hint.textContent = error.message;
    hint.hidden = false;
  } finally {
    q("#draw-location").disabled = false;
  }
});

function renderOverview() {
  const {summary, report, rows} = state.data;
  const reviewed = rows.filter(row => ["approved", "rejected"].includes(row.review_status)).length;
  const sources = Object.entries(state.data.sources || {});
  const scores = summary.judge_status;
  const totalScores = Object.values(scores).reduce((sum, count) => sum + count, 0);
  const cards = [["区域总数", rows.length], ["规则命中", summary.hard_cases], ["待处理", rows.length - reviewed],
    ["已完成复核", reviewed], ["Jev 有效评分率", totalScores ? percent((scores.scored || 0) / totalScores) : "—"]];
  const counts = {};
  rows.forEach(row => row.reasons.forEach(key => { counts[key] = (counts[key] || 0) + 1; }));
  q("#overview").innerHTML = `<h2>批次总览</h2><div class="stat-grid">${cards.map(([name, value]) =>
    `<div class="stat-card"><span>${name}</span><strong>${value}</strong></div>`).join("")}</div>
    ${sources.length ? `<details class="panel inspection-details"><summary>输入文件 · ${sources.length} 份 / ${sources.reduce((n,[,pages])=>n+pages,0)} 页</summary>
      ${sources.map(([doc,pages])=>`<div class="service-row"><strong>${esc(doc)}</strong><span>${pages} 页</span></div>`).join("")}</details>` : ""}
    ${report ? '<section class="panel overview-panel"><header class="panel-heading"><h3>运行时策略对比 · 留出集</h3><select id="strategy-task" aria-label="评测任务"><option value="all">全部任务</option><option value="text">文字</option><option value="table">表格</option></select></header><div id="strategy-grid"></div></section>' : ""}
    <div class="overview-grid">
    <section class="panel overview-panel"><header class="panel-heading"><h3>入队原因分布</h3></header>${Object.entries(counts)
      .sort((a, b) => b[1] - a[1]).map(([key, count]) =>
        `<div class="bar-row"><span>${esc(reasons[key] || key)}</span><meter min="0" max="${rows.length}" value="${count}"></meter><b>${count}</b></div>`).join("") || '<p>没有触发记录。</p>'}</section>
    <section class="panel overview-panel"><header class="panel-heading"><h3>模型状态</h3></header>${Object.entries(summary.model_status)
      .filter(([, counts]) => Object.keys(counts).length).map(([alias, counts]) =>
        `<div class="service-row"><strong>${esc(nameOf(alias))}</strong><span>${Object.entries(counts)
          .map(([key, count]) => esc(key === "ok" ? "有效" : judgeStatuses[key] || key) + " " + count).join(" · ")}</span></div>`)
      .join("")}</section></div>`;
  if (report) {
    q("#strategy-task").addEventListener("change", renderStrategy);
    renderStrategy();
  }
}

function renderStrategy() {
  const report = state.data.report;
  const task = q("#strategy-task").value;
  const strategies = report.strategies[task];
  const labels = {full: "完整挖掘", disagreement: "仅模型分歧", random: "随机抽样"};
  q("#strategy-grid").innerHTML = '<table class="strategy-table"><thead><tr><th>排序策略</th><th>审核量</th>' +
    '<th>发现错误</th><th>有正确替代</th><th>命中率</th><th>召回率</th></tr></thead><tbody>' +
    ["full", "disagreement", "random"].map(key => {
      const row = strategies[key];
      return `<tr><th>${labels[key]}</th><td>${row.reviewed}</td><td>${row.errors_found}</td>
        <td>${row.repairable}</td><td>${percent(row.precision)}</td><td>${percent(row.error_recall)}</td></tr>`;
    }).join("") + '</tbody></table>' + (report.model_metrics?.length ? `<details><summary>各模型参考标注指标</summary><table class="strategy-table"><thead><tr><th>模型</th><th>任务</th><th>样本</th><th>CER</th><th>TEDS</th><th>数字准确率</th></tr></thead><tbody>${
      report.model_metrics.filter(row => task === "all" || row.task === task).map(row =>
        `<tr><th>${esc(nameOf(row.model))}</th><td>${esc(tasks[row.task])}</td><td>${row.cases}</td>
        <td>${percent(row.cer)}</td><td>${percent(row.teds)}</td><td>${percent(row.numeric_accuracy ?? row.numeric_match)}</td></tr>`
      ).join("")}</tbody></table></details>` : "");
}

function view(page) {
  q("#workspace").hidden = page !== "review";
  q("#settings").hidden = page !== "settings";
  q("#view-settings").setAttribute("aria-pressed",page === "settings");
  if (page === "settings") initSettings();
  q("#overview").hidden = page !== "overview";
  q("#view-review").setAttribute("aria-pressed", page === "review");
  q("#view-overview").setAttribute("aria-pressed", page === "overview");
}

q("#view-review").addEventListener("click", () => view("review"));
q("#view-overview").addEventListener("click", () => view("overview"));
q("#batch").addEventListener("change", event => loadBatch(event.target.value));
q("#queue-list").addEventListener("click", event => {
  const item = event.target.closest("[data-case]");
  if (item && !state.saving && !state.loading) selectCase(item.dataset.case, {reveal: true});
});
q("#task-filters").addEventListener("click", event => {
  if (!event.target.dataset.task || !state.data || state.saving) return;
  taskFilter(event.target.dataset.task);
  filterQueue();
});
q("#search").addEventListener("input", () => { if (state.data && !state.saving) filterQueue(); });
for (const id of ["scope", "status", "reason", "judge", "gold"]) {
  q("#" + id).addEventListener("change", () => { if (state.data && !state.saving) filterQueue(); });
}
q("#inspection").addEventListener("click", event => {
  const button = event.target.closest("button");
  if (!button || !state.case) return;
  if (button.id === "previous") changeCase(-1);
  if (button.id === "next") changeCase(1);
  if (button.dataset.model && !state.saving) {
    state.compare = button.dataset.model;
    renderComparison();
    [...q("#model-tiles").children].find(tile => tile.dataset.model === state.compare)?.focus();
  }
  if (button.dataset.move) {
    const axis = button.dataset.move, dir = Number(button.dataset.dir);
    if (axis === "width") previewLeft = Math.max(0, previewLeft + dir * previewSize.width);
    else previewTop = Math.max(0, previewTop + dir * previewSize.height);
    renderComparison();
    q(`[data-move="${axis}"][data-dir="${dir}"]:not(:disabled)`)?.focus();
  }
  if (button.dataset.review) saveReview(button.dataset.review);
  if (button.id === "open-table" || button.hasAttribute("data-open-sheet")) showLargeTable();
  if (button.dataset.image) {
    const isPage = button.dataset.page === "true";
    const geometry = isPage ? state.case.page_geometry : null;
    const dialog = q("#image-dialog");
    q("#dialog-title").textContent = isPage ? (geometry?.location === "manual" ? "原页 · 手动定位"
      : geometry?.box ? "原页 · 当前区域" : "原页 · 位置待核对") : "原始裁图";
    drawing(false);
    q("#draw-location").hidden = !isPage;
    q("#dialog-content").innerHTML = pageFrame(button.dataset.image, geometry);
    dialog.classList.toggle("full-page", isPage);
    dialog.scrollTop = 0;
    dialog.showModal();
    initImageZoom(button.dataset.image, geometry);
  }
});
q("#inspection").addEventListener("change", event => {
  if (event.target.dataset.visible) {
    preferredModels = Array.from(q("#visible-models").querySelectorAll(":checked"), input => input.dataset.visible);
    try { localStorage.setItem(previewKey, JSON.stringify(preferredModels)); } catch {}
    renderComparison();
    Array.from(q("#visible-models").querySelectorAll("input")).find(input =>
      input.dataset.visible === event.target.dataset.visible)?.focus();
  }
  if (event.target.id === "adopt-model") {
    if (!state.saving) state.compare = event.target.value;
    renderComparison();
  }
  if (event.target.id === "only-changes") renderComparison();
  if (event.target.id === "difference") {
    state.difference = event.target.checked;
    renderComparison();
  }
  if (event.target.id === "compare-with") {
    const row = state.case;
    fetchComparison(row,event.target.value).then(() => { if (state.case === row) renderComparison(); }).catch(e=>notice(e.message,true));
  }
});
let previewFrame;
window.addEventListener("resize", () => {
  cancelAnimationFrame(previewFrame);
  previewFrame = requestAnimationFrame(() => {
    if (state.case?.task === "table" && q("#preview-range") && q("#model-pair").clientWidth) renderComparison();
  });
});
q("#table-layout").addEventListener("change", updateTableDialog);
q("#table-zoom").addEventListener("change", updateTableDialog);
q("#table-model").addEventListener("change", event => {
  if (state.saving) {
    event.target.value = state.compare;
    return;
  }
  state.compare = event.target.value;
  renderComparison();
  renderTablePair();
});
q("#table-pass").addEventListener("click", () => saveReview("approved"));
q("#table-reject").addEventListener("click", () => saveReview("rejected"));
for (const item of document.querySelectorAll("[data-export]")) {
  item.addEventListener("click", () => downloadSft(item.dataset.export));
}
api("/api/batches").then(batches => {
  q("#batch").replaceChildren(...Object.entries(batches).map(([id, label]) => new Option(label, id)));
  if (q("#batch").value) return loadBatch(q("#batch").value);
  q("#inspection").innerHTML = '<p class="empty">点击「导入结果」选择命令行生成的批次。</p>';
}).catch(error => notice(error.message, true));

q("#import-open").onclick = async () => {
  const select = q("#import-list"), submit = q("#import-submit");
  q("#import-dialog").showModal();
  select.replaceChildren();
  submit.disabled = true;
  q("#import-message").textContent = "正在读取结果目录…";
  try {
    const data = await api("/api/imports");
    q("#import-root").textContent = data.directory;
    select.replaceChildren(...data.names.map(name => new Option(name, name)));
    submit.disabled = !data.names.length;
    q("#import-message").textContent = data.names.length ? "" : "尚无完成的批次。先用 glyphstack run 处理文件，再重新打开。";
  } catch (error) { q("#import-message").textContent = error.message; }
};
q("#import-submit").onclick = async () => {
  q("#import-submit").disabled = true;
  try {
    const batch = await api("/api/imports/" + encodeURIComponent(q("#import-list").value), {method:"POST"});
    const select = q("#batch");
    if (![...select.options].some(o=>o.value===batch.id)) select.add(new Option(batch.name,batch.id));
    select.value = batch.id;
    q("#import-dialog").close();
    if (q("#settings").hidden) view("review");
    await loadBatch(batch.id);
  } catch (error) { q("#import-message").textContent = error.message; }
  finally { q("#import-submit").disabled = false; }
};
q("#difficulty").onclick = event => {
  if (!event.target.dataset.level || state.saving) return;
  state.level = event.target.dataset.level;
  q("#scope").value = "all";
  filterQueue();
};
async function bulkReview(undo) {
  if (state.saving || !state.data) return;
  const batch = state.batch, key = "undo-" + batch, token = sessionStorage.getItem(key);
  const ids = state.rows.filter(eligible).map(r=>r.case_id);
  state.saving = true;
  q("#batch").disabled = q("#import-open").disabled = true;
  try {
    const result = await api(batchPath() + (undo ? "/bulk/" + encodeURIComponent(token) + "/undo" : "/bulk"), {
      method:"POST", headers:{"Content-Type":"application/json"}, body:undo ? undefined : JSON.stringify(ids)});
    if (undo) sessionStorage.removeItem(key);
    else if (result.changed.length) sessionStorage.setItem(key, result.bulk_id);
    const fresh = await api(batchPath());
    state.data = fresh;
    const current = state.current;
    state.case = null;
    filterQueue();
    if (state.rows.some(row=>row.case_id === current)) await selectCase(current);
    renderOverview();
    notice((undo ? "已撤销 " : "已批量通过 ") + result.changed.length + " 条");
  } catch (error) { notice(error.message,true); }
  finally {
    state.saving = false;
    q("#batch").disabled = q("#import-open").disabled = false;
    if (state.case) renderCase();
    filterQueue();
  }
}
q("#bulk").onclick = () => bulkReview(false);
q("#undo").onclick = () => bulkReview(true);

const sheetRows = 16, sheetCols = 6;
let sheetState;
let sheetRevision=0;
function indexCells(view) {
  const cells = new Map();
  view?.rows?.forEach((row,r)=>row.forEach(cell=>cells.set(r+","+cell.col,cell)));
  return cells;
}

async function renderTablePair() {
  const row=state.case, base=q("#table-base").value, alias=state.compare;
  const revision=++sheetRevision;
  sheetState=null;
  q("#table-prev").disabled=q("#table-next").disabled=true;
  q("#table-dialog-content").textContent="正在准备表格…";
  q("#cell-detail").replaceChildren();
  try {
    await fetchComparison(row,base);
    if (revision!==sheetRevision || row!==state.case || alias!==state.compare || base!==q("#table-base").value) return;
    const views=comparisonViews(row,base);
    const left=base==="reference"?row.reference_view:views[base], right=views[alias];
    const a=indexCells(left), b=indexCells(right);
    const diffs=[];
    for (const key of new Set([...a.keys(),...b.keys()])) {
      const x=a.get(key), y=b.get(key);
      if (!x || !y || x.text!==y.text || x.height!==y.height || x.width!==y.width) {
        const [r,c]=key.split(",").map(Number);
        diffs.push({r,c});
      }
    }
    diffs.sort((x,y)=>x.r-y.r || x.c-y.c);
    sheetState={left,right,base,alias,diffs,idx:-1,top:0,col:0,selected:null,
      rows:Math.max(left?.rows?.length || 0,right?.rows?.length || 0),
      cols:Math.max(sheetWidth(left),sheetWidth(right))};
    q("#table-sync").checked=right?.structure===1;
    q("#table-feedback").textContent=diffs.length ? `${diffs.length} 处坐标差异（含缺失与合并变化）` : "未发现单元格差异";
    paintSheet();
  } catch(error) {q("#table-feedback").textContent=error.message;}
}

function paintSheet() {
  const s=sheetState;
  if (!s) return;
  const content=q("#table-dialog-content");
  const panes=[s.left,s.right];
  const title=alias=>alias==="reference"?"参考标注":nameOf(alias);
  const cards=panes.map((view,i)=>`<section class="output-card"><h4>${esc(title(i?s.alias:s.base))} · ${view?.rows?.length || 0} 行 × ${sheetWidth(view)} 列</h4>
    <div class="output-scroll" tabindex="0">${view?.rows?tableOutput(view,!!i,s.top,s.col,sheetRows,sheetCols,s.selected):"无有效表格"}</div></section>`);
  content.innerHTML=cards.join("");
  const scrolls=[...content.querySelectorAll(".output-scroll")];
  scrolls.forEach((pane,i)=>pane.onscroll=()=>{
    if (q("#table-sync").checked) scrolls[1-i].scrollTo(pane.scrollLeft,pane.scrollTop);
  });
  q("#sheet-range").textContent=`行 ${s.rows?s.top+1:0}–${Math.min(s.rows,s.top+sheetRows)} / ${s.rows}　列 ${s.cols?s.col+1:0}–${Math.min(s.cols,s.col+sheetCols)} / ${s.cols}`;
  q("#sheet-row").max=Math.max(1,s.rows); q("#sheet-col").max=Math.max(1,s.cols);
  q("#sheet-row").value=(s.selected?.r ?? s.top)+1; q("#sheet-col").value=(s.selected?.c ?? s.col)+1;
  const blocked={up:!s.top,down:s.top+sheetRows>=s.rows,left:!s.col,right:s.col+sheetCols>=s.cols};
  for (const [step,disabled] of Object.entries(blocked)) q(`[data-step="${step}"]`).disabled=disabled;
  q("#table-prev").disabled=q("#table-next").disabled=!s.diffs.length;
  const detail=q("#cell-detail");
  if (!s.selected) {detail.textContent="点击单元格查看两份完整内容";return;}
  const {r,c}=s.selected;
  detail.innerHTML=`<strong>第 ${r+1} 行 · 第 ${c+1} 列</strong>`+panes.map((view,i)=>{
    let cell,anchor;
    for(let n=r;n>=0 && !cell;n--) {
      cell=view?.rows?.[n]?.find(x=>n+x.height>r && x.col<=c && x.col+x.width>c);
      if(cell) anchor=n;
    }
    const location=cell && (cell.width>1 || cell.height>1) ? ` · 合并 ${cell.height}×${cell.width}（起点 ${anchor+1}, ${cell.col+1}）`:"";
    return `<div><b>${esc(title(i?s.alias:s.base))}${location}</b><p>${cell ? esc(cell.text || "（空白）") : "（无对应单元格）"}</p></div>`;
  }).join("");
  content.querySelectorAll(".cell-selected").forEach(cell=>{
    const pane=cell.closest(".output-scroll"), rect=cell.getBoundingClientRect(), box=pane.getBoundingClientRect();
    if(rect.width && box.width) pane.scrollBy(Math.max(0,rect.right-box.right),Math.max(0,rect.bottom-box.bottom));
  });
}
function locateCell(r,c) {
  const s=sheetState;
  s.selected={r,c}; s.top=Math.floor(r/sheetRows)*sheetRows; s.col=Math.floor(c/sheetCols)*sheetCols;
  paintSheet();
}
q("#sheet-jump").onsubmit=e=>{
  e.preventDefault();
  if (sheetState) locateCell(q("#sheet-row").valueAsNumber-1,q("#sheet-col").valueAsNumber-1);
};
q(".sheet-tools").onclick=e=>{
  const s=sheetState, step=e.target.dataset.step;
  if (!s || !step) return;
  s.top+=step==="up"?-sheetRows:step==="down"?sheetRows:0;
  s.col+=step==="left"?-sheetCols:step==="right"?sheetCols:0;
  s.selected=null;paintSheet();
};
q("#table-dialog-content").onclick=e=>{
  const td=e.target.closest("td[data-row]");
  if(td && sheetState) locateCell(Math.max(sheetState.top,Number(td.dataset.row)),Math.max(sheetState.col,Number(td.dataset.col)));
};
q("#table-base").onchange=renderTablePair;
for(const [id,direction] of [["table-next",1],["table-prev",-1]]) q("#"+id).onclick=()=>{
  const s=sheetState;
  if (!s?.diffs.length) return;
  s.idx=s.idx<0?(direction===1?0:s.diffs.length-1):(s.idx+direction+s.diffs.length)%s.diffs.length;
  locateCell(s.diffs[s.idx].r,s.diffs[s.idx].c);
  q("#table-feedback").textContent=`差异 ${s.idx+1} / ${s.diffs.length}`;
};

let imageKey, pan;
const imageLocations = new Map();
function resizeImage(width, x, y) {
  const frame = locationContent.querySelector(".page-frame");
  if (!frame || !frame.offsetWidth) return;
  const old = frame.getBoundingClientRect();
  const bounds = locationContent.getBoundingClientRect();
  x ??= bounds.left + bounds.width/2;
  y ??= bounds.top + bounds.height/2;
  const u = (x-old.left)/old.width, v = (y-old.top)/old.height;
  frame.style.width = Math.max(48,Math.min(16000,width)) + "px";
  const next = frame.getBoundingClientRect();
  locationContent.scrollBy(next.left + u*next.width-x,next.top + v*next.height-y);
}
function initImageZoom(key, geometry) {
  imageKey = key;
  const frame = locationContent.querySelector(".page-frame"), img = frame.querySelector("img");
  const init = () => {
    if (!frame.isConnected) return;
    const saved = imageLocations.get(key);
    frame.style.width = (saved?.width || locationContent.clientWidth) + "px";
    if (saved) locationContent.scrollTo(saved.left,saved.top);
    else if (geometry?.box) {
      const [l,t,r,b] = geometry.box;
      locationContent.scrollTo((l+r)/2/geometry.width*frame.offsetWidth-locationContent.clientWidth/2,
        (t+b)/2/geometry.height*frame.offsetHeight-locationContent.clientHeight/2);
    } else locationContent.scrollTo(0,0);
  };
  if (img.complete) init(); else img.onload = init;
}
function imageStep(multiplier, x, y) {
  resizeImage(locationContent.querySelector(".page-frame").offsetWidth*multiplier,x,y);
}
q("#img-minus").onclick = () => imageStep(1/1.35);
q("#img-plus").onclick = () => imageStep(1.35);
q("#img-fit").onclick = () => resizeImage(locationContent.clientWidth);
q("#img-native").onclick = () => resizeImage(locationContent.querySelector("img").naturalWidth);
q("#img-full").onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await locationDialog.requestFullscreen();
  } catch (error) { notice(error.message,true); }
};
locationContent.addEventListener("wheel",e=>{
  if (locationContent.classList.contains("drawing")) return;
  e.preventDefault();
  imageStep(e.deltaY<0 ? 1.15 : 1/1.15,e.clientX,e.clientY);
},{passive:false});
locationContent.addEventListener("pointerdown",e=>{
  if (locationContent.classList.contains("drawing") || e.button!==0) return;
  pan = {x:e.clientX,y:e.clientY,left:locationContent.scrollLeft,top:locationContent.scrollTop};
  locationContent.setPointerCapture(e.pointerId);
});
locationContent.addEventListener("pointermove",e=>{
  if (pan) locationContent.scrollTo(pan.left+pan.x-e.clientX,pan.top+pan.y-e.clientY);
});
for (const type of ["pointerup","pointercancel"]) locationContent.addEventListener(type,()=>{pan=null;});
function rememberImage() {
  const frame = locationContent.querySelector(".page-frame");
  if (frame && locationDialog.open && frame.offsetWidth)
    imageLocations.set(imageKey,{width:frame.offsetWidth,left:locationContent.scrollLeft,top:locationContent.scrollTop});
}
locationContent.onscroll = rememberImage;
locationContent.addEventListener("click",rememberImage);
locationDialog.querySelector(".dialog-actions").addEventListener("click",rememberImage);

async function refreshRuntime() {
  let status;
  try {
    const response = await fetch("/api/runtime", {cache:"no-store"});
    if (response.status === 404) return;
    if (!response.ok) throw new Error("模型后台状态获取失败");
    status = await response.json();
  } catch {
    status = {connected:false};
  }
  if (status.connected === null) return;
  const panel = q("#runtime");
  panel.hidden = false;
  const jobs = status.jobs || [], models = Object.entries(status.models || {});
  const labels = {offline:"离线", starting:"加载中", healthy:"健康", unhealthy:"探测失败", failed:"启动失败", stopping:"释放中"};
  q("#runtime-summary").textContent = status.connected
    ? `模型后台 · ${jobs.length ? jobs.length + " 项任务" : "空闲"}` : "模型后台 · 未连接";
  q("#runtime-detail").innerHTML = models.map(([alias,model]) =>
    `<span><b>${esc(nameOf(alias))}</b> ${esc(labels[model.status] || model.status)}</span>`).join("")
    + jobs.map(job => `<span>${esc(job.value)} · ${esc(job.stage)}</span>`).join("");
  setTimeout(refreshRuntime,4000);
}
refreshRuntime();
