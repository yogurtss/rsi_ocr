"use strict";

const thresholdRules = {
  low_judge: ["Paddle 质量分", "≤", "Jev 对 Paddle 转录的评分。提高阈值会纳入更多低分样本。", "low_target_judge"],
  judge_gain: ["候选质量分增益", "≥", "候选最高 Jev 分减去 Paddle 分，最低取零。降低阈值会纳入更小的增益。", "better_judged_candidate"],
  disagreement: ["跨模型分歧", "≥", "比较教师与 Paddle 的转录距离，再按模型家族汇总。降低阈值会纳入更小的分歧。", "teacher_disagreement"],
};
let previewRevision = 0;
let previewTimer;
let applyingPolicy = false;

function initSettings() {
  loadDeployment();
  clearTimeout(previewTimer);
  previewRevision++;
  const data = state.data;
  const policy = data?.summary.mining;
  q("#settings-batch").textContent = data ? q("#batch").selectedOptions[0]?.text || state.batch : "尚未选择批次";
  q("#settings-empty").hidden = !!data;
  q("#tuning").hidden = !data;
  const readers = policy ? [policy.target, ...policy.teachers] : ["paddle","oxr","ovis"];
  if (policy?.strong_budget && policy.strong_teacher) readers.push(policy.strong_teacher);
  q("#flow-readers").innerHTML = readers.map((key, i) =>
    `<div><strong>${esc(nameOf(key))}</strong><small>${i ? (key === policy?.strong_teacher ? "按需复读" : "候选") : "基线"}</small></div>`).join("");
  document.querySelectorAll("[data-rule]").forEach(button => {
    button.disabled = !data;
    button.setAttribute("aria-pressed","false");
  });
  if (!data) return;
  q("#threshold-controls").innerHTML = Object.entries(thresholdRules).map(([key,[title,op,hint]]) =>
    `<div class="rule-editor" id="editor-${key}">
      <label for="value-${key}">${title} <b>${op}</b></label>
      <input id="value-${key}" type="number" min="0" max="1" step="any" value="${Number(policy[key])}" data-value="${key}">
      <input aria-label="${title}滑块" type="range" min="0" max="1" step="0.001" value="${Number(policy[key])}" data-slider="${key}">
      <p>${hint}</p><small id="hits-${key}"></small></div>`).join("");
  q("#fixed-switches").innerHTML = Object.entries(fixedDescriptions).map(([key,[name,hint]]) =>
    `<label><input type="checkbox" id="rule-${key}" ${policy.fixed_rules?.[key] !== false ? "checked" : ""}><span><strong>${name}</strong> <small id="fixed-hits-${key}"></small><br>${hint}</span></label>`).join("");
  selectRule("low_judge");
  q("#preview-counts").replaceChildren();
  schedulePreview();
}

function selectRule(key, focus = false) {
  document.querySelectorAll("[data-rule]").forEach(button =>
    button.setAttribute("aria-pressed",String(button.dataset.rule === key)));
  document.querySelectorAll(".rule-editor").forEach(editor =>
    editor.classList.toggle("active",editor.id === "editor-" + key));
  if (focus) q("#value-" + key).focus();
}

function readThresholds() {
  const values = {};
  for (const key of Object.keys(thresholdRules)) {
    const field = q("#value-" + key);
    if (!Number.isFinite(field.valueAsNumber) || !field.validity.valid) return null;
    values[key] = field.valueAsNumber;
  }
  values.fixed_rules = Object.fromEntries(Object.keys(fixedDescriptions).map(k=>[k,q("#rule-"+k).checked]));
  return values;
}

function schedulePreview() {
  clearTimeout(previewTimer);
  const revision = ++previewRevision;
  const values = readThresholds();
  q("#apply-policy").disabled = true;
  q("#preview-counts").setAttribute("aria-busy","true");
  for (const key of Object.keys(thresholdRules)) q("#hits-" + key).textContent = "";
  q("#mobile-preview").textContent = values ? "正在试算…" : "阈值应在 0 到 1 之间";
  if (!values) {
    q("#preview-counts").replaceChildren();
    q("#preview-counts").setAttribute("aria-busy","false");
    q("#preview-status").textContent = "请输入 0 到 1 之间的数值。";
    return;
  }
  const path = batchPath(), data = state.data;
  const changed = Object.keys(thresholdRules).some(k=>values[k] !== data.summary.mining[k])
    || Object.keys(fixedDescriptions).some(k=>values.fixed_rules[k] !== (data.summary.mining.fixed_rules?.[k] !== false));
  q("#preview-status").textContent = "正在按当前数值试算…";
  previewTimer = setTimeout(async () => {
    try {
      const result = await api(path + "/policy/preview", {method:"POST",
        headers:{"Content-Type":"application/json"},body:JSON.stringify(values)});
      if (revision !== previewRevision || state.data !== data) return;
      const total = data.rows.length;
      q("#mobile-preview").textContent = `难例 ${result.hard} / ${total}　·　预计入队 ${result.queued}`;
      const difference = result.hard - data.summary.hard_cases;
      const delta = difference ? `比已应用规则${difference > 0 ? "多" : "少"} ${Math.abs(difference)} 条` : "与已应用规则数量相同";
      q("#preview-counts").innerHTML = `<div class="hard-total"><span>难例</span><strong>${result.hard}</strong> / ${total} 条</div>
        <meter aria-label="难例占比" min="0" max="${total || 1}" value="${result.hard}"></meter><p>${delta}</p>
        <dl><div><dt>文字难例</dt><dd>${result.by_task.text || 0}</dd></div><div><dt>表格难例</dt><dd>${result.by_task.table || 0}</dd></div>
        <div><dt>预计入队（含抽检）</dt><dd>${result.queued}</dd></div></dl>`;
      for (const key of Object.keys(fixedDescriptions)) q("#fixed-hits-"+key).textContent = values.fixed_rules[key] ? `命中 ${result.reasons[key] || 0} 条` : "已关闭";
      for (const [key,rule] of Object.entries(thresholdRules))
        q("#hits-" + key).textContent = `命中 ${result.reasons[rule[3]] || 0} 条难例`;
      q("#preview-status").textContent = changed ? "预览中 · 应用后更新本批次复核队列" : "当前批次已应用这些阈值";
      q("#apply-policy").disabled = !changed || applyingPolicy;
    } catch (error) {
      if (revision !== previewRevision) return;
      q("#preview-counts").replaceChildren();
      q("#preview-status").textContent = error.message;
      q("#mobile-preview").textContent = error.message;
    } finally {
      if (revision === previewRevision) q("#preview-counts").setAttribute("aria-busy","false");
    }
  },120);
}

q("#view-settings").onclick = () => view("settings");
q("#settings-import").onclick = () => q("#import-open").click();
q(".flow-sheet").onclick = event => {
  const key = event.target.closest("[data-rule]")?.dataset.rule;
  if (key) selectRule(key,true);
};
q("#threshold-controls").oninput = event => {
  const {value,slider} = event.target.dataset;
  if (slider) q("#value-" + slider).value = event.target.value;
  else if (value && event.target.validity.valid) q(`[data-slider="${value}"]`).value = event.target.value;
  selectRule(value || slider);
  schedulePreview();
};
q("#reset-policy").onclick = () => {
  const original = state.data.original_thresholds;
  for (const key of Object.keys(fixedDescriptions)) q("#rule-"+key).checked = original.fixed_rules?.[key] !== false;
  for (const key of Object.keys(thresholdRules)) {
    const value = original[key];
    q("#value-" + key).value = value;
    q(`[data-slider="${key}"]`).value = value;
  }
  schedulePreview();
};
q("#apply-policy").onclick = async () => {
  const batch = state.batch, values = readThresholds();
  if (!values || applyingPolicy) return;
  applyingPolicy = true;
  q("#tuning").inert = true;
  q("#apply-policy").disabled = true;
  try {
    await api(batchPath() + "/policy",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(values)});
    if (batch === state.batch) await loadBatch(batch);
    notice("已更新批次阈值与复核队列");
  } catch (error) {
    notice(error.message,true);
    if (batch === state.batch) schedulePreview();
  } finally {
    applyingPolicy = false;
    q("#tuning").inert = false;
  }
};

const fixedDescriptions = {
  numeric_conflict: ["数字冲突", "文字按数字序列比较，表格按对应单元格比较。例如基线读作 100，教师读作 1000。"],
  empty: ["空转录", "Paddle 请求完成，但清理后的转录为空。"],
  format_invalid: ["表格格式无效", "Paddle 输出无法按声明的表格格式解析，例如合并单元格占位不合法。"],
  judge_precheck_reject: ["Jev 预检拒绝", "Jev 在正式评分前拒绝 Paddle 转录；该状态记录在原始评分证据中。"],
};
let liveMode;
async function loadDeployment() {
  const select = q("#deployment-mode"), save = q("#save-mode");
  select.disabled = save.disabled = true;
  try {
    const runtime = await api("/api/runtime");
    if (!runtime.connected || !runtime.mode) throw new Error("通过 glyphstack studio 连接后台后可修改部署模式。");
    select.value = liveMode = runtime.mode;
    select.disabled = false;
    q("#mode-status").textContent = "模式保存到后台使用的配置文件，立即生效。";
  } catch (error) { q("#mode-status").textContent = error.message; }
}
q("#deployment-mode").onchange = e => { q("#save-mode").disabled = e.target.value === liveMode; };
q("#save-mode").onclick = async () => {
  const select = q("#deployment-mode"), mode = select.value;
  select.disabled = q("#save-mode").disabled = true;
  try {
    await api("/api/runtime/deployment/" + mode,{method:"PUT"});
    liveMode = mode;
    q("#mode-status").textContent = "已保存部署模式。";
  } catch (error) {
    select.value = liveMode;
    q("#mode-status").textContent = error.message;
  } finally { select.disabled = false; }
};
q("#fixed-switches").onchange = schedulePreview;
