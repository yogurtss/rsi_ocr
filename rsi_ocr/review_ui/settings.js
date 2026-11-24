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
  showFlowReaders();
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
let savedDeployment;
let loadingDeployment = false;

function showFlowReaders() {
  const policy = state.data?.summary.mining;
  const readers = policy ? [policy.target, ...policy.teachers] : ["paddle", "oxr", "ovis"].filter(
    alias => savedDeployment?.models[alias]?.enabled !== false);
  if (policy?.strong_budget && policy.strong_teacher) readers.push(policy.strong_teacher);
  q("#flow-readers").innerHTML = readers.map((key, i) =>
    `<div><strong>${esc(nameOf(key))}</strong><small>${i ? (key === policy?.strong_teacher ? "按需复读" : "候选") : "基线"}</small></div>`).join("");
}

function readDeployment() {
  const models = Object.fromEntries([...document.querySelectorAll(".deployment-model")].map(row => {
    const alias = row.dataset.model;
    const path = row.querySelector("[name=path]").value.trim();
    if (alias === "layout") {
      const device = q("#layout-device").value === "gpu" ? `gpu:${q("#layout-gpu").value}` : "cpu";
      return [alias, {path, device}];
    }
    return [alias, {
      enabled: row.querySelector("[name=enabled]").checked,
      path,
      gpu: row.querySelector("[name=gpu]").value.trim(),
    }];
  }));
  return {mode:q("#deployment-mode").value, models};
}

function updateDeploymentButton() {
  const layoutDevice = q("#layout-device");
  const useGpu = layoutDevice?.value === "gpu";
  if (layoutDevice) {
    q("#layout-gpu-field").hidden = !useGpu;
    q("#layout-gpu").disabled = !useGpu;
  }
  const current = readDeployment();
  q("#save-deployment").disabled = !savedDeployment || !Object.values(current.models).every(model => model.path)
    || (useGpu && !/^gpu:[0-9]+$/.test(current.models.layout.device))
    || JSON.stringify(current) === JSON.stringify(savedDeployment);
}

function layoutDeploymentRow(layout) {
  const [device, gpu = "0"] = layout.device.split(":");
  return `<div class="deployment-model" data-model="layout">
    <label><input type="checkbox" checked disabled aria-label="版面检测已启用"> <strong>版面检测</strong> <small>必开</small></label>
    <label>权重路径<input type="text" name="path" value="${esc(layout.path)}" required></label>
    <div class="layout-device-options">
      <label>设备 <select id="layout-device"><option value="cpu">CPU</option><option value="gpu" ${device === "gpu" ? "selected" : ""}>GPU</option></select></label>
      <label id="layout-gpu-field" ${device === "gpu" ? "" : "hidden"}>GPU 编号 <input id="layout-gpu" type="number" min="0" step="1" value="${esc(gpu)}" required></label>
    </div>
  </div>`;
}

async function loadDeployment(message) {
  if (loadingDeployment || (savedDeployment && JSON.stringify(readDeployment()) !== JSON.stringify(savedDeployment))) return;
  loadingDeployment = true;
  const fields = q("#deployment-fields"), status = q("#deployment-status");
  fields.disabled = true;
  try {
    const runtime = await api("/api/runtime");
    if (!runtime.connected || !runtime.mode) throw new Error("通过 glyphstack studio 连接后台后可修改模型设置。");
    q("#deployment-mode").value = runtime.mode;
    q("#deployment-models").innerHTML = layoutDeploymentRow(runtime.layout) + Object.entries(runtime.models).map(([alias, model]) =>
      `<div class="deployment-model" data-model="${esc(alias)}">
        <label><input type="checkbox" name="enabled" ${model.enabled !== false ? "checked" : ""}
          ${alias === "paddle" ? "disabled" : ""}> <strong>${esc(model.display_name || nameOf(alias))}</strong>
          ${alias === "paddle" ? "<small>必开</small>" : ""}</label>
        <label>权重路径<input type="text" name="path" value="${esc(model.configured_path ?? model.path)}" required></label>
        <label>GPU<input type="text" name="gpu" value="${esc(model.configured_gpu)}" placeholder="${esc(model.gpu)}"></label>
      </div>`).join("");
    savedDeployment = readDeployment();
    fields.disabled = false;
    status.textContent = message || "相对路径基于 YAML 的 model_root。GPU 留空时使用默认设备。";
    showFlowReaders();
  } catch (error) {
    status.textContent = message ? message + " " + error.message : error.message;
  } finally {
    loadingDeployment = false;
    updateDeploymentButton();
  }
}

q("#deployment-form").oninput = updateDeploymentButton;
q("#deployment-form").onsubmit = async event => {
  event.preventDefault();
  if (q("#save-deployment").disabled) return;
  const fields = q("#deployment-fields"), status = q("#deployment-status");
  const current = readDeployment();
  const models = Object.fromEntries(Object.entries(current.models).filter(([alias, model]) =>
    JSON.stringify(model) !== JSON.stringify(savedDeployment.models[alias])));
  loadingDeployment = fields.disabled = q("#save-deployment").disabled = true;
  status.textContent = "正在保存模型设置…";
  try {
    await api("/api/runtime/deployment", {method:"PUT", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({mode:current.mode, models})});
    savedDeployment = current;
    loadingDeployment = false;
    await loadDeployment(current.mode === "managed"
      ? "已保存到 YAML。后台将并行加载已启用的模型，进度见「模型后台」。"
      : "已保存到 YAML。后续批次将逐个加载模型。");
  } catch (error) {
    status.textContent = error.message;
    fields.disabled = false;
  } finally {
    loadingDeployment = false;
    updateDeploymentButton();
  }
};
q("#fixed-switches").onchange = schedulePreview;
