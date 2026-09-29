"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const state = { file: null, info: null };

async function api(path, fd) {
  const res = await fetch("/api" + path, { method: "POST", body: fd });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}
const opt = (list, sel) => list.map((c) => `<option value="${esc(c)}"${c === sel ? " selected" : ""}>${esc(c)}</option>`).join("");
const guess = (cols, re) => cols.find((c) => re.test(c)) || cols[0] || "";

async function loadFile(file) {
  state.file = file;
  $("#srcName").textContent = `Reading ${file.name}...`;
  $("#formArea").hidden = true;
  try {
    state.info = await api("/describe", (() => { const fd = new FormData(); fd.append("file", file); return fd; })());
  } catch (e) {
    $("#srcName").textContent = "No dataset loaded";
    $("#out").innerHTML = `<div class="msg" role="alert">${esc(e.message)}</div>`;
    return;
  }
  const info = state.info;
  $("#srcName").textContent = `${file.name} - ${info.rows.toLocaleString()} rows`;
  $("#dateCol").innerHTML = opt(info.date.length ? info.date : ["(none found)"], info.date[0]);
  $("#metricCol").innerHTML = opt(info.numeric, guess(info.numeric, /revenue|profit|sales|amount|total/));
  $("#dims").innerHTML = opt(info.categorical);
  $("#chips").innerHTML = info.categorical.length
    ? [`Should we exit the worst-performing ${info.categorical[0]}?`, "Should we approve this budget increase?", "Is this decline a one-off or a trend?"]
        .map((c) => `<button type="button" class="chip">${esc(c)}</button>`).join("")
    : "";
  $("#chips").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => { $("#decision").value = b.textContent; }));
  $("#formArea").hidden = false;
  $("#out").innerHTML = "";
}

function verdictClass(ruling) {
  return ruling === "PROCEED" ? "proceed" : ruling === "DO NOT PROCEED" ? "stop" : "conditions";
}
function gavel(ruling) { return ruling === "PROCEED" ? "✅" : ruling === "DO NOT PROCEED" ? "⛔" : "⚠️"; }

function claimHtml(c) {
  const tag = c.evidence_ids.length ? `<span class="evid-tag">[${c.evidence_ids.join(", ")}]</span>` : "";
  return `<div class="claim"><span class="status ${c.status}">${c.status}</span>${esc(c.text)} ${tag}</div>`;
}

function renderResult(r) {
  const v = r.verdict, cls = verdictClass(v.ruling);
  const bars = Array.from({ length: 10 }, (_, i) => `<span class="${i < Math.round(r.grounding.overall * 10) ? "on" : ""}"></span>`).join("");
  const list = (items, empty) => items.length ? `<ul>${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>` : `<p class="empty">${empty}</p>`;

  const roundsHtml = r.rounds.map((rd) => `
    <div class="round">
      <div class="round-title">${esc(rd.name)}</div>
      <div class="bench">
        <div class="side for"><h3>For the decision</h3>${rd.for.length ? rd.for.map(claimHtml).join("") : '<p class="empty">No claims.</p>'}</div>
        <div class="side against"><h3>Against the decision</h3>${rd.against.length ? rd.against.map(claimHtml).join("") : '<p class="empty">No claims.</p>'}</div>
      </div>
    </div>`).join("");

  const evidenceHtml = r.evidence.map((e) => `<div class="evid-item"><span class="evid-id">${e.id}</span>${esc(e.text)}</div>`).join("");
  const deciding = r.verdict.deciding_evidence.map((id) => `<span class="pill">${esc(id)}</span>`).join("") || '<span class="empty">none</span>';

  $("#out").innerHTML = `
    <section class="panel verdict">
      <div class="gavel">${gavel(v.ruling)}</div>
      <div class="ruling ${cls}">${esc(v.ruling)}</div>
      <div class="conf-wrap">
        <div class="conf-bar"><div class="conf-fill" style="width:${v.confidence}%"></div></div>
        <div class="conf-label">${v.confidence}% confidence (capped at ${v.confidence_cap}% by evidence grounding)</div>
      </div>
      <p class="summary">${esc(v.summary)}</p>
      <p>Deciding evidence: ${deciding}</p>
    </section>

    <section class="panel">
      <div class="grid2">
        <div class="list-panel"><h3>Conditions to satisfy</h3>${list(v.conditions, "None given.")}</div>
        <div class="list-panel"><h3>Risks noted</h3>${list(v.risks, "None given.")}</div>
      </div>
      <div class="list-panel" style="margin-top:16px"><h3>What would change this ruling</h3>${list(v.what_would_change_my_mind, "Not given.")}</div>
    </section>

    <section class="panel">
      <h2 style="margin-top:0">Transcript</h2>
      ${roundsHtml}
      <p class="empty" style="text-align:center">Grounding: for side ${(r.grounding.for * 100).toFixed(0)}%, against side ${(r.grounding.against * 100).toFixed(0)}% of claims verified against the data.<br>
      <span class="grounding-bar" style="display:inline-flex;width:160px;vertical-align:middle">${bars}</span></p>
    </section>

    <details class="evid">
      <summary>Full evidence docket (${r.evidence.length} items) - compared last ${r.case.period_days} days (${r.case.windows.recent[0]} to ${r.case.windows.recent[1]}) against the prior ${r.case.period_days} (${r.case.windows.prior[0]} to ${r.case.windows.prior[1]})</summary>
      ${evidenceHtml}
    </details>`;
}

$("#fileInput").addEventListener("change", (e) => e.target.files[0] && loadFile(e.target.files[0]));
$("#sampleBtn").addEventListener("click", async () => {
  const blob = await (await fetch("sample_business.csv")).blob();
  loadFile(new File([blob], "sample_business.csv", { type: "text/csv" }));
});

$("#goBtn").addEventListener("click", async () => {
  const decision = $("#decision").value.trim();
  const dateCol = $("#dateCol").value, metricCol = $("#metricCol").value;
  const dims = Array.from($("#dims").selectedOptions).map((o) => o.value);
  const btn = $("#goBtn"), label = btn.textContent;
  if (!decision) { $("#out").innerHTML = '<div class="msg" role="alert">Describe the decision to try first.</div>'; return; }
  btn.disabled = true; btn.textContent = "The court is in session...";
  $("#out").innerHTML = '<p class="empty" style="text-align:center;margin-top:20px">Hearing arguments and checking every citation against your data - this can take 15 to 40 seconds...</p>';
  try {
    const fd = new FormData();
    fd.append("file", state.file); fd.append("decision", decision);
    fd.append("date_col", dateCol); fd.append("metric_col", metricCol);
    fd.append("dims", dims.join(",")); fd.append("period_days", $("#periodDays").value || "0");
    renderResult(await api("/hearing", fd));
  } catch (e) {
    $("#out").innerHTML = `<div class="msg" role="alert">${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = label;
  }
});

fetch("/api/health").then((r) => r.json()).then((h) => {
  $("#llmStatus").textContent = h.llm_provider ? `AI provider: ${h.llm_provider}` : "Add a free API key to enable hearings (see README) - GEMINI_API_KEY, GROQ_API_KEY or OPENROUTER_API_KEY";
}).catch(() => { $("#llmStatus").textContent = "API not reachable"; });
