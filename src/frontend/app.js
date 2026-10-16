// Dashboard client for the Drug Response Prediction API.
const $ = (sel) => document.querySelector(sel);
const state = { token: null, user: null, drugs: [], charts: {} };
try { state.token = localStorage.getItem("token"); } catch (e) { /* storage blocked */ }

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));

async function api(path, opts = {}) {
  const headers = opts.headers || {};
  if (state.token) headers.Authorization = "Bearer " + state.token;
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401 && state.token) { logout(); throw new Error("Session expired, please sign in again"); }
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(body?.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : res.statusText);
  return body;
}

function setToken(token, user) {
  state.token = token; state.user = user;
  try { token ? localStorage.setItem("token", token) : localStorage.removeItem("token"); } catch (e) {}
}

function logout() { setToken(null, null); renderShell(); }

function show(view) {
  for (const v of ["login", "predict", "history", "model", "admin"]) $("#view-" + v).classList.toggle("hidden", v !== view);
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  if (view === "history") loadHistory();
  if (view === "model") loadModel();
  if (view === "admin") loadUsers();
}

async function renderShell() {
  const loggedIn = !!state.token;
  $("#nav").classList.toggle("hidden", !loggedIn);
  $("#who").classList.toggle("hidden", !loggedIn);
  if (!loggedIn) return show("login");
  try { state.user = await api("/api/auth/me"); } catch (e) { return; }
  $("#who-email").textContent = state.user.email;
  $("#who-role").textContent = state.user.role;
  $("#nav-admin").classList.toggle("hidden", state.user.role !== "admin");
  show("predict");
  loadSamples();
  loadDrugs();
}

// ---------------------------------------------------------------- auth
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  try {
    const r = await api("/api/auth/login", { method: "POST", json: { email: f.get("email"), password: f.get("password") } });
    setToken(r.token, r.user); $("#login-error").textContent = ""; renderShell();
  } catch (err) { $("#login-error").textContent = err.message; }
});
$("#signup-btn").addEventListener("click", async () => {
  const f = new FormData($("#login-form"));
  try {
    const r = await api("/api/auth/signup", { method: "POST", json: { email: f.get("email"), password: f.get("password") } });
    setToken(r.token, r.user); renderShell();
  } catch (err) { $("#login-error").textContent = err.message; }
});
$("#logout").addEventListener("click", logout);
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));

// ---------------------------------------------------------------- samples & drugs
async function loadSamples(selectId) {
  const samples = await api("/api/samples");
  const sel = $("#sample-select");
  sel.innerHTML = samples.length ? "" : "<option value=''>Upload a sample first</option>";
  for (const s of samples) {
    const counts = Object.entries(s.feature_counts || {}).map(([m, n]) => `${m}:${n}`).join(" ");
    sel.insertAdjacentHTML("beforeend", `<option value="${s.id}">#${s.id} ${esc(s.name)} ${s.cancer_type ? "(" + esc(s.cancer_type) + ")" : ""} – ${esc(counts)}</option>`);
  }
  if (selectId) sel.value = selectId;
}

async function loadDrugs() {
  try { state.drugs = await api("/api/drugs"); } catch (e) { $("#predict-msg").textContent = e.message; return; }
  renderDrugs();
}

function renderDrugs() {
  const q = $("#drug-filter").value.toLowerCase();
  if (!state.selected) state.selected = new Set(state.drugs.map((d) => d.drug_id));
  $("#drug-list").innerHTML = state.drugs
    .filter((d) => d.drug_name.toLowerCase().includes(q))
    .map((d) => `<label><input type="checkbox" value="${esc(d.drug_id)}" ${state.selected.has(d.drug_id) ? "checked" : ""}> ${esc(d.drug_name)}</label>`)
    .join("");
}
$("#drug-filter").addEventListener("input", renderDrugs);
$("#drug-list").addEventListener("change", (e) => {
  e.target.checked ? state.selected.add(e.target.value) : state.selected.delete(e.target.value);
});
$("#all-drugs").addEventListener("click", () => { state.selected = new Set(state.drugs.map((d) => d.drug_id)); renderDrugs(); });
$("#no-drugs").addEventListener("click", () => { state.selected = new Set(); renderDrugs(); });

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector("button"); btn.disabled = true;
  $("#upload-msg").textContent = "Uploading…";
  try {
    const created = await api("/api/samples", { method: "POST", body: new FormData(e.target) });
    $("#upload-msg").textContent = `Uploaded ${created.length} sample(s).`;
    await loadSamples(created[0].id);
    e.target.reset();
  } catch (err) { $("#upload-msg").textContent = "Error: " + err.message; }
  btn.disabled = false;
});

// ---------------------------------------------------------------- predict
$("#predict-btn").addEventListener("click", async () => {
  const sampleId = parseInt($("#sample-select").value, 10);
  const drugIds = [...(state.selected || [])];
  if (!sampleId) return ($("#predict-msg").textContent = "Upload or select a sample first.");
  if (!drugIds.length) return ($("#predict-msg").textContent = "Select at least one drug.");
  const btn = $("#predict-btn"); btn.disabled = true; $("#predict-msg").textContent = "Running model…";
  try {
    const r = await api("/api/predict", { method: "POST", json: { sample_id: sampleId, drug_ids: drugIds, top_k: 10 } });
    $("#predict-msg").textContent = "";
    renderResults(r);
  } catch (err) { $("#predict-msg").textContent = "Error: " + err.message; }
  btn.disabled = false;
});

function chart(id, config) {
  if (state.charts[id]) state.charts[id].destroy();
  state.charts[id] = new Chart(document.getElementById(id), config);
}

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function renderResults(r) {
  $("#results").classList.remove("hidden");
  $("#result-meta").textContent = `· ${r.sample.name} · model ${r.model_version}`;
  const cov = Object.entries(r.coverage).map(([m, c]) => `${m} ${(c * 100).toFixed(0)}%`).join(", ");
  $("#coverage").textContent = "Feature coverage vs. model inputs: " + cov + ". Missing modalities are masked out by the model.";
  const preds = r.predictions;
  $("#result-rows").innerHTML = preds.map((p, i) => `
    <tr class="clickable" data-i="${i}">
      <td>${esc(p.drug_name)}</td><td class="num">${fmt(p.predicted_ic50_um, 2)}</td>
      <td class="num">${fmt(p.predicted_ln_ic50)}</td>
      <td><span class="badge ${p.sensitivity}">${p.sensitivity}</span></td>
      <td><button class="link">Explain</button></td></tr>`).join("");
  document.querySelectorAll("#result-rows tr").forEach((tr) => tr.addEventListener("click", () => renderExplain(preds[tr.dataset.i])));

  chart("ic50-chart", {
    type: "bar",
    data: {
      labels: preds.map((p) => p.drug_name),
      datasets: [{
        label: "Sensitivity score (drug median ln IC50 − predicted)",
        data: preds.map((p) => p.sensitivity_score),
        backgroundColor: preds.map((p) => (p.sensitivity === "Sensitive" ? cssVar("--sens") : cssVar("--res"))),
      }],
    },
    options: {
      indexAxis: "y", maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `score ${fmt(c.raw, 2)} · IC50 ${fmt(preds[c.dataIndex].predicted_ic50_um, 2)} µM` } } },
      scales: { x: { title: { display: true, text: "← more resistant      more sensitive →" }, ticks: { color: cssVar("--muted") } },
                y: { ticks: { color: cssVar("--text") } } },
    },
  });
  if (preds.length) renderExplain(preds[0]);
}

function renderExplain(p) {
  $("#explain").classList.remove("hidden");
  $("#explain-drug").textContent = "– " + p.drug_name;
  const att = p.explanation?.modality_attention || {};
  chart("attn-chart", {
    type: "bar",
    data: { labels: Object.keys(att), datasets: [{ data: Object.values(att), backgroundColor: cssVar("--primary") }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } },
               scales: { y: { beginAtZero: true, ticks: { color: cssVar("--muted") } }, x: { ticks: { color: cssVar("--text") } } } },
  });
  $("#feature-rows").innerHTML = (p.explanation?.top_features || []).map((f) => `
    <tr><td>${esc(f.feature)}</td><td>${esc(f.modality)}</td>
    <td><span class="badge ${f.direction === "sensitivity" ? "Sensitive" : "Resistant"}">${f.direction === "sensitivity" ? "→ sensitive" : "→ resistant"}</span></td></tr>`).join("");
}

// ---------------------------------------------------------------- history / model / admin
function tile(k, v) { return `<div class="tile"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`; }

async function loadHistory() {
  const [stats, preds, samples] = await Promise.all([api("/api/stats"), api("/api/predictions?limit=200"), api("/api/samples")]);
  const names = Object.fromEntries(samples.map((s) => [s.id, s.name]));
  const sens = Object.values(stats.by_drug).reduce((a, v) => a + v.Sensitive, 0);
  $("#stat-tiles").innerHTML = tile("Samples", stats.samples) + tile("Predictions", stats.predictions) +
    tile("Predicted sensitive", stats.predictions ? Math.round((100 * sens) / stats.predictions) + "%" : "–");
  $("#history-rows").innerHTML = preds.map((p) => `<tr><td>${new Date(p.created_at).toLocaleString()}</td>
    <td>${esc(names[p.sample_id] || "#" + p.sample_id)}</td><td>${esc(p.drug_name)}</td><td class="num">${fmt(p.predicted_ic50_um, 2)}</td>
    <td><span class="badge ${p.sensitivity}">${p.sensitivity}</span></td><td class="muted small">${esc(p.model_version)}</td></tr>`).join("");
}

async function loadModel() {
  const info = await api("/api/model");
  $("#model-info").innerHTML = `<p>Version <code>${esc(info.version)}</code> · target <code>${esc(info.target)}</code> · ${info.n_drugs} drugs</p>
    <p class="muted">Inputs: ${Object.entries(info.n_features).map(([m, n]) => `${esc(m)} (${n} features)`).join(", ")}.
    Architecture: per-omics encoders + GNN drug encoder fused by a Transformer.</p>`;
  const t = info.metrics?.test || {};
  const keys = [["pearson", "Pearson r"], ["spearman", "Spearman ρ"], ["rmse", "RMSE (ln IC50)"], ["r2", "R²"],
                ["roc_auc", "ROC-AUC (sensitive)"], ["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["n", "Test pairs"]];
  $("#metric-tiles").innerHTML = keys.filter(([k]) => k in t).map(([k, label]) => tile(label, k === "n" ? t[k] : fmt(t[k]))).join("");
}

async function loadUsers() {
  const users = await api("/api/users");
  $("#user-rows").innerHTML = users.map((u) => `<tr><td>${esc(u.email)}</td><td><span class="badge">${esc(u.role)}</span></td></tr>`).join("");
}
$("#user-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  try {
    await api("/api/users", { method: "POST", json: Object.fromEntries(f) });
    $("#user-msg").textContent = "User created."; e.target.reset(); loadUsers();
  } catch (err) { $("#user-msg").textContent = "Error: " + err.message; }
});

renderShell();
