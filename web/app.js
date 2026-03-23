const state = {
  user: null,
  role: null,
  game: null,
  question: null,
  modal: null,
  figureModal: null,
  adminFigures: [],
};

function qs(id) {
  return document.getElementById(id);
}

function setText(id, text) {
  const node = qs(id);
  if (node) node.textContent = text ?? "";
}

function setHtml(id, html) {
  const node = qs(id);
  if (node) node.innerHTML = html ?? "";
}

function show(id, visible = true) {
  const node = qs(id);
  if (!node) return;
  node.classList.toggle("hidden", !visible);
}

async function api(path, payload = null, method = "POST") {
  const options = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (payload) options.body = JSON.stringify(payload);
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Error del servidor");
  }
  return await response.json();
}

function predictionText(prediction) {
  if (!prediction) return "La IA aun no evaluo este ejercicio.";
  const map = { baja: "facil", media: "intermedio", alta: "retador" };
  const label = map[prediction.difficulty_label] || prediction.difficulty_label;
  const prob = Math.round((prediction.probability || 0) * 100);
  return `La IA estima que este ejercicio es ${label} (confianza: ${prob}%).`;
}

function updatePredictionPanel() {
  const latest = state.game?.latest_prediction;
  setText("predDiff", latest?.difficulty_label || "-");
  setText("predConf", (latest?.probability ?? 0).toFixed(2));
  setText("predPoints", String(state.game?.latest_points ?? 0));
  setText("iaBanner", predictionText(latest));
}

function updateStatsPanel() {
  const g = state.game || {};
  setText("statRound", String(g.round ?? 1));
  setText("statLevel", String(g.current_level ?? 3));
  setText("statScore", String(g.score ?? 0));
  setText("statStreak", String(g.streak ?? 0));
  setText("statAttempts", String(g.question_attempts ?? 0));

  const rows = (g.history || []).slice().reverse().map((row, idx) => {
    const globalIndex = (g.history.length - 1) - idx;
    return `
      <tr>
        <td>${row.ronda ?? "-"}</td>
        <td>${row.pregunta ?? "-"}</td>
        <td>${row.respuesta_usuario ?? "-"}</td>
        <td>${row.es_correcta === 1 ? "Si" : "No"}</td>
        <td>${row.nivel_antes ?? "-"} -> ${row.nivel_despues ?? "-"}</td>
        <td>${row.prediccion_modelo ?? "-"}</td>
        <td>${Number(row.confianza_modelo ?? 0).toFixed(2)}</td>
        <td><button class="btn btn-sm btn-outline-info" data-index="${globalIndex}">Ver mas</button></td>
      </tr>
    `;
  }).join("");

  setHtml("historyBody", rows || "<tr><td colspan='8'>Sin historial aun.</td></tr>");
  qs("historyBody")?.querySelectorAll("button[data-index]").forEach((btn) => {
    btn.addEventListener("click", () => openModelModal(Number(btn.dataset.index)));
  });
}

function renderQuestion() {
  const q = state.question;
  if (!q) {
    setText("questionText", "Presiona > para cargar una pregunta");
    setText("hintText", "");
    return;
  }
  setText("questionText", q.text);
  const showHint = state.game?.show_hint;
  setText("hintText", showHint ? q.hint : "");
}

function updateAll() {
  updatePredictionPanel();
  updateStatsPanel();
  renderQuestion();
}

function openModelModal(index) {
  const item = state.game?.history?.[index];
  if (!item) return;

  const payload = {
    inputs_modelo: item.model_inputs || {},
    salida_modelo: item.model_output || {},
    accion_adaptativa: item.accion_adaptativa,
    nivel_antes: item.nivel_antes,
    nivel_despues: item.nivel_despues,
    puntaje_ronda: item.puntaje_ronda,
  };

  setText("modalTitle", `Detalle de modelo - Ronda ${item.ronda ?? "-"}`);
  setText("modalQuestion", item.pregunta || "-");
  setText("modalAnswer", String(item.respuesta_usuario ?? "-"));
  setText("modalCorrect", String(item.respuesta_correcta ?? "-"));
  setText("modalPrediction", `${item.prediccion_modelo ?? "-"} (${Number(item.confianza_modelo ?? 0).toFixed(2)})`);
  setText("modalAction", item.accion_adaptativa || "-");
  setText("modalPayload", JSON.stringify(payload, null, 2));

  state.modal.show();
}

async function loadProgress() {
  const data = await api("/api/player/load-progress", { username: state.user });
  state.game = data.state;
  state.question = data.state.current_question;
  updateAll();
}

async function saveProgress() {
  await api("/api/player/save-progress", { username: state.user });
  alert("Progreso guardado.");
}

async function resetProgress() {
  const data = await api("/api/player/reset-progress", { username: state.user });
  state.game = data.state;
  state.question = data.state.current_question;
  updateAll();
}

async function logout() {
  await api("/api/auth/logout", { username: state.user });
  state.user = null;
  state.role = null;
  state.game = null;
  state.question = null;
  show("authSection", true);
  show("playerSection", false);
  show("adminSection", false);
}

async function nextQuestion() {
  const data = await api("/api/game/next-question", { username: state.user });
  state.question = data.question;
  state.game = data.state;
  setText("gameMessage", "Nueva pregunta cargada. Resuelve y presiona Responder.");
  updateAll();
}

async function requestHint() {
  const data = await api("/api/game/hint", { username: state.user });
  if (!data.ok) {
    setText("gameMessage", data.message || "No hay pregunta activa.");
    return;
  }
  state.game = data.state;
  setText("hintText", data.hint || "");
  setText("gameMessage", "Pista mostrada.");
}

async function submitAnswer() {
  const answer = qs("answerInput").value;
  if (!answer) {
    setText("gameMessage", "Ingresa una respuesta entera.");
    return;
  }
  const payload = {
    username: state.user,
    student_id: state.user,
    answer,
  };
  const data = await api("/api/game/submit", payload);
  state.game = data.state;
  if (data.is_correct) {
    setText("gameMessage", `Correcto. ${data.message} Usa > para la siguiente pregunta.`);
    qs("answerInput").value = "";
    state.question = null;
  } else {
    setText("gameMessage", `Aun no es correcto. ${data.message} Intenta de nuevo.`);
    state.question = state.game.current_question;
  }
  updateAll();
}

function figureModelTag(fileName) {
  const lower = String(fileName || "").toLowerCase();
  if (!lower.startsWith("model_")) return "general";
  const parts = lower.replace(".png", "").split("_");
  if (parts.length < 3) return "modelo";
  const tail = parts.slice(1, -2);
  return tail.join("_") || "modelo";
}

function setupFigureFilter() {
  const select = qs("figureModelFilter");
  if (!select) return;
  const models = Array.from(new Set(state.adminFigures.map((f) => figureModelTag(f.name)))).sort();
  const options = ["all", ...models].map((m) =>
    `<option value="${m}">${m === "all" ? "Todos" : m}</option>`
  );
  select.innerHTML = options.join("");
}

function renderAdminFigures() {
  const selected = qs("figureModelFilter")?.value || "all";
  const filtered = state.adminFigures.filter((f) => {
    if (selected === "all") return true;
    return figureModelTag(f.name) === selected;
  });

  const figCards = filtered.map((fig, idx) => `
    <div class="col-md-6">
      <div class="border rounded p-2 h-100">
        <div class="small text-muted mb-1">${fig.name}</div>
        <img src="${fig.url}" alt="${fig.name}" class="img-fluid rounded admin-figure" loading="lazy" data-index="${idx}">
      </div>
    </div>
  `).join("");

  setHtml("adminFigures", figCards || "<div class='col-12 text-muted small'>No hay graficas para este filtro.</div>");
  qs("adminFigures")?.querySelectorAll(".admin-figure").forEach((img) => {
    img.addEventListener("click", () => {
      const idx = Number(img.dataset.index);
      const item = filtered[idx];
      if (!item) return;
      setText("figureModalTitle", item.name);
      qs("figureModalImage").src = item.url;
      qs("figureModalImage").alt = item.name;
      state.figureModal.show();
    });
  });
}

async function loadAdmin() {
  const data = await api("/api/admin/summary", null, "GET");
  let figuresData = { figures: [] };
  try {
    figuresData = await api("/api/admin/figures", null, "GET");
  } catch (err) {
    setText("adminTestMsg", "No se pudieron cargar las graficas.");
  }

  const leadRows = (data.leaderboard || []).map((row) => `
    <tr>
      <td>${row.model_name ?? "-"}</td>
      <td>${Number(row.accuracy ?? 0).toFixed(4)}</td>
      <td>${Number(row.precision ?? 0).toFixed(4)}</td>
      <td>${Number(row.recall ?? 0).toFixed(4)}</td>
      <td>${Number(row.f1_weighted ?? row.f1_score ?? 0).toFixed(4)}</td>
      <td>${Number(row.f1_macro ?? 0).toFixed(4)}</td>
      <td>${Number(row.auc_ovr_weighted ?? row.auc ?? 0).toFixed(4)}</td>
    </tr>
  `).join("");
  setHtml("adminLeaderboard", leadRows || "<tr><td colspan='7'>Sin datos.</td></tr>");

  const logRows = (data.logs || []).slice().reverse().map((row) => `
    <tr>
      <td>${row.timestamp ?? "-"}</td>
      <td>${row.username ?? "-"}</td>
      <td>${row.question_text ?? "-"}</td>
      <td>${row.user_answer ?? "-"}</td>
      <td>${row.predicted_difficulty ?? "-"}</td>
      <td>${Number(row.predicted_probability ?? 0).toFixed(2)}</td>
      <td>${row.recommended_action ?? "-"}</td>
    </tr>
  `).join("");
  setHtml("adminLogs", logRows || "<tr><td colspan='7'>Sin logs aun.</td></tr>");

  renderAdminMetrics(data.metrics || {});

  state.adminFigures = figuresData.figures || [];
  setupFigureFilter();
  renderAdminFigures();
}

function renderAdminMetrics(metrics) {
  const overall = metrics?.overall;
  const perUser = metrics?.per_user || [];

  if (!overall) {
    setHtml("adminMetricsOverall", "<tr><td colspan='2'>Sin métricas (aún no hay logs).</td></tr>");
    setHtml("adminMetricsUsers", "<tr><td colspan='6'>Sin datos.</td></tr>");
    return;
  }

  const rows = [
    ["Intentos", overall.attempts],
    ["Usuarios únicos", overall.unique_users],
    ["Estudiantes únicos", overall.unique_students],
    ["Accuracy", Number(overall.accuracy ?? 0).toFixed(4)],
    ["Tiempo promedio (s)", Number(overall.avg_time_sec ?? 0).toFixed(2)],
    ["Pistas promedio", Number(overall.avg_hints ?? 0).toFixed(2)],
    ["Errores promedio", Number(overall.avg_incorrects ?? 0).toFixed(2)],
    ["Dependencia de pistas (correctos)", `${Math.round((overall.help_dependency ?? 0) * 100)}%`],
    ["Intentos promedio hasta resolver", Number(overall.avg_attempts_to_resolve ?? 0).toFixed(2)],
    ["Tasa de oscilación", `${Math.round((overall.oscillation_rate ?? 0) * 100)}%`],
  ].map(([k, v]) => `<tr><td>${k}</td><td class="text-end">${v}</td></tr>`).join("");
  setHtml("adminMetricsOverall", rows);

  const userRows = perUser.map((row) => `
    <tr>
      <td>${row.username ?? "-"}</td>
      <td>${row.attempts ?? 0}</td>
      <td>${Number(row.accuracy ?? 0).toFixed(4)}</td>
      <td>${Number(row.delta_accuracy_2half ?? 0).toFixed(4)}</td>
      <td>${Number(row.delta_time_2half ?? 0).toFixed(2)}</td>
      <td>${Math.round((row.oscillation_rate ?? 0) * 100)}%</td>
    </tr>
  `).join("");
  setHtml("adminMetricsUsers", userRows || "<tr><td colspan='6'>Sin datos.</td></tr>");
}

async function testModelAdmin() {
  const payload = {
    duration: Number(qs("admDuration").value || 60),
    incorrects: Number(qs("admIncorrects").value || 0),
    hints: Number(qs("admHints").value || 0),
    correct_first_attempt: Number(qs("admCfa").value || 1),
    step_name: qs("admStep").value || "2x+3=11",
  };
  const data = await api("/api/admin/test-model", payload);
  setText("adminTestMsg", data.message || "");
  setText("adminTestOut", JSON.stringify(data.prediction, null, 2));
}

async function retrainAdmin() {
  const data = await api("/api/admin/retrain", {});
  setText("adminTestMsg", data.message || "Reentrenado");
  await loadAdmin();
}

async function login() {
  const username = qs("loginUser").value.trim();
  const password = qs("loginPass").value;
  if (!username || !password) {
    setText("authMsg", "Completa usuario y contrasena.");
    return;
  }

  const data = await api("/api/auth/login", { username, password });
  if (!data.ok) {
    setText("authMsg", data.message || "No se pudo iniciar sesion.");
    return;
  }

  state.user = username.toLowerCase();
  state.role = data.role;
  state.game = data.state;
  state.question = data.state.current_question;

  show("authSection", false);
  show("playerSection", data.role === "player");
  show("adminSection", data.role === "admin");

  if (data.role === "player") {
    setText("playerUser", state.user);
    updateAll();
  } else {
    setText("adminUser", state.user);
    await loadAdmin();
  }
}

async function register() {
  const username = qs("regUser").value.trim();
  const password = qs("regPass").value;
  const role = qs("regRole").value;
  if (!username || !password) {
    setText("authMsg", "Completa usuario y contrasena para registrar.");
    return;
  }
  const data = await api("/api/auth/register", { username, password, role });
  setText("authMsg", data.message || "Registro procesado");
}

function bindEvents() {
  qs("btnLogin").addEventListener("click", login);
  qs("btnRegister").addEventListener("click", register);

  qs("btnSave").addEventListener("click", saveProgress);
  qs("btnLoad").addEventListener("click", loadProgress);
  qs("btnReset").addEventListener("click", resetProgress);
  qs("btnLogoutPlayer").addEventListener("click", logout);

  qs("btnHint").addEventListener("click", requestHint);
  qs("btnNext").addEventListener("click", nextQuestion);
  qs("btnSubmit").addEventListener("click", submitAnswer);
  qs("answerInput").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") submitAnswer();
  });

  qs("btnAdminRefresh").addEventListener("click", loadAdmin);
  qs("btnAdminTest").addEventListener("click", testModelAdmin);
  qs("btnAdminRetrain").addEventListener("click", retrainAdmin);
  qs("btnLogoutAdmin").addEventListener("click", logout);
  qs("figureModelFilter").addEventListener("change", renderAdminFigures);
}

function init() {
  state.modal = new bootstrap.Modal(qs("modelModal"));
  state.figureModal = new bootstrap.Modal(qs("figureModal"));
  bindEvents();
}

document.addEventListener("DOMContentLoaded", init);
