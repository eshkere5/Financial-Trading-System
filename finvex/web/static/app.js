"use strict";
const TOKEN = document.querySelector('meta[name=finvex-token]').content;
const $ = (s, r = document) => r.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const DEC = { allow: "Разрешено", revise: "Доработать", deny: "Отклонено" };
const SIDE = { buy: "Покупка", sell: "Продажа" };
const ST = { pass: "ПРОЙДЕНО", fail: "НАРУШЕНО", warn: "ВНИМАНИЕ", skip: "НЕ ПРИМЕНЯЛОСЬ" };
const RANGES = [["1d", "1Д"], ["1w", "1Н"], ["1m", "1М"], ["3m", "3М"], ["1y", "1Г"]];
const RANGE_NAME = { "1d": "день", "1w": "неделя", "1m": "месяц", "3m": "3 месяца", "1y": "год" };
const EFFECT = { deny: ["Отклоняет", "deny"], revise: ["На доработку", "revise"], warn: ["Предупреждает", "warn"], mixed: ["Доработка или отказ", "revise"], bands: ["Три зоны", "revise"] };
const GROUPS = [
  ["Допуск", ["kill_switch", "allowed_symbols", "blocked_symbols", "require_rationale", "max_rationale", "min_confidence"]],
  ["Размер и риск", ["max_order_value", "max_position_pct", "no_short", "regime_block_buy"]],
  ["Время и данные", ["trading_hours", "data_freshness", "price_deviation"]],
  ["Частота", ["max_orders_per_day", "cooldown", "max_revisions"]],
  ["Аналитика по событиям", ["news_sentiment", "kronos_view"]],
];
const CTX_LABELS = { price: "Цена", price_time: "Время последней свечи", data_age_min: "Возраст данных, мин.", regime: "Режим рынка", adx: "ADX", position_qty: "Позиция, шт.", equity: "Капитал портфеля, ₽" };
const S = { view: "verdicts", symbol: "BTCUSDT", range: "1w", decision: "", agent: "", sel: "", state: null, agents: [], seen: null, asset: null };
const fmt = (n, d = 2) => (n == null || Number.isNaN(n)) ? "-" : Number(n).toLocaleString("ru-RU", { maximumFractionDigits: d });

function add(el, ...kids) {
  for (const c of kids.flat(Infinity)) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
}
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  return add(el, kids);
}
const clear = el => { while (el.firstChild) el.removeChild(el.firstChild); return el; };

async function api(path, opt = {}) {
  const r = await fetch(path, { method: opt.method || "GET", headers: { "X-Token": TOKEN, "Content-Type": "application/json" },
    body: opt.body ? JSON.stringify(opt.body) : undefined });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((j.error && j.error.message) || ("Ошибка " + r.status));
  return j;
}
let toastT;
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toastT); toastT = setTimeout(() => (t.hidden = true), 3600);
}
const guard = fn => async (...a) => { try { return await fn(...a); } catch (e) { toast(e.message); } };
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

/* навигация */
const TITLES = { verdicts: "Вердикты", chat: "Чат", rules: "Правила", agents: "Агенты", settings: "Настройки", sandbox: "Песочница", modules: "Модули" };
const NAV_MODULE = { verdicts: "guard", rules: "guard", agents: "guard", chat: "hub", sandbox: "hub" };   // какой модуль открывает раздел
function setView(v) {
  if (!TITLES[v]) v = "verdicts";
  S.view = v; document.body.dataset.view = v; $("#title").textContent = TITLES[v];
  document.title = `${TITLES[v]} · Finvex`;
  document.querySelectorAll(".nav").forEach(b => b.dataset.view === v ? b.setAttribute("aria-current", "page") : b.removeAttribute("aria-current"));
  document.body.classList.remove("panel-open");
  if (location.hash !== "#" + v) history.replaceState(null, "", "#" + v);
  const need = NAV_MODULE[v];
  S.locked = !!need && !modOn(need); document.body.dataset.locked = S.locked ? "1" : "";
  if (S.locked) { renderLocked(need); return; }
  if (v === "verdicts") loadVerdicts();
  if (v === "rules") loadRules();
  if (v === "agents") loadAgents();
  if (v === "chat") initChat();
  if (v === "settings") { loadSettings(); loadAdapters(); }
  if (v === "sandbox") loadSandbox();
  if (v === "modules") loadModules();
}
document.querySelectorAll(".nav").forEach(b => b.addEventListener("click", () => { setView(b.dataset.view); $("#content").focus({ preventScroll: true }); }));
$("#btn-panel").addEventListener("click", () => { document.body.classList.add("panel-open"); $("#sym").focus(); });
$("#close-panel").addEventListener("click", () => { document.body.classList.remove("panel-open"); $("#btn-panel").focus(); });

/* состояние и сводка */
const loadState = guard(async () => {
  const s = await api("/api/state"); S.state = s;
  const chain = s.stats.ledger, meta = clear($("#top-meta"));
  const item = (k, v, bad) => h("div", {}, h("dt", {}, k), h("dd", { class: bad ? "bad" : "" }, v));
  add(meta, item("Данные", s.demo ? "демо, синтетика" : "живые"),
    item("Исполнение", s.execution.enabled ? "включено" : "не подключено"),
    item("Журнал", chain.ok ? `цел, записей ${chain.checked}` : "НАРУШЕН", !chain.ok),
    item("Модули", modSummary()),
    item("TSFM", !s.tsfm?.active ? "выключен" : ({not_loaded: "готов к загрузке", loading: "загрузка", ready: "модель загружена", unavailable: "недоступен"}[s.tsfm.state] || s.tsfm.state), s.tsfm?.active && s.tsfm.state === "unavailable"));
  renderSummary(s.stats);
});
function renderSummary(st) {
  const box = clear($("#summary")), t = Math.max(st.total, 1);
  const parts = [["allow", "Разрешено", "var(--data)"], ["revise", "На доработку", "var(--data2)"], ["deny", "Отклонено", "var(--tx)"]];
  const bar = h("div", { class: "bar", role: "img", "aria-label": parts.map(([k, l]) => `${l}: ${st[k]}`).join(", ") });
  parts.forEach(([k, , c]) => { const i = h("i"); i.style.width = (st[k] / t * 100) + "%"; i.style.background = c; bar.append(i); });
  add(box, h("div", { class: "sum-head" }, h("b", {}, fmt(st.total, 0)), h("span", {}, "намерений агентов в последней выборке")),
    bar, h("div", { class: "legend" }, parts.map(([k, l, c]) => { const m = h("i"); m.style.background = c; return h("span", {}, m, l, h("b", {}, fmt(st[k], 0))); })));
}

function renderStopped(r) {
  const box = clear($("#stopped"));
  box.hidden = !r.total;
  if (!r.total) return;
  const top = r.top_rules, max = Math.max(...top.map(x => x.count), 1);
  add(box,
    h("div", { class: "stop-head" },
      h("h2", {}, "Что остановил шлюз"),
      h("p", { class: "mu" }, r.deny + r.revise ? `${fmt(r.stopped_pct, 1)}% намерений не прошли как есть. Это ${fmt(r.held_notional, 0)} по ценам на момент решения.` : "Пока всё проходило без замечаний.")),
    top.length ? h("ul", { class: "stop-list" }, top.slice(0, 5).map(x => {
      const b = h("i"); b.style.width = (x.count / max * 100) + "%";
      return h("li", {}, h("span", { class: "stop-name" }, x.title), h("span", { class: "stop-bar", "aria-hidden": "true" }, b), h("b", { class: "num" }, fmt(x.count, 0)));
    })) : null);
}

/* настройки: ключ LLM */
const loadSettings = guard(async () => {
  const s = await api("/api/settings");
  $("#llm-model").value = s.model; $("#llm-url").value = s.base_url; $("#llm-key").value = "";
  const src = { env: "из переменной окружения DEEPSEEK_API_KEY", file: "из файла настроек", none: "" }[s.source];
  $("#llm-state").textContent = s.has_key ? `Ключ задан (${s.key_hint}), ${src}.` : "Ключ не задан: чат и анализ рынка отвечать не смогут.";
  $("#llm-clear").disabled = s.source !== "file";
});
function llmMsg(text, bad) { const m = $("#llm-msg"); m.textContent = text; m.classList.toggle("bad", !!bad); }
$("#llm-form").addEventListener("submit", async e => {
  e.preventDefault();
  try { await api("/api/settings", { method: "PUT", body: Object.fromEntries(new FormData(e.target)) }); llmMsg("Сохранено"); toast("Настройки сохранены"); await loadSettings(); await loadState(); }
  catch (err) { llmMsg(err.message, true); }
});
$("#llm-check").addEventListener("click", async () => {
  llmMsg("Проверяю…");
  try { const r = await api("/api/settings/check", { method: "POST", body: {} }); llmMsg(r.message, !r.ok); } catch (err) { llmMsg(err.message, true); }
});
$("#llm-clear").addEventListener("click", async () => {
  if (!confirm("Удалить сохранённый ключ? Чат и анализ перестанут отвечать.")) return;
  try { await api("/api/settings", { method: "PUT", body: { base_url: $("#llm-url").value, model: $("#llm-model").value, clear: true } }); llmMsg("Ключ удалён"); await loadSettings(); await loadState(); }
  catch (err) { llmMsg(err.message, true); }
});

/* источники данных */
const loadAdapters = guard(async () => {
  const a = await api("/api/adapters"), body = clear($("#ad-table tbody"));
  $("#ad-note").textContent = `${a.note} Исполнение: ${a.execution.enabled ? "включено" : "выключено"}.${a.sdk.pybit ? "" : " Библиотека pybit не установлена: она нужна только для торговли Bybit, публичные данные работают без неё."}`;
  a.market_data.forEach(r => body.append(h("tr", {}, h("td", {}, h("b", {}, r.name)), h("td", {}, r.type), h("td", { class: "mu" }, r.source),
    h("td", { class: "r" }, h("button", { class: "btn ghost sm", type: "button", onclick: async e => {
      e.target.disabled = true; e.target.textContent = "Проверяю…";
      try { const c = await api("/api/adapters/check", { method: "POST", body: { name: r.name } }); $("#ad-msg").textContent = `${r.name}: ${c.message}`; $("#ad-msg").classList.toggle("bad", !c.ok); }
      catch (err) { $("#ad-msg").textContent = err.message; $("#ad-msg").classList.add("bad"); }
      e.target.disabled = false; e.target.textContent = "Проверить"; } }, "Проверить"),
      r.removable ? h("button", { class: "btn ghost sm", type: "button", style: "margin-left:8px", onclick: guard(async () => { if (!confirm(`Удалить источник ${r.name}?`)) return; await api("/api/adapters", { method: "PUT", body: { remove: r.name } }); loadAdapters(); }) }, "Удалить") : null))));
  const sel = clear($("#ad-type")); a.types.forEach(t => sel.append(h("option", { value: t.type }, t.desc ? `${t.type}: ${t.desc}` : t.type)));
});
$("#ad-form").addEventListener("submit", async e => {
  e.preventDefault(); const f = new FormData(e.target); let params;
  try { params = JSON.parse(f.get("params") || "{}"); } catch (_) { $("#ad-msg").textContent = "Параметры должны быть корректным JSON"; $("#ad-msg").classList.add("bad"); return; }
  try { await api("/api/adapters", { method: "PUT", body: { name: f.get("name"), type: f.get("type"), params } }); $("#ad-msg").textContent = "Добавлено. Источник заработает после перезапуска."; $("#ad-msg").classList.remove("bad"); e.target.reset(); loadAdapters(); }
  catch (err) { $("#ad-msg").textContent = err.message; $("#ad-msg").classList.add("bad"); }
});

/* песочница */
let SBX = [];
const loadSandbox = guard(async () => {
  const r = await api("/api/sandbox/presets"); SBX = r.items;
  const sel = clear($("#sbx-preset")); r.items.forEach(p => sel.append(h("option", { value: p.id }, p.name)));
  const sync = () => { $("#sbx-prompt-wrap").hidden = sel.value !== "custom"; };
  sel.onchange = sync; sync();
  $("#sbx-msg").textContent = r.has_key ? "" : "Ключ LLM не задан: добавьте его во вкладке «Настройки».";
});
$("#sbx-form").addEventListener("submit", async e => {
  e.preventDefault(); const f = Object.fromEntries(new FormData(e.target)), out = clear($("#sbx-out")), btn = $("#sbx-run");
  btn.disabled = true; $("#sbx-msg").textContent = "Агент думает…"; $("#sbx-msg").classList.remove("bad");
  try {
    const r = await api("/api/sandbox/run", { method: "POST", body: f });
    $("#sbx-msg").textContent = `Готово за ${fmt(r.latency_ms / 1000, 1)} с`;
    const m = r.model, v = r.verdict;
    add(out, h("div", { class: "card form sbx-res" },
      h("h2", {}, `${r.preset} · ${r.symbol}`),
      m ? h("dl", { class: "kv" }, [["Решение модели", m.action], ["Количество", m.quantity ?? "-"], ["Тип", m.order_type || "market"], ["Уверенность", m.confidence ?? "-"]].flatMap(([k, x]) => [h("dt", {}, k), h("dd", {}, String(x))])) : null,
      m && m.rationale ? h("p", { class: "agent-msg" }, m.rationale) : null,
      r.note ? h("p", { class: "mu" }, r.note) : null,
      v ? [h("h3", { class: "sec" }, "Вердикт шлюза"), h("p", {}, decPill(v.decision), h("span", { class: "mu" }, `  ${v.execution_note}`)),
        h("p", { class: "agent-msg" }, v.agent_message),
        v.checks.filter(c => c.status !== "skip").map(c => h("div", { class: "chk " + c.status }, h("div", { class: "st" }, ST[c.status] || c.status), h("div", {}, h("div", { class: "t" }, c.name), h("div", { class: "m" }, c.message))))] : null,
      r.tokens && r.tokens.prompt ? h("p", { class: "hint" }, `Токены: запрос ${r.tokens.prompt}, ответ ${r.tokens.completion}`) : null));
  } catch (err) { $("#sbx-msg").textContent = err.message; $("#sbx-msg").classList.add("bad"); }
  btn.disabled = false;
});

/* вердикты */
function actionCells(i) {
  const lim = i.order_type === "limit" ? ` по ${fmt(i.limit_price, 4)}` : "";
  return h("div", { class: "act" }, h("b", {}, `${SIDE[i.side]} ${i.symbol}`), h("span", { class: "num" }, `${fmt(i.quantity, 6)} шт.${lim}`));
}
const actionText = i => `${SIDE[i.side]} ${i.symbol}, ${fmt(i.quantity, 6)} шт.${i.order_type === "limit" ? " по " + fmt(i.limit_price, 4) : ""}`;
const decPill = d => h("span", { class: "dec " + d }, DEC[d] || d);

function skeleton() {
  const body = clear($("#vt tbody"));
  for (let k = 0; k < 6; k++) body.append(h("tr", { class: "sk", "aria-hidden": "true" }, [1, 2, 3, 4, 5].map(() => h("td", {}, h("i")))));
}
const loadVerdicts = guard(async () => {
  if (S.seen === null) skeleton();
  const [v, ag, rep] = await Promise.all([
    api(`/api/verdicts?limit=200&decision=${S.decision}&agent=${S.agent}`), api("/api/agents"), api("/api/report"), loadState()]);
  S.items = v.items; renderStopped(rep);
  S.agents = ag.items; renderFilters();
  const body = clear($("#vt tbody")), first = S.seen === null, now = new Set();
  v.items.forEach(x => {
    now.add(x.id);
    const fresh = !first && S.seen && !S.seen.has(x.id);
    const tr = h("tr", { class: "clk" + (x.id === S.sel ? " sel" : "") + (fresh ? " fresh" : ""), tabindex: "0", role: "button", "aria-label": `${actionText(x.intent)}: ${DEC[x.decision]}. Открыть проверки.` },
      h("td", { class: "num mu" }, x.ts.slice(5, 16).replace("T", " ")),
      h("td", {}, x.agent_name, x.mode === "enforce" ? h("span", { class: "tag" }, "enforce") : null),
      h("td", {}, actionCells(x.intent)),
      h("td", {}, decPill(x.decision)),
      h("td", { class: "why c-why" }, x.reasons[0] || "Все проверки пройдены"));
    const open = () => { S.sel = x.id; openDrawer(x); setSymbol(x.intent.symbol); body.querySelectorAll("tr.sel").forEach(r => r.classList.remove("sel")); tr.classList.add("sel"); };
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    body.append(tr);
  });
  S.seen = now;
  $("#v-empty").hidden = v.items.length > 0;
  $(".table-wrap").hidden = v.items.length === 0;
});
function renderFilters() {
  const f = clear($("#filters"));
  [["", "Все"], ["allow", "Разрешено"], ["revise", "Доработать"], ["deny", "Отклонено"]].forEach(([k, l]) =>
    f.append(h("button", { class: "chip", type: "button", "aria-pressed": String(S.decision === k), onclick: () => { S.decision = k; loadVerdicts(); } }, l)));
  const sel = h("select", { "aria-label": "Агент", onchange: async e => { S.agent = e.target.value; loadVerdicts(); setSymbol(await symbolForAgent()); } },
    h("option", { value: "" }, "Все агенты"), S.agents.map(a => h("option", { value: a.id, selected: a.id === S.agent }, a.name)));
  f.append(sel, h("a", { class: "chip sp", href: `/api/export.csv?t=${TOKEN}`, download: "finvex_verdicts.csv" }, "Выгрузить CSV"));
}
const drawer = $("#drawer");
function openDrawer(v) {
  const i = v.intent, d = clear(drawer);
  add(d, h("div", { class: "dr" },
    h("button", { class: "btn ghost sm", type: "button", onclick: () => drawer.close() }, "Закрыть"),
    h("h2", { id: "dr-title" }, actionText(i)),
    h("p", { class: "sub" }, `${v.agent_name} · ${v.ts.replace("T", " ")} · режим ${v.mode === "shadow" ? "проверки" : "enforce"}`),
    h("p", {}, decPill(v.decision), v.suggested_quantity != null ? h("span", { class: "mu" }, `  Допустимый объём: ${fmt(v.suggested_quantity, 6)}`) : null),
    h("p", { class: "mu", style: "margin-top:8px" }, v.execution_note),
    v.agent_message ? [h("h3", { class: "sec" }, "Что получил агент"), h("p", { class: "agent-msg" }, v.agent_message)] : null,
    h("h3", { class: "sec" }, "Проверки"),
    v.checks.map(c => h("div", { class: "chk " + c.status }, h("div", { class: "st" }, ST[c.status] || c.status),
      h("div", {}, h("div", { class: "t" }, c.name), h("div", { class: "m" }, c.message)))),
    h("h3", { class: "sec" }, "Обоснование агента"),
    h("p", {}, i.rationale || "не указано"),
    i.confidence != null ? h("p", { class: "mu" }, `Уверенность агента: ${fmt(i.confidence)}`) : null,
    h("h3", { class: "sec" }, "Контекст в момент решения"),
    h("dl", { class: "kv" }, Object.entries(v.context || {}).filter(([k]) => CTX_LABELS[k]).flatMap(([k, val]) => [h("dt", {}, CTX_LABELS[k]), h("dd", {}, k === "price_time" ? String(val).slice(0, 16).replace("T", " ") : typeof val === "number" ? fmt(val, 4) : String(val ?? "-"))])),
    h("h3", { class: "sec" }, "Запись в журнале"),
    h("div", { class: "hash" }, `${v.id}\nмандат ${(v.mandate_hash || "").slice(0, 16) || "-"}\n${v.ledger_hash}`)));
  if (!drawer.open) drawer.showModal();
  drawer.scrollTop = 0;
}
drawer.addEventListener("click", e => { if (e.target === drawer) drawer.close(); });

/* панель актива */
RANGES.forEach(([k, l]) => $("#ranges").append(h("button", { type: "button", "data-r": k, "aria-label": RANGE_NAME[k], onclick: () => { S.range = k; loadAsset(); } }, l)));
const DEFAULT_SYMBOL = "BTCUSDT";
async function symbolForAgent() {
  if (!S.agent) return DEFAULT_SYMBOL;
  try {   // последнее решение агента, независимо от фильтра по решению
    const last = (await api(`/api/verdicts?limit=1&agent=${encodeURIComponent(S.agent)}`)).items[0];
    if (last) return last.intent.symbol;
    const rules = (await api("/api/rules")).rules.allowed_symbols;   // иначе первая разрешённая бумага
    if (rules && rules.enabled && rules.value.length) return rules.value[0];
  } catch (e) { /* ниже оставим текущий график */ }
  toast("У агента ещё нет решений: график не меняется");
  return null;
}
function setSymbol(s) { if (!s) return; S.symbol = s; $("#sym").value = s; loadAsset(); loadCtx(); }
$("#sym").addEventListener("keydown", e => { if (e.key === "Enter") setSymbol(e.target.value.trim().toUpperCase()); });
$("#sym").addEventListener("change", e => setSymbol(e.target.value.trim().toUpperCase()));
let ASSET_REQ = 0, CTX_REQ = 0;   // номер последнего запроса: запоздавший ответ по прежней бумаге игнорируется
const loadAsset = guard(async () => {
  const mine = ++ASSET_REQ;
  document.querySelectorAll("#ranges button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.r === S.range)));
  let a;
  try { a = await api(`/api/asset?symbol=${encodeURIComponent(S.symbol)}&range=${S.range}`); }
  catch (e) { if (mine !== ASSET_REQ) return; S.asset = null; clear($("#chart")).append(h("div", { class: "none" }, e.message)); $("#px").textContent = "-"; $("#px-c").textContent = ""; clear($("#kv")); return; }
  if (mine !== ASSET_REQ) return;
  S.asset = a;
  $("#px").textContent = fmt(a.last, 4);
  const ch = (a.last / a.first - 1) * 100;
  $("#px-c").textContent = `${ch >= 0 ? "+" : "−"}${fmt(Math.abs(ch))}% за ${RANGE_NAME[S.range]}`;
  $("#synthetic").hidden = !a.synthetic;
  const kv = clear($("#kv"));
  [["Максимум", fmt(a.high, 4)], ["Минимум", fmt(a.low, 4)], ["Объём за период", fmt(a.volume, 0)], ["Шаг свечи", a.timeframe]].forEach(([k, v]) => kv.append(h("dt", {}, k), h("dd", {}, v)));
  drawChart(a);
});
const loadCtx = guard(async () => {
  const mine = ++CTX_REQ;
  const kv = clear($("#ctx"));
  if (!modOn("guard")) return;
  try {
    const c = await api(`/api/context?symbol=${encodeURIComponent(S.symbol)}`);
    if (mine !== CTX_REQ) return;
    const p = (v, u = "") => v != null ? fmt(v) + u : null;
    [["Режим рынка", c.regime], ["Свежесть данных", c.data_age_min != null ? fmt(c.data_age_min, 0) + " мин." : null], ["Изменение за день", p(c.change_1d_pct, "%")],
      ["Волатильность 20д", p(c.volatility_20d_pct, "%")], ["RSI 14", c.rsi14 != null ? fmt(c.rsi14, 1) : null], ["ADX", c.adx != null ? fmt(c.adx, 1) : null],
      ["Просадка от макс. 60д", p(c.drawdown_60d_pct, "%")], ["Позиция, шт.", fmt(c.position_qty, 6)]].forEach(([k, v]) => kv.append(h("dt", {}, k), h("dd", {}, v ?? "-")));
  } catch (e) { kv.append(h("dt", {}, "Нет данных"), h("dd", {}, e.message)); }
});
function drawChart(a) {
  const box = clear($("#chart")), pts = a.points;
  if (pts.length < 2) { box.append(h("div", { class: "none" }, "Мало данных для графика")); return; }
  const W = box.clientWidth || 360, H = 200, pr = 50, pt = 10, pb = 22;
  const ys = pts.map(p => p[1]), lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.1 || hi * 0.01;
  const y0 = lo - pad, y1 = hi + pad, X = i => i / (pts.length - 1) * (W - pr), Y = v => pt + (1 - (v - y0) / (y1 - y0)) * (H - pt - pb);
  const el = (t, at) => { const e = document.createElementNS(NS, t); for (const k in at) e.setAttribute(k, at[k]); return e; };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${a.symbol}: последняя ${fmt(a.last, 4)}, максимум ${fmt(a.high, 4)}, минимум ${fmt(a.low, 4)} за ${RANGE_NAME[S.range]}` });
  const defs = el("defs", {}), g = el("linearGradient", { id: "gr", x1: 0, y1: 0, x2: 0, y2: 1 });
  g.append(el("stop", { offset: "0", "stop-color": "#2f80ff", "stop-opacity": ".26" }), el("stop", { offset: "1", "stop-color": "#2f80ff", "stop-opacity": "0" }));
  defs.append(g); svg.append(defs);
  const text = (x, y, s, anchor = "start") => { const t = el("text", { x, y, fill: "#9aa2ae", "font-size": 11, "text-anchor": anchor, "font-family": "Onest,system-ui,sans-serif" }); t.textContent = s; return t; };
  for (let k = 0; k < 4; k++) {
    const v = y0 + (y1 - y0) * (k + 0.5) / 4, yy = Y(v);
    svg.append(el("line", { x1: 0, x2: W - pr, y1: yy, y2: yy, stroke: "#1d2127" }), text(W - pr + 8, yy + 4, fmt(v, v < 100 ? 2 : 0)));
  }
  const d = pts.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
  svg.append(el("path", { d: `${d}L${X(pts.length - 1)},${H - pb}L0,${H - pb}Z`, fill: "url(#gr)" }),
    el("path", { class: "ln", pathLength: 1, d, fill: "none", stroke: "#2f80ff", "stroke-width": 1.8, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  const label = i => pts[i][0].slice(5, 10).split("-").reverse().join(".");
  svg.append(text(0, H - 5, label(0)), text(W - pr, H - 5, label(pts.length - 1), "end"));
  const cur = el("line", { y1: pt, y2: H - pb, stroke: "#6b7380", visibility: "hidden" }), dot = el("circle", { r: 4, fill: "#2f80ff", stroke: "#050607", "stroke-width": 2, visibility: "hidden" });
  svg.append(cur, dot); box.append(svg);
  const tip = h("div", { class: "tip", hidden: true }); box.append(tip);
  const move = e => {
    const r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * W;
    const i = Math.max(0, Math.min(pts.length - 1, Math.round(x / (W - pr) * (pts.length - 1))));
    cur.setAttribute("x1", X(i)); cur.setAttribute("x2", X(i)); cur.setAttribute("visibility", "visible");
    dot.setAttribute("cx", X(i)); dot.setAttribute("cy", Y(pts[i][1])); dot.setAttribute("visibility", "visible");
    tip.hidden = false; tip.textContent = `${pts[i][0].slice(5, 16).replace("T", " ")}  ${fmt(pts[i][1], 4)}`;
    tip.style.left = Math.min(Math.max(X(i) / W * r.width, 60), r.width - 60) + "px"; tip.style.top = (Y(pts[i][1]) / H * r.height) + "px";
  };
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerleave", () => { cur.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tip.hidden = true; });
}
window.addEventListener("resize", debounce(() => { if (S.asset) drawChart(S.asset); }, 150));

/* ручная проверка */
const form = $("#intent");
form.order_type.addEventListener("change", () => { form.limit_price.disabled = form.order_type.value !== "limit"; if (!form.limit_price.disabled) form.limit_price.focus(); });
form.addEventListener("submit", guard(async e => {
  e.preventDefault();
  const f = new FormData(form), num = k => (f.get(k) || "").toString().replace(",", ".");
  const v = await api("/api/intent", { method: "POST", body: { symbol: S.symbol, side: f.get("side"), quantity: num("quantity"), order_type: f.get("order_type"),
    limit_price: num("limit_price"), confidence: num("confidence"), rationale: f.get("rationale"), client_id: "ui-" + Date.now() } });
  const r = clear($("#result")); r.hidden = false;
  add(r, h("div", {}, decPill(v.decision), " ", h("b", {}, actionText(v.intent))),
    v.reasons.length ? h("ul", {}, v.reasons.map(x => h("li", {}, x))) : h("p", { class: "mu" }, "Все проверки пройдены."),
    v.suggested_quantity != null ? h("div", {}, `Допустимый объём: ${fmt(v.suggested_quantity, 6)}`) : null,
    h("div", {}, h("button", { class: "btn ghost sm", type: "button", onclick: () => openDrawer(v) }, "Все проверки")));
  loadState(); if (S.view === "verdicts") loadVerdicts();
}));

/* чат */
const SID = sessionStorage.sid || (sessionStorage.sid = Math.random().toString(36).slice(2, 12));
const CHATS = {};   // история по чатам: "" общий, иначе id агента
function chatKey() { return S.chatAgent || ""; }
function chatSid() { return S.chatAgent ? "agent-" + S.chatAgent : SID; }
function renderChat() {
  const box = clear($("#msgs")), hist = CHATS[chatKey()] || [];
  hist.forEach(m => paint(m));
  if (!hist.length) pushMsg("bot", S.chatAgent ? "Это чат агента: он знает его режим и последние решения шлюза. Спросите, почему заявку остановили, или попросите разобрать бумагу." : "Задайте вопрос по рынку, например: «полный анализ Сбера за неделю». Результат можно сверить с вердиктами агентов.");
  box.scrollTop = 1e9;
}
async function initChat() {
  if (!(S.state && S.state.chat)) {
    $("#chat-note").hidden = false; $("#chat-note").textContent = (S.state && S.state.chat_note) || "Чат отключён.";
    $("#q").disabled = true; $("#send").disabled = true;
  } else if (S.state && S.state.llm && !S.state.llm.has_key) {
    $("#chat-note").hidden = false; $("#chat-note").textContent = "Для аналитики нужен ключ LLM. Добавьте его во вкладке «Настройки», затем задайте вопрос.";
  }
  try { S.agents = (await api("/api/agents")).items; } catch (e) { /* список агентов не обязателен */ }
  const sel = $("#chat-agent"); sel.length = 1;
  (S.agents || []).filter(a => !a.revoked).forEach(a => sel.append(h("option", { value: a.id, selected: a.id === S.chatAgent }, a.name)));
  sel.onchange = () => { S.chatAgent = sel.value; renderChat(); };
  renderChat();
}
function inline(text) {
  const out = []; let last = 0;
  for (const m of text.matchAll(/\*\*([^*]+)\*\*|`([^`]+)`/g)) {
    out.push(text.slice(last, m.index)); out.push(m[1] ? h("strong", {}, m[1]) : h("code", {}, m[2])); last = m.index + m[0].length;
  }
  out.push(text.slice(last)); return out;
}
function markdown(src) {
  const box = h("div"), lines = src.split("\n"); let i = 0;
  const li = /^\s*[-*]\s+/, ol = /^\s*\d+\.\s+/;
  while (i < lines.length) {
    const l = lines[i];
    if (!l.trim()) { i++; continue; }
    const hd = l.match(/^(#{1,4})\s+(.*)/);
    if (hd) { box.append(h(hd[1].length <= 2 ? "h3" : "h4", {}, inline(hd[2]))); i++; continue; }
    if (li.test(l)) { const ul = h("ul"); while (i < lines.length && li.test(lines[i])) ul.append(h("li", {}, inline(lines[i++].replace(li, "")))); box.append(ul); continue; }
    if (ol.test(l)) { const o = h("ol"); while (i < lines.length && ol.test(lines[i])) o.append(h("li", {}, inline(lines[i++].replace(ol, "")))); box.append(o); continue; }
    if (l.includes("|") && lines[i + 1] && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      const row = s => s.trim().replace(/^\||\|$/g, "").split("|").map(x => x.trim());
      const t = h("table"), head = h("tr"); row(l).forEach(c => head.append(h("th", {}, inline(c)))); t.append(head); i += 2;
      while (i < lines.length && lines[i].includes("|")) { const tr = h("tr"); row(lines[i++]).forEach(c => tr.append(h("td", {}, inline(c)))); t.append(tr); }
      box.append(t); continue;
    }
    const p = []; while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*[-*]\s|\s*\d+\.\s)/.test(lines[i])) p.push(lines[i++]);
    box.append(h("p", {}, inline(p.join(" "))));
  }
  return box;
}
function paint(m) {
  const el = m.role === "user" ? h("div", { class: "msg user" }, m.text) : h("div", { class: "msg bot" + (m.pending ? " pending" : "") }, m.pending ? m.text : markdown(m.text));
  (m.images || []).forEach(n => el.append(h("img", { alt: "График из ответа", src: `/api/chart?sid=${m.sid}&name=${encodeURIComponent(n)}&t=${TOKEN}` })));
  $("#msgs").append(el); $("#msgs").scrollTop = 1e9; return el;
}
function pushMsg(role, text, images = []) {
  const m = { role, text, images, sid: chatSid() };
  (CHATS[chatKey()] = CHATS[chatKey()] || []).push(m);
  return paint(m);
}
$("#composer").addEventListener("submit", async e => {
  e.preventDefault();
  const q = $("#q"), text = q.value.trim(); if (!text) return;
  q.value = ""; const key = chatKey(), sid = chatSid(); pushMsg("user", text);
  const pend = paint({ role: "bot", text: "Считаю…", pending: true }); $("#send").disabled = true;
  try {
    const r = await api("/api/chat", { method: "POST", body: { sid, text, agent: S.chatAgent || "" } }); pend.remove();
    (CHATS[key] = CHATS[key] || []).push({ role: "bot", text: r.answer, images: r.images || [], sid }); if (key === chatKey()) renderChat();
  } catch (err) {
    pend.remove(); (CHATS[key] = CHATS[key] || []).push({ role: "bot", text: "Не получилось: " + err.message + ". Повторите запрос или уточните тикер и период.", images: [], sid }); if (key === chatKey()) renderChat();
  }
  $("#send").disabled = false; q.focus();
});
$("#q").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });

/* правила */
let RULE_STATE = {};
const loadRules = guard(async () => {
  const r = await api("/api/rules"), meta = Object.fromEntries(r.meta.map(m => [m.id, m]));
  RULE_STATE = JSON.parse(JSON.stringify(r.rules));
  const box = clear($("#rules"));
  GROUPS.forEach(([title, ids]) => {
    const g = h("section", { class: "group", "aria-labelledby": "g-" + ids[0] }, h("h2", { id: "g-" + ids[0] }, title));
    ids.filter(id => meta[id]).forEach(id => {
      const m = meta[id], st = RULE_STATE[id], row = h("div", { class: "rule" + (st.enabled ? "" : " off") });
      const sw = h("label", { class: "sw" }, h("input", { type: "checkbox", checked: st.enabled, "aria-label": `Включить: ${m.title}`,
        onchange: e => { st.enabled = e.target.checked; row.classList.toggle("off", !st.enabled); } }), h("i"));
      const val = h("div", { class: "val" });
      if (m.kind === "number") add(val, h("input", { inputmode: "decimal", value: st.value, "aria-label": `${m.title}, ${m.unit}`, oninput: e => (st.value = e.target.value.replace(",", ".")) }), h("span", { class: "unit" }, m.unit));
      else if (m.kind === "list") add(val, h("input", { type: "text", value: (st.value || []).join(", "), placeholder: "через запятую", "aria-label": m.title, oninput: e => (st.value = e.target.value) }));
      else if (m.kind === "band") {
        const lbl = m.below ? ["Доработка при ≤", "Отказ при ≤"] : ["Доработка при ≥", "Отказ при ≥"];
        const num = (key, label) => h("label", { class: "band-f" }, label, h("input", { inputmode: "decimal", value: st.value[key], "aria-label": `${m.title}: ${label}`,
          oninput: e => (st.value[key] = e.target.value.replace(",", ".")) }));
        const mode = h("select", { "aria-label": `${m.title}: режим`, onchange: e => (st.value.enforce = e.target.value === "1") },
          h("option", { value: "0", selected: !st.value.enforce }, "Только предупреждать"), h("option", { value: "1", selected: !!st.value.enforce }, "Применять к решению"));
        add(val, num("revise", lbl[0]), num("deny", lbl[1]), mode, h("span", { class: "unit" }, m.unit));
        val.classList.add("band");
      } else add(val, h("span", { class: "unit" }, "включено или выключено"));
      const eff = EFFECT[m.effect] || ["", ""];
      const title = h("h3", {}, m.title, " ", h("span", { class: "eff " + eff[1] }, eff[0]));
      if (m.options) {
        const tip = h("div", { class: "tip-pop", role: "tooltip", id: "tip-" + m.id }, h("b", {}, "Режимы рынка"),
          h("dl", {}, m.options.flatMap(o => [h("dt", {}, `${o.id} · ${o.title}`), h("dd", {}, o.desc)])));
        title.append(h("span", { class: "tip-wrap" }, h("button", { class: "help", type: "button", "aria-label": "Что означают режимы", "aria-describedby": "tip-" + m.id }, "?"), tip));
      }
      add(row, sw, h("div", {}, title, h("p", {}, m.desc)), val); g.append(row);
    });
    if (title === "Аналитика по событиям" && !modOn("analytics")) {
      g.classList.add("locked-group"); g.setAttribute("inert", ""); g.append(lockNote("analytics"));
    }
    box.append(g);
  });
});
$("#save-rules").addEventListener("click", guard(async () => {
  const r = await api("/api/rules", { method: "PUT", body: { rules: RULE_STATE } });
  RULE_STATE = r.rules; $("#rules-msg").textContent = "Сохранено. Новые правила действуют для следующих намерений.";
  setTimeout(() => ($("#rules-msg").textContent = ""), 4500); loadRules();
}));

/* агенты */
const MODE_NOTE = { shadow: "проверка", enforce: "проверка и передача брокеру" };
const loadAgents = guard(async () => {
  const r = await api("/api/agents"), body = clear($("#at tbody"));
  r.items.forEach(a => {
    const mode = h("select", { "aria-label": `Режим агента ${a.name}`, disabled: a.revoked, onchange: guard(async e => {
      if (e.target.value === "enforce" && !confirm("Режим enforce передаёт разрешённые заявки брокеру, если исполнение подключено. Продолжить?")) { e.target.value = "shadow"; return; }
      await api("/api/agents", { method: "PATCH", body: { id: a.id, mode: e.target.value } }); toast("Режим изменён"); loadAgents();
    }) }, ["shadow", "enforce"].map(m => h("option", { value: m, selected: a.mode === m }, MODE_NOTE[m])));
    const prof = h("select", { "aria-label": `Контекст агента ${a.name}`, disabled: a.revoked, onchange: guard(async e => { await api("/api/agents", { method: "PATCH", body: { id: a.id, context_profile: e.target.value } }); toast("Профиль контекста изменён"); }) },
      [["minimal", "минимальный"], ["standard", "стандартный"], ["full", "полный"]].map(([k, l]) => h("option", { value: k, selected: a.context_profile === k }, l)));
    body.append(h("tr", {}, h("td", {}, h("div", {}, h("b", {}, a.name)), h("div", { class: "mu num", style: "font-size:12px" }, a.id)), h("td", {}, mode), h("td", {}, prof), h("td", { class: "num r" }, a.verdicts),
      h("td", {}, a.revoked ? "Отозван" : "Активен"),
      h("td", { class: "r" }, a.revoked ? null : h("button", { class: "btn ghost sm", type: "button", onclick: guard(async () => { if (confirm(`Отозвать ключ агента ${a.name}? Это действие нельзя отменить.`)) { await api("/api/agents", { method: "PATCH", body: { id: a.id, revoked: true } }); loadAgents(); } }) }, "Отозвать"))));
  });
  $("#a-empty").hidden = r.items.length > 0; $("#at").parentElement.hidden = r.items.length === 0;
});
$("#add-form").addEventListener("submit", guard(async e => {
  e.preventDefault();
  const f = new FormData(e.target);
  const r = await api("/api/agents", { method: "POST", body: { name: f.get("name"), mode: f.get("mode"), context_profile: f.get("context_profile") } });
  e.target.reset(); $("#add-box").open = false; showKey(r); loadAgents();
}));
const kd = $("#keydlg");
function showKey(r) {
  const url = location.origin;
  const mcp = JSON.stringify({ mcpServers: { finvex: { command: "python", args: ["-m", "finvex.gateway.mcp", "--url", url, "--key", r.key] } } }, null, 2);
  const curl = `curl -X POST ${url}/gateway/v1/propose \\\n  -H "Authorization: Bearer ${r.key}" -H "Content-Type: application/json" \\\n  -d '{"symbol":"SBER","side":"buy","quantity":10,"rationale":"причина решения"}'`;
  const copy = t => guard(async () => { await navigator.clipboard.writeText(t); toast("Скопировано"); });
  add(clear(kd), h("h2", { id: "kd-title" }, `Агент «${r.agent.name}» создан`),
    h("p", { class: "mu" }, "Ключ показывается один раз. В хранилище остаётся только его хэш, восстановить ключ нельзя."),
    h("div", { class: "code" }, r.key), h("button", { class: "btn ghost sm", type: "button", onclick: copy(r.key) }, "Скопировать ключ"),
    h("h3", { class: "sec" }, "Подключение через MCP"), h("div", { class: "code" }, mcp), h("button", { class: "btn ghost sm", type: "button", onclick: copy(mcp) }, "Скопировать конфигурацию"),
    h("h3", { class: "sec" }, "Подключение по HTTP"), h("div", { class: "code" }, curl),
    h("div", { class: "btns" }, h("button", { class: "btn solid", type: "button", onclick: () => kd.close() }, "Ключ сохранён")));
  kd.showModal();
}

/* модули: доступ к разделам и элементам зависит от состояния модулей (сервер проверяет то же самое независимо) */
S.modules = {}; S.modulesLoaded = false; S.locked = false;
const modOn = id => !id || !S.modulesLoaded || !!(S.modules[id] && S.modules[id].enabled);
const modName = id => (S.modules[id] || {}).title || id;
function modSummary() { const l = Object.values(S.modules); return l.length ? `${l.filter(m => m.enabled).length} из ${l.length}` : "-"; }
async function loadModulesState() {
  try { const r = await api("/api/modules"); S.modules = Object.fromEntries(r.items.map(m => [m.id, m])); S.modulesLoaded = true; }
  catch (e) { toast("Состояние модулей недоступно: " + e.message); }
  applyModules();
}
const toggleModule = guard(async (id, enabled) => {
  await api("/api/modules", { method: "PATCH", body: { id, enabled } });
  await loadModulesState(); await loadState();
  $("#mod-msg") && ($("#mod-msg").textContent = `${modName(id)}: ${enabled ? "включён" : "выключен"}.`);
  setView(S.view);
});
function lockNote(need, big) {
  const m = S.modules[need] || {}, miss = (m.requires || []).filter(r => !modOn(r));
  const box = h("div", { class: "lock-note" + (big ? " big" : ""), role: "note" },
    h("b", {}, `Модуль «${modName(need)}» выключен`),
    h("p", {}, big ? `Раздел недоступен. ${m.summary || ""}` : "Этот блок заработает после включения модуля."));
  const row = h("div", { class: "lock-actions" });
  if (m.can_enable) row.append(h("button", { class: "btn solid", type: "button", onclick: () => toggleModule(need, true) }, "Включить модуль"));
  else if (miss.length) row.append(h("span", { class: "mu" }, "Сначала включите: " + miss.map(modName).join(", ")));
  row.append(h("button", { class: "btn ghost", type: "button", onclick: () => setView("modules") }, "Открыть «Модули»"));
  return add(box, row);
}
function renderLocked(need) {
  const box = clear($("#v-locked"));
  add(box, h("h2", {}, TITLES[S.view]), lockNote(need, true));
}
function applyModules() {
  document.querySelectorAll(".nav[data-view]").forEach(b => {
    const need = NAV_MODULE[b.dataset.view], off = !!need && !modOn(need), label = b.querySelector("span").textContent;
    b.classList.toggle("locked", off);
    b.title = off ? `Требуется модуль «${modName(need)}»` : "";
    off ? b.setAttribute("aria-label", `${label}: требуется модуль «${modName(need)}»`) : b.removeAttribute("aria-label");
  });
  document.querySelectorAll("[data-requires]").forEach(el => {
    const need = el.dataset.requires, off = !modOn(need), sib = el.nextElementSibling;
    let note = sib && sib.classList.contains("lock-note") && sib.dataset.for === need ? sib : null;
    el.hidden = off; el.toggleAttribute("inert", off);
    if (note && (!off || note.dataset.stale)) { note.remove(); note = null; }
    if (off) { if (note) note.remove(); note = lockNote(need); note.dataset.for = need; el.after(note); }
  });
}
const loadModules = guard(async () => {
  await loadModulesState();
  const grid = clear($("#mod-grid"));
  Object.values(S.modules).forEach(m => {
    const toggle = h("button", { class: "btn " + (m.enabled ? "ghost" : "solid"), type: "button",
      disabled: m.enabled ? !m.can_disable : !m.can_enable, "aria-label": `${m.enabled ? "Выключить" : "Включить"}: ${m.title}`,
      onclick: () => { if (m.enabled && m.id === "guard" && !confirm("Без защитного слоя заявки агентов будут отклоняться. Выключить?")) return; toggleModule(m.id, !m.enabled); } },
      m.enabled ? "Выключить" : "Включить");
    const why = m.enabled ? (m.can_disable ? "" : "Нельзя выключить: от модуля зависят другие включённые.") : (m.can_enable ? "" : "Сначала включите: " + m.requires.map(modName).join(", "));
    add(grid, h("article", { class: "card form mod" + (m.enabled ? "" : " off") },
      h("div", { class: "mod-head" }, h("h3", {}, m.title), h("span", { class: "dec " + (m.enabled ? "allow" : "hold") }, m.enabled ? "Включён" : "Выключен")),
      h("p", {}, m.summary),
      h("p", { class: "mu" }, (m.enabled ? "Открыто в интерфейсе: " : "Будет открыто: ") + (m.ui || []).join(", ")),
      m.requires.length ? h("p", { class: "mu" }, "Требует: " + m.requires.map(modName).join(", ")) : null,
      h("details", {}, h("summary", {}, "Программный интерфейс"), h("div", { class: "code" }, m.api.join("\n"))),
      why ? h("p", { class: "hint" }, why) : null, toggle));
  });
});

window.addEventListener("hashchange", () => { const v = location.hash.slice(1); if (v && v !== S.view) setView(v); });

/* старт */
(async () => {
  await loadState(); await loadModulesState(); setSymbol(S.symbol); setView((location.hash || "#verdicts").slice(1));
  setInterval(() => { if (S.view === "verdicts" && !S.locked && document.visibilityState === "visible" && !drawer.open) loadVerdicts(); }, 15000);
})();
