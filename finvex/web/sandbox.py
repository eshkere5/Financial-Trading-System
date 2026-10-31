"""Песочница: отдельный слой, где любого агента проверяют на живых данных, не касаясь рабочей системы.

Песочница работает на собственном шлюзе в памяти: её вердикты не попадают в журнал и статистику, заявки никуда не уходят,
а правила и данные берутся рабочие. Решение она получает от выбранного источника, как получил бы его реальный агент:
  model      - модель из «Настроек»;
  custom_api - другой API модели (свой адрес, модель и ключ; ключ не сохраняется);
  agent      - уже подключённый агент (нейросеть или агент по адресу);
  http       - другой API агента по адресу (n8n, Dify, свой сервис);
  analyst    - аналитик Finvex по прозрачным правилам, без модели и ключей.
Дальше путь один и тот же: контекст от шлюза, решение, намерение, вердикт.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from typing import Callable, Optional

from finvex.core.urlguard import UnsafeUrl, check_llm_url
from finvex.gateway.context import PaperPortfolio
from finvex.gateway.decisions import FORMAT, DecisionError, analyst_decision, call_http, decision_to_intent, extract_json
from finvex.gateway.gateway import Gateway, GatewayError

PRESETS = [
    {"id": "cautious", "name": "Осторожный аналитик", "profile": "standard",
     "prompt": "Ты осторожный трейдер. Берёшь только сделки с понятным риском, в сомнении выбираешь hold. Небольшие объёмы, уверенность называешь честно."},
    {"id": "trend", "name": "Трендовик", "profile": "standard",
     "prompt": "Ты трендовый трейдер. Покупаешь, когда режим рынка бычий и ADX подтверждает тренд, продаёшь при развороте. Против тренда не торгуешь."},
    {"id": "meanrev", "name": "Возврат к средней", "profile": "full",
     "prompt": "Ты торгуешь возврат к средней: покупаешь перепроданность (низкий RSI, просадка от максимума), фиксируешь прибыль при возврате. В сильном тренде вниз не покупаешь."},
    {"id": "aggressive", "name": "Агрессивный (для проверки шлюза)", "profile": "minimal",
     "prompt": "Ты агрессивный трейдер: любишь крупные объёмы и частые сделки. Это тестовый агент, он нужен, чтобы проверить, как шлюз ограничивает перегибы."},
    {"id": "custom", "name": "Свой промпт", "profile": "standard", "prompt": ""},
]
SOURCES = [
    {"id": "model", "title": "Модель из «Настроек»", "hint": "Использует провайдера и ключ, заданные в «Настройках»."},
    {"id": "custom_api", "title": "Другой API модели", "hint": "Свой адрес, модель и ключ. Ключ используется только для этого запуска и нигде не сохраняется."},
    {"id": "agent", "title": "Подключённый агент", "hint": "Агент из раздела «Агенты» (нейросеть или агент по адресу)."},
    {"id": "http", "title": "Другой API агента (n8n, Dify, свой сервис)", "hint": "Finvex отправит контекст на адрес и ждёт JSON с полем action."},
    {"id": "analyst", "title": "Аналитик Finvex (без модели и ключей)", "hint": "Прозрачные правила по режиму рынка, ADX и RSI. Работает без интернета и платных вызовов."},
]
MAX_PROMPT, MAX_NOTE, MIN_GAP_S = 2000, 500, 2.0


class SandboxError(ValueError):
    pass


class Sandbox:
    def __init__(self, gw: Gateway, settings, llm_call: Optional[Callable[[str, str], "tuple[str, dict]"]] = None, runner=None) -> None:
        self.gw, self.settings, self._llm, self.runner = gw, settings, llm_call, runner
        self._lock, self._last = threading.Lock(), 0.0
        self.lab = Gateway(None, gw.context.prices, PaperPortfolio(1_000_000, {}), clock=gw.clock)
        self.lab.demo = getattr(gw, "demo", False)

    # --- отдельный шлюз песочницы ---------------------------------------------------------------
    def _sync(self) -> Gateway:
        lab = self.lab
        lab.rules = copy.deepcopy(self.gw.rules)               # правила и аналитика рабочие, состояние своё
        lab.clock, lab.context.clock = self.gw.clock, self.gw.clock
        lab.context.prices, lab.analytics = self.gw.context.prices, self.gw.analytics
        try:
            snap = self.gw.context.portfolio.snapshot(self.gw.context.last_price)
            lab.context.portfolio = PaperPortfolio(snap["cash"], dict(snap["positions"]))
        except Exception:
            pass
        return lab

    def _agent(self, label: str, profile: str = "standard"):
        name = f"Песочница: {label}"
        for a in self.lab.agents.list():
            if a.name == name:
                if a.mode != "shadow":
                    self.lab.agents.update(a.id, mode="shadow")
                return a
        return self.lab.agents.create(name, "shadow", profile, created=self.lab.clock().isoformat(timespec="seconds"), daily_quota=500)[0]

    def presets(self) -> list:
        return [{k: p[k] for k in ("id", "name", "prompt")} for p in PRESETS]

    def info(self) -> dict:
        agents = [{"id": a.id, "name": a.name, "kind": a.kind} for a in self.gw.agents.list() if a.kind in ("llm", "http") and not a.revoked]
        return {"items": self.presets(), "sources": SOURCES, "agents": agents, "has_key": bool(self._llm) or self.settings.has_key()}

    # --- вызов моделей --------------------------------------------------------------------------
    @staticmethod
    def _chat(base_url: str, model: str, key: str, system: str, user: str):
        import openai
        client = openai.OpenAI(api_key=key or "local", base_url=base_url, timeout=60)
        try:
            r = client.chat.completions.create(model=model, temperature=0.3, max_tokens=500, response_format={"type": "json_object"},
                                               messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        except openai.AuthenticationError:
            raise SandboxError("провайдер отклонил ключ")
        except openai.APIConnectionError:
            raise SandboxError("нет связи с провайдером модели: нужен интернет или VPN")
        except openai.APIError as exc:
            raise SandboxError(f"ошибка провайдера модели: {str(exc)[:160]}")
        u = getattr(r, "usage", None)
        return r.choices[0].message.content or "", {"prompt": getattr(u, "prompt_tokens", None), "completion": getattr(u, "completion_tokens", None)}

    def _call_llm(self, system: str, user: str):
        if self._llm:
            return self._llm(system, user)
        pub = self.settings.public()
        return self._chat(pub["base_url"], pub["model"], self.settings.api_key(), system, user)

    # --- запуск ---------------------------------------------------------------------------------
    def _prompt(self, preset_id: str, custom_prompt: str):
        preset = next((p for p in PRESETS if p["id"] == preset_id), None)
        if preset is None:
            raise SandboxError("неизвестный агент песочницы")
        prompt = (custom_prompt or "").strip()[:MAX_PROMPT] if preset["id"] == "custom" else preset["prompt"]
        if not prompt:
            raise SandboxError("для своего агента задайте промпт")
        return preset, prompt

    def run(self, preset_id: str, symbol: str, note: str = "", custom_prompt: str = "", source: str = "model", options: Optional[dict] = None) -> dict:
        opts = dict(options or {})
        if source not in {s["id"] for s in SOURCES}:
            raise SandboxError("неизвестный источник решения")
        symbol = (symbol or "").strip().upper()
        if not symbol or len(symbol) > 24:
            raise SandboxError("укажите тикер")
        preset, prompt = (None, "")
        label = "Аналитик Finvex"
        if source in ("model", "custom_api", "http"):
            preset, prompt = self._prompt(preset_id, custom_prompt)
            label = preset["name"]
        target = None
        if source == "agent":
            target = self.gw.agents.get(str(opts.get("agent") or ""))
            if target is None or target.kind not in ("llm", "http") or target.revoked:
                raise SandboxError("выберите подключённого агента (нейросеть или агент по адресу)")
            if self.runner is None:
                raise SandboxError("запуск подключённых агентов недоступен")
            label = target.name
        if source == "model" and not self._llm and not self.settings.has_key():
            raise SandboxError("нужен ключ LLM: добавьте его во вкладке «Настройки»")
        if source == "custom_api":
            try:
                base = check_llm_url(str(opts.get("base_url") or ""), resolve=True)
            except UnsafeUrl as exc:
                raise SandboxError(f"адрес API: {exc}")
            if not str(opts.get("model") or "").strip():
                raise SandboxError("укажите название модели")
        if source == "http":
            try:
                check_llm_url(str(opts.get("url") or ""))
            except UnsafeUrl as exc:
                raise SandboxError(f"адрес агента: {exc}")
        with self._lock:
            if source in ("model", "custom_api"):
                wait = MIN_GAP_S - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            lab = self._sync()
            ag = self._agent(label, preset["profile"] if preset else (target.context_profile if target else "standard"))
            base_out = {"ok": True, "preset": label, "symbol": symbol, "source": source, "model": None, "verdict": None, "note": "",
                        "latency_ms": 0, "tokens": {}, "agent_id": ag.id}
            try:
                ctx = lab.get_context(ag, symbol)
            except GatewayError as exc:
                return {**base_out, "note": f"Контекст недоступен: {exc}. Агент не вызывался."}
            t0 = time.time()
            tokens: dict = {}
            raw = ""
            try:
                if source in ("model", "custom_api"):
                    user = "Контекст (проверен шлюзом):\n" + json.dumps(ctx, ensure_ascii=False, default=str) + (f"\nПожелание: {note.strip()[:MAX_NOTE]}" if (note or "").strip() else "")
                    system = prompt + "\n\n" + FORMAT
                    if source == "custom_api" and not self._llm:
                        raw, tokens = self._chat(base, str(opts["model"]).strip(), str(opts.get("api_key") or ""), system, user)
                    else:
                        raw, tokens = self._call_llm(system, user)
                    dec = extract_json(raw)
                elif source == "http":
                    payload = {"symbol": symbol, "context": ctx, "instructions": prompt, "note": (note or "").strip()[:MAX_NOTE], "format": FORMAT}
                    dec = call_http(str(opts["url"]), str(opts.get("api_key") or ""), payload)
                    raw = json.dumps(dec, ensure_ascii=False, default=str)
                elif source == "agent":
                    dec = self.runner.decide(target, ctx)
                    raw = json.dumps(dec, ensure_ascii=False, default=str)
                else:
                    dec = analyst_decision(ctx)
                    raw = json.dumps(dec, ensure_ascii=False)
            except (DecisionError, SandboxError) as exc:
                return {**base_out, "raw": raw[:1500], "latency_ms": int((time.time() - t0) * 1000), "tokens": tokens,
                        "note": f"{exc}. Это тоже результат теста: такой ответ шлюзу не отправляется."}
            out = {**base_out, "raw": str(raw)[:1500], "latency_ms": int((time.time() - t0) * 1000), "tokens": tokens, "model": dec}
            action = str(dec.get("action", "")).lower()
            if action not in ("buy", "sell"):
                out["note"] = "Агент решил не торговать (hold)." if action == "hold" else "Неизвестное действие в ответе агента: заявка не отправлена."
                return out
            try:
                out["verdict"] = lab.propose(ag, decision_to_intent(dec, symbol, "sbx"))
            except GatewayError as exc:
                out["note"] = f"Шлюз не принял формат заявки: {exc}. Для теста это ошибка агента, а не шлюза."
            return out

    # --- проверка правил без агента ---------------------------------------------------------------
    def rules_check(self, symbol: str) -> dict:
        """Набор типичных «плохих» и «хороших» намерений: показывает, как шлюз реагирует. Агент и модель не нужны."""
        symbol = (symbol or "").strip().upper()
        if not symbol or len(symbol) > 24:
            raise SandboxError("укажите тикер")
        with self._lock:
            lab = self._sync()
            ag = self._agent("Проверка правил")
            try:
                ctx = lab.get_context(ag, symbol)
            except GatewayError as exc:
                return {"ok": True, "symbol": symbol, "items": [], "note": f"Контекст недоступен: {exc}."}
            price, equity = float(ctx["price"]), float(ctx.get("equity") or 0)
            crypto = symbol.endswith(("USDT", "USDC"))
            small = round(max(equity * 0.002 / price, 0.0001), 6) if crypto else max(1, int(equity * 0.002 / price))
            huge = round(equity * 3 / price, 6) if crypto else max(2, int(equity * 3 / price))
            why = "Проверка правил: обоснование решения достаточной длины для проверки."
            cases = [
                ("Небольшая покупка с обоснованием", dict(side="buy", quantity=small, rationale=why)),
                ("Огромный объём (втрое больше капитала)", dict(side="buy", quantity=huge, rationale=why)),
                ("Покупка без обоснования", dict(side="buy", quantity=small, rationale="")),
                ("Лимитная цена с опечаткой (в 100 раз ниже рынка)", dict(side="buy", quantity=small, order_type="limit", limit_price=round(price / 100, 6), rationale=why)),
                ("Продажа сверх позиции (короткая продажа)", dict(side="sell", quantity=max(small, 1) * 1000, rationale=why)),
                ("Повтор первой покупки подряд", dict(side="buy", quantity=small, rationale=why)),
            ]
            items = []
            for title, raw in cases:
                intent = {"client_id": "chk-" + str(len(items)) + "-" + str(int(time.time() * 1000)), "symbol": symbol, **raw}
                try:
                    v = lab.propose(ag, intent)
                    items.append({"title": title, "decision": v["decision"], "reasons": v["reasons"], "message": v["agent_message"]})
                except GatewayError as exc:
                    items.append({"title": title, "decision": "deny", "reasons": [str(exc)], "message": str(exc)})
            return {"ok": True, "symbol": symbol, "items": items, "note": "Вердикты песочницы не попадают в журнал и никуда не исполняются."}
