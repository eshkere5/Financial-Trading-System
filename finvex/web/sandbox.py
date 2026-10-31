"""Песочница: подключает ваш ключ LLM к разным тестовым агентам и пропускает их решения через шлюз.

Агенты песочницы всегда в режиме проверки (shadow). Модель получает проверенный контекст и лимиты, отвечает JSON-решением,
оно уходит в шлюз как обычное намерение, и возвращается вердикт.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from typing import Callable, Optional

from finvex.gateway.gateway import Gateway, GatewayError

FORMAT = ("Ответь ТОЛЬКО одним JSON-объектом без пояснений вне него: "
          '{"action":"buy|sell|hold","quantity":число,"order_type":"market|limit","limit_price":число или null,'
          '"confidence":число от 0 до 1,"rationale":"2-3 предложения: почему именно это решение"}. '
          "Если нет явного основания, выбери hold. Количество указывай в штуках (для акций) или в монетах (для крипто). "
          "Не придумывай тикеры и данные: используй только то, что дано в контексте. Шлюз проверит лимиты, но не превышай их сам.")

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
MAX_PROMPT, MAX_NOTE, MIN_GAP_S = 2000, 500, 2.0


class SandboxError(ValueError):
    pass


def _extract_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        obj = json.loads(t)
    except ValueError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m:
            raise SandboxError("модель не вернула JSON")
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            raise SandboxError("модель вернула некорректный JSON")
    if not isinstance(obj, dict):
        raise SandboxError("ожидался JSON-объект")
    return obj


class Sandbox:
    def __init__(self, gw: Gateway, settings, llm_call: Optional[Callable[[str, str], "tuple[str, dict]"]] = None) -> None:
        self.gw, self.settings, self._llm, self._lock, self._last = gw, settings, llm_call, threading.Lock(), 0.0

    def presets(self) -> list:
        return [{k: p[k] for k in ("id", "name", "prompt")} for p in PRESETS]

    def _agent(self, preset: dict):
        name = f"Песочница: {preset['name']}"
        for a in self.gw.agents.list():
            if a.name == name and not a.revoked:
                if a.mode != "shadow":
                    self.gw.agents.update(a.id, mode="shadow")
                    a.mode = "shadow"
                return a
        return self.gw.agents.create(name, "shadow", preset["profile"], created=self.gw.clock().isoformat(timespec="seconds"), daily_quota=500)[0]

    def _call_llm(self, system: str, user: str):
        if self._llm:
            return self._llm(system, user)
        import openai
        pub = self.settings.public()
        client = openai.OpenAI(api_key=self.settings.api_key(), base_url=pub["base_url"], timeout=60)
        try:
            r = client.chat.completions.create(model=pub["model"], temperature=0.3, max_tokens=500, response_format={"type": "json_object"},
                                               messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        except openai.AuthenticationError:
            raise SandboxError("провайдер отклонил ключ: проверьте его во вкладке «Настройки»")
        except openai.APIConnectionError:
            raise SandboxError("нет связи с провайдером модели: нужен интернет или VPN")
        except openai.APIError as exc:
            raise SandboxError(f"ошибка провайдера модели: {str(exc)[:160]}")
        u = getattr(r, "usage", None)
        return r.choices[0].message.content or "", {"prompt": getattr(u, "prompt_tokens", None), "completion": getattr(u, "completion_tokens", None)}

    def run(self, preset_id: str, symbol: str, note: str = "", custom_prompt: str = "") -> dict:
        preset = next((p for p in PRESETS if p["id"] == preset_id), None)
        if preset is None:
            raise SandboxError("неизвестный агент песочницы")
        prompt = (custom_prompt or "").strip()[:MAX_PROMPT] if preset["id"] == "custom" else preset["prompt"]
        if not prompt:
            raise SandboxError("для своего агента задайте промпт")
        symbol = (symbol or "").strip().upper()
        if not symbol or len(symbol) > 24:
            raise SandboxError("укажите тикер")
        if not self._llm and not self.settings.has_key():
            raise SandboxError("нужен ключ LLM: добавьте его во вкладке «Настройки»")
        with self._lock:
            wait = MIN_GAP_S - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            ag = self._agent(preset)
            try:
                ctx = self.gw.get_context(ag, symbol)
            except GatewayError as exc:
                return {"ok": True, "preset": preset["name"], "symbol": symbol, "model": None, "verdict": None,
                        "note": f"Контекст недоступен: {exc}. Модель не вызывалась.", "latency_ms": 0, "tokens": {}}
            user = "Контекст (проверен шлюзом):\n" + json.dumps(ctx, ensure_ascii=False, default=str) + (f"\nПожелание: {note.strip()[:MAX_NOTE]}" if (note or "").strip() else "")
            t0 = time.time()
            raw, tokens = self._call_llm(prompt + "\n\n" + FORMAT, user)
            ms = int((time.time() - t0) * 1000)
            out = {"ok": True, "preset": preset["name"], "symbol": symbol, "raw": str(raw)[:1500], "latency_ms": ms,
                   "tokens": tokens, "model": None, "verdict": None, "note": "", "agent_id": ag.id}
            try:
                dec = _extract_json(raw)
            except SandboxError as exc:
                out["note"] = f"{exc}. Это тоже результат теста: такой ответ шлюзу не отправляется."
                return out
            out["model"] = dec
            action = str(dec.get("action", "")).lower()
            if action not in ("buy", "sell"):
                out["note"] = "Агент решил не торговать (hold)." if action == "hold" else "Неизвестное действие в ответе модели: заявка не отправлена."
                return out
            intent = {"client_id": "sbx-" + uuid.uuid4().hex[:10], "symbol": symbol, "side": action, "quantity": dec.get("quantity"),
                      "order_type": dec.get("order_type", "market"), "limit_price": dec.get("limit_price"),
                      "confidence": dec.get("confidence"), "rationale": dec.get("rationale", "")}
            try:
                out["verdict"] = self.gw.propose(ag, intent)
            except GatewayError as exc:
                out["note"] = f"Шлюз не принял формат заявки: {exc}. Для теста это ошибка модели, а не шлюза."
            return out
