"""Агенты, которыми управляет Finvex: нейросеть по API (llm) и агент по адресу (http, например n8n или Dify).

Finvex получает проверенный контекст от шлюза, спрашивает агента и отправляет его решение в шлюз как обычное намерение.
Любой сбой (нет контекста, ответ не JSON, ошибка провайдера, агент недоступен) означает «ничего не делать».
"""
from __future__ import annotations

import json
import threading
from typing import Callable, Optional

from .agents import Agent
from .decisions import FORMAT, DecisionError, call_http, decision_to_intent, extract_json
from .gateway import Gateway, GatewayError

CHAT_RULES = ("Отвечай коротко и по делу на русском. Ты не можешь отправлять заявки в этом диалоге: "
              "только объясняешь свою позицию и решения.")
MAX_CTX = 6000


class AgentRunner:
    def __init__(self, gw: Gateway, llm_call: Optional[Callable] = None, default_llm: Optional[Callable[[], dict]] = None) -> None:
        self.gw, self._llm, self._default = gw, llm_call, default_llm or (lambda: {})
        self._busy: dict = {}
        self._guard = threading.Lock()

    def _endpoint(self, ag: Agent) -> dict:
        own_base = ag.config.get("base_url") or ""
        shared = self._default() or {}
        key = self.gw.agents.secret(ag.id, "api_key")
        if not own_base and not key:
            key = shared.get("api_key", "")        # общий ключ уходит только на общий адрес, не на чужой
        return {"base_url": own_base or shared.get("base_url", ""), "model": ag.config.get("model") or shared.get("model", ""), "api_key": key}

    def _call(self, ag: Agent, system: str, user: str, json_mode: bool) -> str:
        if self._llm:
            return str(self._llm(system, user))
        ep = self._endpoint(ag)
        local = ep["base_url"].startswith("http://")
        if not ep["api_key"] and not local:
            raise DecisionError("у агента нет ключа модели: добавьте его в карточке агента или в «Настройках»")
        if not ep["base_url"] or not ep["model"]:
            raise DecisionError("не задан адрес или название модели")
        import openai
        client = openai.OpenAI(api_key=ep["api_key"] or "local", base_url=ep["base_url"], timeout=60)
        kw = dict(model=ep["model"], temperature=0.2, max_tokens=700,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        try:
            try:
                r = client.chat.completions.create(**kw, **({"response_format": {"type": "json_object"}} if json_mode else {}))
            except openai.BadRequestError:
                r = client.chat.completions.create(**kw)
        except openai.AuthenticationError:
            raise DecisionError("провайдер модели отклонил ключ агента")
        except openai.APIConnectionError:
            raise DecisionError("нет связи с провайдером модели: проверьте адрес, интернет или VPN")
        except openai.APIError as exc:
            raise DecisionError(f"ошибка провайдера модели: {str(exc)[:160]}")
        return r.choices[0].message.content or ""

    def decide(self, ag: Agent, ctx: dict) -> dict:
        """Решение агента по контексту. Бросает DecisionError, если агент не смог ответить."""
        if ag.kind == "llm":
            user = "Контекст (проверен шлюзом):\n" + json.dumps(ctx, ensure_ascii=False, default=str)[:MAX_CTX]
            return extract_json(self._call(ag, ag.config.get("system_prompt", "") + "\n\n" + FORMAT, user, True))
        if ag.kind == "http":
            payload = {"symbol": ctx.get("symbol"), "context": ctx, "instructions": ag.config.get("system_prompt", ""), "format": FORMAT}
            return call_http(ag.config["url"], self.gw.agents.secret(ag.id, "api_key"), payload)
        raise DecisionError("этого агента Finvex не вызывает: он сам приходит с ключом")

    def _agent(self, agent_id: str, kinds=("llm", "http")) -> Agent:
        ag = self.gw.agents.get(agent_id)
        if ag is None:
            raise GatewayError("not_found", "агент не найден", 404)
        if ag.kind not in kinds:
            raise GatewayError("bad_kind", "этот агент подключён снаружи: его запускает он сам", 409)
        if ag.revoked:
            raise GatewayError("revoked", "агент отозван", 409)
        return ag

    def run(self, agent_id: str, symbols: Optional[list] = None) -> dict:
        ag = self._agent(agent_id)
        syms = [s.upper() for s in (symbols or ag.config.get("symbols") or [])][:10]
        if not syms:
            raise GatewayError("no_symbols", "у агента не заданы тикеры", 422)
        with self._guard:
            if self._busy.get(ag.id):
                raise GatewayError("busy", "агент уже выполняется", 409)
            self._busy[ag.id] = True
        try:
            return {"agent": ag.id, "mode": ag.mode, "results": [self._one(ag, s) for s in syms]}
        finally:
            with self._guard:
                self._busy.pop(ag.id, None)

    def _one(self, ag: Agent, symbol: str) -> dict:
        try:
            ctx = self.gw.get_context(ag, symbol)
        except GatewayError as exc:
            return {"symbol": symbol, "status": "no_context", "note": f"{exc}. Агент не вызывался."}
        try:
            dec = self.decide(ag, ctx)
        except Exception as exc:                        # любой сбой агента = не торговать
            return {"symbol": symbol, "status": "error", "note": f"{str(exc)[:200]}. Заявка не отправлена."}
        action = str(dec.get("action", "")).lower()
        if action not in ("buy", "sell"):
            return {"symbol": symbol, "status": "hold" if action == "hold" else "error", "model": dec,
                    "note": "Агент решил не торговать." if action == "hold" else "Неизвестное действие: заявка не отправлена."}
        try:
            return {"symbol": symbol, "status": "verdict", "model": dec, "verdict": self.gw.propose(ag, decision_to_intent(dec, symbol))}
        except GatewayError as exc:
            return {"symbol": symbol, "status": "error", "model": dec, "note": f"Шлюз не принял заявку: {exc}"}

    def chat(self, agent_id: str, text: str) -> str:
        ag = self._agent(agent_id, ("llm",))
        last = "; ".join(f"{v['intent']['side']} {v['intent']['symbol']} {v['intent']['quantity']:g} -> {v['decision']}"
                         for v in self.gw.list_verdicts(5, "", ag.id)) or "решений пока нет"
        system = ag.config.get("system_prompt", "") + "\n\n" + CHAT_RULES + f"\nПоследние решения шлюза по тебе: {last}."
        try:
            return self._call(ag, system, str(text)[:2000], False).strip() or "Модель вернула пустой ответ."
        except DecisionError as exc:
            raise GatewayError("agent_unavailable", str(exc), 502)
