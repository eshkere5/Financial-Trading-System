"""Шлюз: принимает намерения агентов, проверяет по правилам и контексту, пишет журнал.

Принципы: fail-closed (любая ошибка означает отказ), агент не получает токен брокера,
исполнение возможно только в режиме enforce и только если подключён включённый адаптер.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from finvex.core.models import MSK, InstrumentId, OrderRequest, OrderType, Side

from . import rules as R
from .agents import Agent, AgentStore
from .context import ContextBuilder, PaperPortfolio
from finvex.modules import ModuleDisabled, ModuleRegistry

from .ledger import Ledger
from .models import Check, Intent, Verdict, now_iso
from .analytics import AnalyticsStore
from .prices import PriceSource
from .report import build_report


class GatewayError(Exception):
    def __init__(self, code: str, message: str, http: int = 400) -> None:
        super().__init__(message)
        self.code, self.http = code, http


class Gateway:
    def __init__(self, data_dir: "str | Path | None", prices: PriceSource, portfolio: Optional[PaperPortfolio] = None,
                 clock: Optional[Callable[[], datetime]] = None, executor=None, execution_enabled: bool = False) -> None:
        self.data_dir = Path(data_dir) if data_dir else None
        self.clock = clock or (lambda: datetime.now(MSK))
        self.agents = AgentStore(self.data_dir / "agents.json" if self.data_dir else None)
        self.ledger = Ledger(self.data_dir / "ledger.jsonl" if self.data_dir else None)
        self.modules = ModuleRegistry(self.data_dir / "modules.json" if self.data_dir else None)
        self.rules_path = self.data_dir / "rules.json" if self.data_dir else None
        self.rules = R.load_rules(self.rules_path)
        self.context = ContextBuilder(prices, portfolio or PaperPortfolio(), self.clock)
        self.analytics = AnalyticsStore(self.data_dir / "analytics.json" if self.data_dir else None, self.clock)
        self.executor, self.execution_enabled = executor, execution_enabled
        self._lock = threading.RLock()
        self._verdicts: "list[dict]" = []
        self._by_id: dict = {}
        self._by_client: dict = {}
        for row in self.ledger.rows():
            self._index({**row["data"], "ledger_hash": row["hash"]})

    def _index(self, v: dict) -> None:
        self._verdicts.append(v)
        self._by_id[v["id"]] = v
        self._by_client[(v["agent_id"], v["intent"]["client_id"])] = v["id"]

    def set_rules(self, rules: dict) -> dict:
        with self._lock:
            self.rules = R.normalize(rules)
            R.save_rules(self.rules_path, self.rules)
            return self.rules

    def authenticate(self, key: str) -> Agent:
        ag = self.agents.authenticate(key)
        if ag is None:
            raise GatewayError("unauthorized", "неверный или отозванный ключ агента", 401)
        return ag

    def _agent_rules(self, ag: Agent) -> dict:
        return R.merge(self.rules, ag.overrides)

    def _guard(self) -> None:
        try:
            self.modules.require("guard")
        except ModuleDisabled as exc:
            raise GatewayError("module_off", f"{exc}: заявки не принимаются", 503)

    def limits(self, ag: Agent) -> dict:
        self._guard()
        rules = self._agent_rules(ag)
        used = self._orders_today(ag.id)
        on = rules["max_orders_per_day"]["enabled"]
        return {"rules": {k: v for k, v in rules.items() if v["enabled"]}, "orders_today": used,
                "orders_left": max(int(rules["max_orders_per_day"]["value"]) - used, 0) if on else None, "mode": ag.mode}

    def get_context(self, ag: Agent, symbol: str) -> dict:
        self._guard()
        symbol = symbol.strip().upper()
        try:
            ctx = self.context.build(symbol)
        except Exception as exc:
            msg = str(exc)
            raise GatewayError("context_unavailable", msg if msg.startswith("нет данных") else f"нет данных по {symbol}: {msg[:120]}", 503)
        return self.context.for_agent(ctx, ag.context_profile, self.limits(ag))

    def _orders_today(self, agent_id: str) -> int:
        today = self.clock().date().isoformat()
        return sum(1 for v in self._verdicts if v["agent_id"] == agent_id and v["ts"][:10] == today)

    def _last_same(self, agent_id: str, intent: Intent) -> Optional[datetime]:
        for v in reversed(self._verdicts):
            i = v["intent"]
            if (v["agent_id"] == agent_id and v["decision"] == "allow"
                    and i["symbol"] == intent.symbol and i["side"] == intent.side):
                return datetime.fromisoformat(v["ts"])
        return None

    def _revisions(self, agent_id: str, intent: Intent, now: datetime, window_s: float = 3600.0) -> int:
        n = 0
        for v in reversed(self._verdicts):
            if (now - datetime.fromisoformat(v["ts"])).total_seconds() > window_s:
                break
            i = v["intent"]
            if v["agent_id"] == agent_id and v["decision"] == "revise" and i["symbol"] == intent.symbol and i["side"] == intent.side:
                n += 1
        return n

    def propose(self, ag: Agent, raw: dict) -> dict:
        self._guard()   # fail-closed: без защитного слоя заявки не принимаются и не исполняются
        try:
            intent = Intent.from_dict(raw)
        except ValueError as exc:
            raise GatewayError("bad_intent", str(exc), 422)
        with self._lock:
            prior = self._by_client.get((ag.id, intent.client_id))
            if prior:
                return {**self._by_id[prior], "duplicate": True}
            now = self.clock()
            checks: "list[Check]" = []
            ctx: dict = {}
            decision, suggested = "deny", None
            try:
                if self._orders_today(ag.id) >= ag.daily_quota:
                    checks.append(Check("quota", "Суточная квота агента", "fail", f"Квота {ag.daily_quota} исчерпана"))
                else:
                    ctx = self.context.build(intent.symbol)
                    an = self.analytics.snapshot(intent.symbol, now) if self.modules.is_enabled("analytics") else {}
                    state = R.State(ctx=ctx, now=now, orders_today=self._orders_today(ag.id),
                                    last_same=self._last_same(ag.id, intent), position_qty=ctx["position_qty"],
                                    equity=ctx["equity"], analytics=an, revisions=self._revisions(ag.id, intent, now))
                    decision, checks, suggested = R.evaluate(self._agent_rules(ag), intent, state)
            except Exception as exc:
                decision = "deny"
                checks.append(Check("context", "Контекст рынка", "fail", f"Нет проверенных данных: {str(exc)[:160]}"))
            reasons = [c.message for c in checks if c.status == "fail"]
            codes = [c.rule_id for c in checks if c.status == "fail"]
            warns = [c.message for c in checks if c.status == "warn"]
            retry = max((c.retry_after for c in checks if c.status == "fail" and c.retry_after), default=None) if decision == "revise" else None
            keys = ("price", "price_time", "data_age_min", "regime", "adx", "position_qty", "equity")
            v = Verdict(id="vd_" + uuid.uuid4().hex[:10], ts=now_iso(now), agent_id=ag.id, agent_name=ag.name,
                        mode=ag.mode, decision=decision, intent=intent.to_dict(), checks=[c.to_dict() for c in checks],
                        reasons=reasons, suggested_quantity=suggested, context=({**{k: ctx.get(k) for k in keys}, "analytics_used": {k: an[k] for k in ("news_score", "news_ids", "kronos_score", "kronos_valid_until") if k in an}} if ctx else {}),
                        retry_after=retry,
                        codes=codes, mandate_hash=R.mandate_hash(self._agent_rules(ag)),
                        agent_message=self._message(decision, reasons, warns, suggested, retry))
            self._execute(ag, v, intent)
            data = v.to_dict()
            data["ledger_hash"] = self.ledger.append({k: x for k, x in data.items() if k != "ledger_hash"})
            self._index(data)
            return data

    @staticmethod
    def _message(decision: str, reasons: list, warns: list, suggested, retry=None) -> str:
        """Пояснение для агента: что случилось и что с этим делать."""
        why = " ".join(r.rstrip(".") + "." for r in reasons)
        note = (" Замечания, которые не блокируют: " + "; ".join(w.rstrip(".") for w in warns) + ".") if warns else ""
        if decision == "allow":
            return "Принято." + note
        if decision == "revise":
            if retry:
                wait = f"{retry / 3600:.1f} ч." if retry >= 3600 else (f"{retry / 60:.0f} мин." if retry >= 90 else f"{retry:.0f} сек.")
                fix = f" Ошибки в заявке нет, повторите через {wait} с новым client_id."
            else:
                fix = f" Допустимый объём: {suggested:g}." if suggested is not None else " Исправьте указанное и отправьте заново с новым client_id."
            return "На доработку. " + why + fix + note
        return "Отклонено. " + why + " Повторная отправка без изменения условий не поможет." + note

    def agent_counts(self) -> dict:
        cnt: dict = {}
        for v in self._verdicts:
            cnt[v["agent_id"]] = cnt.get(v["agent_id"], 0) + 1
        return cnt

    def candles(self, symbol: str, timeframe: str, bars: int):
        return self.context.prices.candles(symbol, timeframe, bars, self.clock())

    def report(self) -> dict:
        return build_report(self._verdicts)

    def _execute(self, ag: Agent, v: Verdict, intent: Intent) -> None:
        if v.decision != "allow":
            v.execution_note = "Не исполнено: вердикт не разрешает"
            return
        if ag.mode != "enforce":
            v.execution_note = "Режим проверки: исполнение не запрашивалось"
            return
        if not (self.execution_enabled and self.executor is not None):
            v.execution_note = "Исполнение не подключено: заявка не отправлена"
            return
        try:
            req = OrderRequest(plan_id=v.id, client_order_id=v.id, instrument=InstrumentId.parse(intent.symbol, "MOEX"),
                               side=Side(intent.side), quantity=intent.quantity,
                               order_type=OrderType(intent.order_type), limit_price=intent.limit_price)
            ack = self.executor.submit(req)
            v.executed, v.execution_note = True, f"Отправлено брокеру: {ack.status}"
        except Exception as exc:
            v.execution_note = f"Брокер отклонил или недоступен: {str(exc)[:140]}"

    def get_verdict(self, ag: Optional[Agent], verdict_id: str) -> dict:
        v = self._by_id.get(verdict_id)
        if not v or (ag is not None and v["agent_id"] != ag.id):
            raise GatewayError("not_found", "вердикт не найден", 404)
        return v

    def list_verdicts(self, limit: int = 100, decision: str = "", agent_id: str = "") -> list:
        out = [v for v in reversed(self._verdicts)
               if (not decision or v["decision"] == decision) and (not agent_id or v["agent_id"] == agent_id)]
        return out[:max(1, min(limit, 1000))]

    def stats(self) -> dict:
        today = self.clock().date().isoformat()
        day = [v for v in self._verdicts if v["ts"][:10] == today] or self._verdicts[-50:]
        cnt = {"allow": 0, "deny": 0, "revise": 0}
        for v in day:
            cnt[v["decision"]] += 1
        return {"total": len(day), **cnt, "all_time": len(self._verdicts), "agents": len(self.agents.list()),
                "ledger": self.ledger.verify()}
