"""Подставной стек для самопроверки прогона: демо-шлюз, сценарная «модель» и условные тестовые контуры. Не для оценки качества."""
import json
from itertools import cycle
from types import SimpleNamespace

from finvex.execution.testnet import TestVenues
from finvex.gateway.factory import build
from finvex.web import sandbox as sbx

SCRIPT = [
    {"action": "buy", "quantity": 0.01, "order_type": "market", "limit_price": None, "confidence": 0.7, "rationale": "Режим бычий, ADX подтверждает тренд, риск ограничен стопом."},
    {"action": "hold", "rationale": "Нет явного основания."},
    {"action": "buy", "quantity": 5000, "order_type": "market", "limit_price": None, "confidence": 0.8, "rationale": "Агрессивный вход крупным объёмом."},
    "это не JSON",
    {"action": "sell", "quantity": 100000, "order_type": "market", "limit_price": None, "confidence": 0.6, "rationale": "Продажа сверх позиции."},
    {"action": "buy", "quantity": 3, "order_type": "market", "limit_price": None, "confidence": 0.65, "rationale": "Отскок от поддержки, ожидаю возврат к средней."},
]


class FakeVenue:
    def __init__(self, name):
        self.n, self.name = 0, name

    def submit(self, order):
        from finvex.core.models import OrderAck
        from datetime import datetime, timezone
        self.n += 1
        return OrderAck(order.client_order_id, f"{self.name}-{self.n:04d}", "accepted", datetime.now(timezone.utc))

    def portfolio(self):
        return {"cash": 10000.0 - 10 * self.n, "positions": self.n}


def make_fake():
    sbx.MIN_GAP_S = 0
    it = cycle(SCRIPT)

    def llm(system, user):
        d = next(it)
        return (d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)), {"prompt": 900, "completion": 80}

    gw = build(".", demo=True)
    sandbox = sbx.Sandbox(gw, SimpleNamespace(has_key=lambda: True), llm)
    venues = TestVenues({"BYBIT": lambda: FakeVenue("BYBIT"), "MOEX": lambda: FakeVenue("MOEX")})
    venues.status = lambda: [{"venue": "BYBIT", "title": "Bybit Testnet (подставной)", "ready": True, "message": ""},
                             {"venue": "MOEX", "title": "T-Invest Sandbox (подставной)", "ready": True, "message": ""}]
    return gw, sandbox, venues
