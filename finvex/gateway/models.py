"""Типы шлюза: намерение агента, проверка, вердикт."""
from __future__ import annotations

import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

SYMBOL_RE = re.compile(r"^[A-Z0-9_.\-]{1,24}$")


@dataclass
class Intent:
    """То, что агент хочет сделать. Это не заявка: заявкой она станет только после вердикта."""
    client_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    rationale: str = ""
    confidence: Optional[float] = None
    sources: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Intent":
        if not isinstance(d, dict):
            raise ValueError("намерение должно быть объектом")
        symbol = str(d.get("symbol", "")).strip().upper()
        if not SYMBOL_RE.match(symbol):
            raise ValueError("некорректный тикер")
        side = str(d.get("side", "")).strip().lower()
        if side not in ("buy", "sell"):
            raise ValueError("side: buy или sell")
        try:
            qty = float(d.get("quantity"))
        except (TypeError, ValueError):
            raise ValueError("quantity должна быть числом")
        if not math.isfinite(qty) or qty <= 0:
            raise ValueError("quantity должна быть > 0")
        otype = str(d.get("order_type", "market")).strip().lower()
        if otype not in ("market", "limit"):
            raise ValueError("order_type: market или limit")
        lp = d.get("limit_price")
        if lp in ("", None):
            lp = None
        else:
            try:
                lp = float(lp)
            except (TypeError, ValueError):
                raise ValueError("limit_price должна быть числом")
            if not math.isfinite(lp) or lp <= 0:
                raise ValueError("limit_price должна быть > 0")
        if otype == "limit" and lp is None:
            raise ValueError("для limit нужен limit_price")
        conf = d.get("confidence")
        if conf in ("", None):
            conf = None
        else:
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                raise ValueError("confidence должна быть числом")
            if not 0.0 <= conf <= 1.0:
                raise ValueError("confidence в диапазоне 0..1")
        srcs = d.get("sources") or []
        if not isinstance(srcs, list):
            raise ValueError("sources должен быть списком")
        return cls(client_id=str(d.get("client_id") or uuid.uuid4().hex[:12])[:64], symbol=symbol, side=side,
                   quantity=qty, order_type=otype, limit_price=lp, rationale=str(d.get("rationale") or "")[:4000],
                   confidence=conf, sources=[str(s)[:200] for s in srcs[:10]])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Check:
    rule_id: str
    name: str
    status: str            # pass | fail | warn | skip
    message: str
    severity: str = "block"  # block | revise
    suggested_quantity: Optional[float] = None
    retry_after: Optional[float] = None   # через сколько секунд повтор имеет смысл

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    id: str
    ts: str
    agent_id: str
    agent_name: str
    mode: str
    decision: str          # allow | deny | revise
    intent: dict
    checks: list
    reasons: list
    suggested_quantity: Optional[float] = None
    executed: bool = False
    execution_note: str = ""
    context: dict = field(default_factory=dict)
    codes: list = field(default_factory=list)      # rule_id нарушенных правил
    mandate_hash: str = ""                         # отпечаток правил на момент решения
    retry_after: Optional[float] = None
    agent_message: str = ""                        # понятное агенту пояснение
    ledger_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def now_iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")
