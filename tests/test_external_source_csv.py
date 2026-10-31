"""Подключение стороннего источника данных (биржа или брокер вне MOEX/Bybit) через конфиг и CSV: от providers.yaml до вердикта шлюза."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml

from finvex.core.models import MSK
from finvex.gateway.factory import build
from finvex.gateway.rules import session_open


@pytest.fixture
def root(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "csv").mkdir()
    end = datetime.now(MSK).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    for tf, step, n in (("1h", timedelta(hours=1), 420), ("1d", timedelta(days=1), 330)):
        last = end if tf == "1h" else end.replace(hour=0)
        ts = [(last - step * (n - 1 - k)).replace(tzinfo=None) for k in range(n)]
        px = 1500 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.0002, 0.01, n)))
        pd.DataFrame({"timestamp": ts, "open": px, "high": px * 1.004, "low": px * 0.996, "close": px, "volume": 1e5}).to_csv(tmp_path / "csv" / f"KZTEST_{tf}.csv", index=False)
    cfg = {"default_market_data": "other", "market_data": {"other": {"type": "csv_file", "params": {"directory": str(tmp_path / "csv"), "venue": "KASE"}}},
           "execution": {"type": "disabled", "enabled": False}}
    (tmp_path / "configs" / "providers.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "configs" / "gateway.yaml").write_text(yaml.safe_dump({"data_dir": "data/gw", "portfolio": {"cash": 1_000_000, "positions": {}}, "execution": {"enabled": False}}), encoding="utf-8")
    return tmp_path


def test_foreign_source_flows_from_config_to_gateway_verdict(root):
    gw = build(root, demo=False)
    ag, _ = gw.agents.create("kase", "shadow", "standard")
    rules = {k: dict(v) for k, v in gw.rules.items()}
    rules["trading_hours"]["enabled"] = False       # часы MOEX не относятся к стороннему рынку
    gw.set_rules(rules)
    ctx = gw.get_context(ag, "KZTEST")
    assert ctx["price"] > 0 and ctx["regime"]
    ok = gw.propose(ag, dict(client_id="a1", symbol="KZTEST", side="buy", quantity=10, confidence=0.7, rationale="Проверка стороннего источника данных."))
    assert ok["decision"] == "allow"
    ghost = gw.propose(ag, dict(client_id="a2", symbol="NOFILE", side="buy", quantity=10, confidence=0.7, rationale="Тикера нет в подключённом источнике."))
    assert ghost["decision"] == "deny" and "context" in ghost["codes"]


def test_known_limitation_session_calendar_is_moex_only():
    """Сейчас для всех некриптовых тикеров действует сессия MOEX (10:00-18:40, будни). Для других бирж календарь нужно задавать отдельно."""
    sunday = datetime(2026, 10, 25, 12, 0, tzinfo=MSK)
    assert session_open(sunday, "KZTEST") is False and session_open(sunday, "BTCUSDT") is True
