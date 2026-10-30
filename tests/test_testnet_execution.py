from types import SimpleNamespace

import pytest

from finvex.core.errors import ProviderError
from finvex.core.models import InstrumentId, OrderRequest, OrderType, Side
from finvex.execution.testnet import BybitTestnetExecution, TestVenueError, TestVenues, TinkoffSandboxExecution
from finvex.execution.testtrade import send_verdict


class FakeBybit:
    category = "spot"

    def __init__(self, testnet=True, retcode=0):
        self.cfg = SimpleNamespace(testnet=testnet)
        self.session = SimpleNamespace(place_order="place", cancel_order="cancel")
        self.calls, self.retcode = [], retcode

    def get_instrument_spec(self, sym):
        return {"qty_step": 0.001, "min_order_qty": 0.001}

    def _call(self, fn, *, write, **kw):
        self.calls.append((fn, kw))
        return {"retCode": self.retcode, "retMsg": "Insufficient balance", "result": {"orderId": "B-1"}}

    def get_portfolio(self):
        return {"cash": 1000.0}


class FakeTinkoff:
    def __init__(self, sandbox=True):
        self.sandbox, self.sent = sandbox, []

    def get_lot_size(self, sym):
        return 10

    def get_figi(self, sym):
        return "FIGI-" + sym

    def submit_order(self, figi, direction, quantity, price=None, order_type=None, order_id=None):
        self.sent.append((figi, direction, quantity, price, order_type, order_id))
        return "T-1"

    def get_portfolio(self):
        return {"cash": 1.0}


SDK = SimpleNamespace(OrderDirection=SimpleNamespace(ORDER_DIRECTION_BUY="BUY", ORDER_DIRECTION_SELL="SELL"),
                      OrderType=SimpleNamespace(ORDER_TYPE_MARKET="MKT", ORDER_TYPE_LIMIT="LMT"), decimal_to_quotation=lambda d: ("q", str(d)))


def req(sym, venue, qty, otype=OrderType.MARKET, price=None, side=Side.BUY):
    return OrderRequest("p1", "c1", InstrumentId(sym, venue), side, qty, otype, price)


def test_both_adapters_refuse_a_non_test_client():
    with pytest.raises(ValueError):
        BybitTestnetExecution(client=FakeBybit(testnet=False))
    with pytest.raises(ValueError):
        TinkoffSandboxExecution(client=FakeTinkoff(sandbox=False), sdk=SDK)


def test_bybit_market_buy_uses_base_coin_and_rounds_down():
    c = FakeBybit()
    ack = BybitTestnetExecution(client=c).submit(req("BTCUSDT", "BYBIT", 0.00555))
    kw = c.calls[0][1]
    assert ack.status == "accepted" and ack.venue_order_id == "B-1"
    assert kw["qty"] == "0.005" and kw["orderType"] == "Market" and kw["marketUnit"] == "baseCoin" and kw["side"] == "Buy"


def test_bybit_limit_and_below_minimum_and_api_rejection():
    c = FakeBybit()
    ex = BybitTestnetExecution(client=c)
    ex.submit(req("BTCUSDT", "BYBIT", 0.01, OrderType.LIMIT, 90000.0))
    assert c.calls[0][1]["orderType"] == "Limit" and c.calls[0][1]["price"] == "90000.0" and "marketUnit" not in c.calls[0][1]
    assert ex.submit(req("BTCUSDT", "BYBIT", 0.0001)).status.startswith("rejected") and len(c.calls) == 1
    assert BybitTestnetExecution(client=FakeBybit(retcode=10001)).submit(req("BTCUSDT", "BYBIT", 0.01)).status == "rejected: Insufficient balance"


def test_wrong_venue_is_rejected_by_each_adapter():
    with pytest.raises(ProviderError):
        BybitTestnetExecution(client=FakeBybit()).submit(req("SBER", "MOEX", 1))
    with pytest.raises(ProviderError):
        TinkoffSandboxExecution(client=FakeTinkoff(), sdk=SDK).submit(req("BTCUSDT", "BYBIT", 1))


def test_tinkoff_converts_shares_to_lots():
    c = FakeTinkoff()
    ex = TinkoffSandboxExecution(client=c, sdk=SDK)
    ack = ex.submit(req("SBER", "MOEX", 35))
    assert ack.status == "accepted" and c.sent[0][:3] == ("FIGI-SBER", "BUY", 3) and c.sent[0][4] == "MKT" and c.sent[0][5] == "c1"
    assert ex.submit(req("SBER", "MOEX", 5)).status.startswith("rejected") and len(c.sent) == 1
    ex.submit(req("SBER", "MOEX", 20, OrderType.LIMIT, 300.5, Side.SELL))
    assert c.sent[1][1] == "SELL" and c.sent[1][3] == ("q", "300.5") and c.sent[1][4] == "LMT"


def test_router_picks_venue_and_reports_missing_keys():
    assert TestVenues.venue_for("BTCUSDT") == "BYBIT" and TestVenues.venue_for("SBER") == "MOEX"

    def no_keys():
        raise TestVenueError("ключ не найден. Нужен configs/…")

    tv = TestVenues({"BYBIT": lambda: BybitTestnetExecution(client=FakeBybit()), "MOEX": no_keys})
    assert tv.for_symbol("ETHUSDT").info.name == "bybit_testnet"
    st = {s["venue"]: s for s in tv.status()}
    assert st["BYBIT"]["ready"] and not st["MOEX"]["ready"] and "ключ" in st["MOEX"]["message"]


def verdict(decision="allow", symbol="BTCUSDT", qty=0.01):
    return {"id": "vd_1", "decision": decision, "intent": {"symbol": symbol, "side": "buy", "quantity": qty, "order_type": "market", "limit_price": None}}


def test_only_allowed_verdicts_reach_the_test_venue():
    c = FakeBybit()
    tv = TestVenues({"BYBIT": lambda: BybitTestnetExecution(client=c), "MOEX": lambda: None})
    assert send_verdict(tv, verdict("deny"))["sent"] is False and not c.calls
    assert send_verdict(tv, verdict("revise"))["sent"] is False and not c.calls
    out = send_verdict(tv, verdict())
    assert out["sent"] and out["venue"] == "BYBIT" and out["order_id"] == "B-1" and len(c.calls) == 1


def test_send_verdict_reports_missing_keys_and_broker_errors_without_raising():
    def no_keys():
        raise TestVenueError("Нужен ключ тестовой сети")

    assert "Нужен ключ" in send_verdict(TestVenues({"BYBIT": no_keys, "MOEX": no_keys}), verdict())["message"]
    bad = TestVenues({"BYBIT": lambda: BybitTestnetExecution(client=FakeBybit(retcode=10001)), "MOEX": no_keys})
    out = send_verdict(bad, verdict())
    assert out["sent"] is False and "Insufficient" in out["message"]


def test_registered_for_config_use_and_importable_without_sdk():
    from finvex.core.registry import available
    assert {"bybit_testnet", "tinkoff_sandbox"} <= set(available("execution"))
