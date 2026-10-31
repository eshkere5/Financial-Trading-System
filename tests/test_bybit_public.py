from datetime import datetime, timedelta, timezone

import pytest

from finvex.core.errors import ProviderError
from finvex.venues.bybit.market import BybitPublicProvider

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


class FakeTransport:
    """Отдаёт свечи как Bybit: страницы от новых к старым, максимум 1000."""

    def __init__(self, total=1500, fail=False):
        self.total, self.fail, self.calls = total, fail, []
        self.t0 = int((NOW - timedelta(hours=total)).timestamp() * 1000)

    def get_json(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        if self.fail:
            return {"retCode": 10001, "retMsg": "params error"}
        if url.endswith("/instruments-info"):
            return {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "baseCoin": "BTC", "quoteCoin": "USDT"}]}}
        end, start = params["end"], params["start"]
        ts = [self.t0 + i * 3600_000 for i in range(self.total)]
        sel = sorted((t for t in ts if start <= t <= end), reverse=True)[: params["limit"]]
        return {"retCode": 0, "result": {"list": [[str(t), "100", "101", "99", "100.5", "10", "1000"] for t in sel]}}

    def close(self):
        pass


def mk(**kw):
    return BybitPublicProvider(transport=FakeTransport(**kw), now=lambda: NOW)


def test_paginates_back_to_start_and_orders_candles():
    p = mk(total=1500)
    s = p.get_candles("BTCUSDT", "1h", NOW - timedelta(hours=1400), NOW)
    assert len(s) > 1000 and len(p._tr.calls) >= 2
    ts = [c.ts_open for c in s.candles]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)
    assert s.instrument.venue == "BYBIT" and s.candles[-1].ts_close <= NOW   # формирующаяся свеча не отдаётся


def test_frame_has_expected_columns():
    df = mk(total=50).get_candles("BTCUSDT", "1h", NOW - timedelta(hours=40), NOW).to_frame()
    assert {"open", "high", "low", "close", "volume"} <= set(df.columns) and df["close"].iloc[-1] == 100.5


def test_api_error_is_reported_not_swallowed():
    with pytest.raises(ProviderError):
        mk(fail=True).get_candles("BTCUSDT", "1h", NOW - timedelta(hours=5), NOW)


def test_rejects_foreign_venue_and_bad_category_and_timeframe():
    p = mk()
    with pytest.raises(ProviderError):
        p.get_candles("SBER.MOEX", "1h", NOW - timedelta(hours=5), NOW)
    with pytest.raises(ValueError):
        BybitPublicProvider(category="x", transport=FakeTransport())
    with pytest.raises(Exception):
        p.get_candles("BTCUSDT", "10m", NOW - timedelta(hours=5), NOW)


def test_instruments_and_registry():
    from finvex.core.registry import available
    assert [i.id.symbol for i in mk().get_instruments()] == ["BTCUSDT"] and "bybit_public" in available("market_data")


def test_hub_price_source_routes_crypto_to_bybit():
    from finvex.core.registry import ProviderHub
    from finvex.gateway.prices import HubPriceSource
    from finvex.venues.csv_file import CsvFileProvider  # noqa: F401  (регистрирует csv)
    bybit, other = mk(total=500), mk(total=10)
    other._info = type(other._info)(name="x", venue="MOEX", capabilities=other._info.capabilities, timeframes=other._info.timeframes)
    hub = ProviderHub({"moex": other, "bb": bybit}, "moex")
    src = HubPriceSource(hub)
    df = src.candles("BTCUSDT", "1h", 50, NOW)
    assert len(df) == 50 and bybit._tr.calls and not other._tr.calls
