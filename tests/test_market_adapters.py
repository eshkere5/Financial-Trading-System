"""
Тесты слоя адаптеров. Работают без сети.
Ответы ISS в этих тестах — вручную собранные payload в формате ISS (значения условные,
проверяется разбор и логика, а не рынок). Реальные ответы: tools/record_moex_fixtures.py,
затем test_recorded_fixtures; живая проверка: FINVEX_LIVE=1.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finvex.core import (
    Capability, ConfigError, DataFormatError, ExecutionDisabledError, InstrumentId, MSK, OrderRequest,
    ProviderError, ProviderHub, Side, Timeframe, UnknownProviderError, UnsupportedCapability,
    UnsupportedTimeframe, available, check_candles, create,
)
from finvex.core.base import MarketDataProvider, ProviderInfo
from finvex.core.models import Candle
from finvex.venues.moex import MoexISSProvider
from finvex.analysis.kronos.bridge import to_kronos_frame

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures" / "moex"
NOW = datetime(2026, 10, 14, 13, 30, tzinfo=timezone.utc)  # 16:30 MSK
START = datetime(2026, 10, 14, 0, 0, tzinfo=MSK)

CANDLE_COLS = ["open", "close", "high", "low", "value", "volume", "begin", "end"]


def hourly_payload(hours=(10, 11, 12), day="2026-10-14", base=280.0):
    rows = []
    for i, h in enumerate(hours):
        o = base + i
        rows.append([o, o + 0.5, o + 1.0, o - 0.5, 1_000_000.0 + i, 3000 + i,
                     f"{day} {h:02d}:00:00", f"{day} {h:02d}:59:59"])
    return {"candles": {"columns": CANDLE_COLS, "data": rows}}


SECURITIES = {"securities": {
    "columns": ["SECID", "BOARDID", "SHORTNAME", "ISIN", "LOTSIZE", "MINSTEP", "DECIMALS", "CURRENCYID"],
    "data": [["SBER", "TQBR", "Сбербанк", "RU0009029540", 10, 0.01, 2, "SUR"],
             ["GAZP", "TQBR", "Газпром", "RU0007661625", 10, 0.01, 2, "SUR"]]}}

MARKETDATA = {"marketdata": {
    "columns": ["SECID", "BID", "OFFER", "LAST", "VOLTODAY", "TRADINGSTATUS", "TIME", "SYSTIME"],
    "data": [["SBER", 283.75, 283.77, 283.78, 2687155, "T", "11:16:50", "2026-10-14 11:31:49"],
             ["GAZP", None, None, None, None, "N", "", "2026-10-14 11:31:49"]]}}

TRADES = {"trades": {
    "columns": ["TRADENO", "PRICE", "QUANTITY", "BUYSELL", "SYSTIME"],
    "data": [[5, 283.77, 3, "B", "2026-10-14 11:16:50"], [4, 283.76, 2, "S", "2026-10-14 11:16:49"],
             [3, 283.75, 1, "S", "2026-10-14 11:16:48"]]}}


class Replay:
    """Транспорт-заглушка: отдаёт заданные payload; свечи листает страницами, как ISS."""

    def __init__(self, routes, page_size=500):
        self.routes, self.page_size, self.calls = routes, page_size, []

    def get_json(self, url, params=None):
        params = dict(params or {})
        self.calls.append((url, params))
        for tail, payload in self.routes.items():
            if url.endswith(tail):
                if tail.endswith("candles.json"):
                    off = int(params.get("start", 0))
                    blk = payload["candles"]
                    return {"candles": {"columns": blk["columns"], "data": blk["data"][off:off + self.page_size]}}
                return payload
        raise ProviderError(f"нет маршрута {url}")

    def close(self):
        pass


def moex(routes=None, page_size=500, now=NOW, **kw):
    routes = routes if routes is not None else {
        "TQBR/securities.json": SECURITIES,
        "SBER/candles.json": hourly_payload(),
        "SBER/trades.json": TRADES,
    }
    return MoexISSProvider(transport=Replay(routes, page_size), now=lambda: now, **kw)


# ── MOEX: инструменты, свечи ──────────────────────────────────────────────
def test_instruments_parsed():
    items = {i.id.symbol: i for i in moex().get_instruments()}
    sber = items["SBER"]
    assert str(sber.id) == "SBER.MOEX" and sber.lot_size == 10 and sber.tick_size == 0.01
    assert sber.currency == "SUR" and sber.board == "TQBR" and sber.isin == "RU0009029540"


def test_candles_parsed_and_timezone_converted():
    s = moex().get_candles("SBER", "1h", START, NOW)
    assert len(s) == 3 and s.timeframe is Timeframe.H1 and s.source.startswith("moex_iss")
    first = s.candles[0]
    assert first.ts_open == datetime(2026, 10, 14, 7, 0, tzinfo=timezone.utc)  # 10:00 MSK
    assert first.ts_close.minute == 59 and first.turnover == 1_000_000.0
    assert s.quality.ok


def test_pagination_follows_start_offset():
    p = moex(page_size=2)
    assert len(p.get_candles("SBER", "1h", START, NOW)) == 3
    assert [c[1]["start"] for c in p._tr.calls] == [0, 2, 3]


def test_only_closed_drops_forming_bar():
    cut = datetime(2026, 10, 14, 8, 30, tzinfo=timezone.utc)  # 11:30 MSK: бар 11:00 ещё идёт
    p = moex(now=cut)
    closed = p.get_candles("SBER", "1h", START, cut)
    assert [c.ts_open.astimezone(MSK).hour for c in closed.candles] == [10]
    with_open = p.get_candles("SBER", "1h", START, cut, only_closed=False)
    assert [c.ts_open.astimezone(MSK).hour for c in with_open.candles] == [10, 11]


def test_unsupported_timeframe_rejected_before_network():
    p = moex()
    with pytest.raises(UnsupportedTimeframe):
        p.get_candles("SBER", "5m", START, NOW)
    assert p._tr.calls == []


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        moex().get_candles("SBER", "1h", datetime(2026, 10, 14), NOW)


def test_end_before_start_rejected():
    with pytest.raises(ValueError):
        moex().get_candles("SBER", "1h", NOW, START)


def test_foreign_venue_rejected():
    with pytest.raises(ProviderError):
        moex().get_candles("BTCUSDT.BYBIT", "1h", START, NOW)


def test_index_zero_volume_is_warning_not_error():
    rows = [[2800 + i, 2801 + i, 2810 + i, 2790 + i, 0.0, 0, f"2026-10-{12 + i} 00:00:00", f"2026-10-{12 + i} 23:59:59"]
            for i in range(2)]
    p = moex({"IMOEX/candles.json": {"candles": {"columns": CANDLE_COLS, "data": rows}}}, market="index", board="SNDX")
    s = p.get_candles("IMOEX", "1d", datetime(2026, 10, 12, tzinfo=MSK), NOW + timedelta(days=1))
    assert s.quality.ok and any(i.code == "zero_volume" for i in s.quality.issues)


# ── MOEX: котировки, сделки ───────────────────────────────────────────────
def test_quote_times_snapshot_vs_last_trade():
    p = moex({"TQBR/securities.json": MARKETDATA})
    q = {x.instrument.symbol: x for x in p.get_quotes(["SBER", "GAZP"])}["SBER"]
    assert q.spread == pytest.approx(0.02) and q.mid == pytest.approx(283.76)
    assert q.ts_snapshot == datetime(2026, 10, 14, 8, 31, 49, tzinfo=timezone.utc)
    assert q.ts_last_trade == datetime(2026, 10, 14, 8, 16, 50, tzinfo=timezone.utc)
    assert q.age_seconds(q.ts_snapshot) == pytest.approx(899)  # цена старше снимка на 15 минут
    assert p._tr.calls[0][1]["securities"] == "SBER,GAZP"


def test_quote_without_trades_has_no_age():
    p = moex({"TQBR/securities.json": MARKETDATA})
    g = {x.instrument.symbol: x for x in p.get_quotes()}["GAZP"]
    assert g.last is None and g.ts_last_trade is None and g.age_seconds(NOW) is None and g.spread is None


def test_last_trade_after_midnight_rolls_to_previous_day():
    data = {"marketdata": {"columns": ["SECID", "TIME", "SYSTIME"],
                           "data": [["SBER", "23:59:50", "2026-10-14 00:00:10"]]}}
    q = moex({"TQBR/securities.json": data}).get_quotes()[0]
    assert q.ts_last_trade == datetime(2026, 10, 13, 20, 59, 50, tzinfo=timezone.utc)


def test_trades_sorted_and_limit_snapped():
    p = moex()
    trades = p.get_trades("SBER", limit=2)
    call = p._tr.calls[0][1]
    assert call["limit"] == 10 and call["reversed"] == 1
    assert len(trades) == 2 and [t.ts_event for t in trades] == sorted(t.ts_event for t in trades)
    assert {t.side for t in trades} <= {Side.BUY, Side.SELL}


def test_missing_block_gives_format_error():
    p = moex({"TQBR/securities.json": {"oops": 1}})
    with pytest.raises(DataFormatError):
        p.get_instruments()


def test_capabilities_and_license_declared():
    info = moex().info
    assert info.delayed and Capability.CANDLES in info.capabilities
    assert Capability.EXECUTION not in info.capabilities and "коммерческого" in info.license_note


# ── качество ──────────────────────────────────────────────────────────────
def _c(minute, o=100.0, h=101.0, l=99.0, c=100.5, v=10.0):
    t = datetime(2026, 10, 14, 7, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return Candle(t, t + timedelta(minutes=1), o, h, l, c, v)


def test_quality_detects_bad_data():
    bars = [_c(0), _c(1), _c(1), _c(2, h=98.0), _c(3, o=-1.0), _c(20)]
    q = check_candles(bars, Timeframe.M1, now=bars[-1].ts_close)
    assert {"duplicate_ts", "ohlc_inconsistent", "non_positive_price", "intraday_gap"} <= {i.code for i in q.issues}
    assert not q.ok and q.worst == "error"


def test_quality_empty_is_error():
    assert not check_candles([], Timeframe.H1).ok


def test_quality_stale_is_info_only():
    bars = [_c(i) for i in range(3)]
    q = check_candles(bars, Timeframe.M1, now=bars[-1].ts_close + timedelta(days=2))
    assert q.ok and any(i.code == "stale" for i in q.issues)


# ── мост к Kronos ─────────────────────────────────────────────────────────
def test_kronos_frame_contract():
    df = to_kronos_frame(moex().get_candles("SBER", "1h", START, NOW))
    assert list(df.columns) == ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    assert df["timestamps"].dt.tz is None and df["timestamps"].is_monotonic_increasing
    assert df["timestamps"].iloc[0].hour == 10 and not df.isna().any().any()


def test_kronos_bridge_refuses_bad_quality():
    s = moex().get_candles("SBER", "1h", START, NOW)
    dup = s.candles + (s.candles[-1],)
    broken = type(s)(s.instrument, s.timeframe, dup, s.source, s.fetched_at, check_candles(dup, s.timeframe, now=NOW))
    with pytest.raises(DataFormatError):
        to_kronos_frame(broken)


# ── реестр, конфиг, безопасность ──────────────────────────────────────────
def test_builtin_providers_registered():
    assert {"moex_iss", "csv_file"} <= set(available("market_data"))
    assert "disabled" in available("execution")


def test_unknown_provider():
    with pytest.raises(UnknownProviderError):
        create("market_data", "nope")


def test_bad_params_are_config_error():
    with pytest.raises(ConfigError):
        create("market_data", "moex_iss", no_such_param=1)


def test_dotted_path_plugin_without_registry_changes():
    hub = ProviderHub.from_config({
        "market_data": {"x": {"type": f"{__name__}:FakeProvider", "params": {"tag": "t"}}}})
    assert hub.market_data().info.name == "fake:t"


def test_hub_from_project_yaml():
    hub = ProviderHub.from_yaml(ROOT / "configs" / "providers.yaml")
    assert hub.profile == "bank_analytics" and hub.market_data().info.name == "moex_iss:shares:TQBR"
    assert hub.market_data("moex_index").board == "SNDX" and hub.execution.info.name == "disabled"
    hub.close()


def test_execution_stays_disabled_without_enabled_flag():
    hub = ProviderHub.from_config({
        "market_data": {"x": {"type": f"{__name__}:FakeProvider"}},
        "execution": {"type": f"{__name__}:FakeProvider", "enabled": False}})
    assert hub.execution.info.name == "disabled"
    order = OrderRequest("plan-1", "c1", InstrumentId("SBER", "MOEX"), Side.BUY, 1, limit_price=280.0)
    with pytest.raises(ExecutionDisabledError):
        hub.execution.submit(order)


def test_order_requires_plan_and_price():
    iid = InstrumentId("SBER", "MOEX")
    for args in (("", "c", iid, Side.BUY, 1), ("p", "c", iid, Side.BUY, 1), ("p", "c", iid, Side.BUY, 0)):
        kw = {"limit_price": 1.0} if args[0] == "" or args[4] == 0 else {}
        with pytest.raises(ValueError):
            OrderRequest(*args, **kw)


def test_unsupported_quotes_on_csv(tmp_path):
    with pytest.raises(UnsupportedCapability):
        create("market_data", "csv_file", directory=str(tmp_path)).get_quotes()


def test_csv_roundtrip(tmp_path):
    s = moex().get_candles("SBER", "1h", START, NOW)
    df = s.to_frame()
    df.index = df.index.tz_localize(None)
    df.index.name = "timestamp"
    df.to_csv(tmp_path / "SBER_1h.csv")
    p = create("market_data", "csv_file", directory=str(tmp_path), venue="MOEX")
    back = p.get_candles("SBER", "1h", START, NOW)
    assert [c.close for c in back.candles] == [c.close for c in s.candles]
    assert [c.ts_open for c in back.candles] == [c.ts_open for c in s.candles]
    assert [str(i.id) for i in p.get_instruments()] == ["SBER.MOEX"]


def test_instrument_id_parse():
    assert str(InstrumentId.parse("sber.moex")) == "SBER.MOEX"
    assert str(InstrumentId.parse("SBER", default_venue="moex")) == "SBER.MOEX"
    with pytest.raises(ValueError):
        InstrumentId.parse("SBER")


# ── реальные данные (по желанию) ──────────────────────────────────────────
@pytest.mark.skipif(not (FIX / "sber_candles_60.json").exists(), reason="запусти tools/record_moex_fixtures.py")
def test_recorded_fixtures():
    data = json.loads((FIX / "sber_candles_60.json").read_text(encoding="utf-8"))
    last = data["candles"]["data"][-1][data["candles"]["columns"].index("end")]
    now = datetime.strptime(last, "%Y-%m-%d %H:%M:%S").replace(tzinfo=MSK) + timedelta(minutes=1)
    p = MoexISSProvider(transport=Replay({"SBER/candles.json": data}), now=lambda: now)
    s = p.get_candles("SBER", "1h", datetime(2026, 10, 1, tzinfo=MSK), now)
    assert len(s) > 0 and s.quality.ok
    assert all(c.low <= min(c.open, c.close) and c.high >= max(c.open, c.close) for c in s.candles)


@pytest.mark.skipif(os.environ.get("FINVEX_LIVE") != "1", reason="живой ISS: FINVEX_LIVE=1")
def test_live_moex_smoke():
    with create("market_data", "moex_iss") as p:
        end = datetime.now(timezone.utc)
        s = p.get_candles("SBER", "1h", end - timedelta(days=7), end)
        assert len(s) > 20 and s.quality.ok
        assert to_kronos_frame(s).shape[1] == 7


class FakeProvider(MarketDataProvider):
    def __init__(self, tag: str = ""):
        self._info = ProviderInfo(name=f"fake:{tag}", venue="FAKE", capabilities=Capability.INSTRUMENTS)

    @property
    def info(self):
        return self._info

    def get_instruments(self):
        return []

    def get_candles(self, *a, **kw):
        raise UnsupportedCapability("fake")
