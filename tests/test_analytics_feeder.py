"""Подающий процесс: из результатов Kronos и разбора новостей в хранилище шлюза. Источники подставные, правило: оценок не выдумываем."""
import math
import time
from datetime import datetime, timedelta

import pytest

from finvex.core.models import MSK
from finvex.feeds.analytics_feeder import DEFAULT_TTL_H, AnalyticsFeeder, kronos_score, kronos_valid_until, map_category, start_background
from tests.effectiveness.harness import Lab, Scenario

NOW = datetime(2026, 10, 26, 12, 20, tzinfo=MSK)
KR_BEAR = {"status": "ok", "change": -0.03, "volatility_norm": 0.01}


def buy(lab, sym="SBER"):
    """Заявка без сброса аналитики (Lab.decide очищает хранилище и подставляет значения из сценария)."""
    lab.sc, lab.clock.t = Scenario(symbol=sym, qty=5, price=300.0, when=NOW), NOW
    ag, _ = lab.gw.agents.create(f"f{len(lab.gw.agents.list())}", daily_quota=10**6)
    return lab.gw.propose(ag, dict(client_id="c1", symbol=sym, side="buy", quantity=5, confidence=0.7, rationale="Проверка подачи аналитики в шлюз."))


def mk():
    lab = Lab("analytics")
    lab.clock.t = NOW
    return lab, AnalyticsFeeder(lab.gw.analytics, clock=lambda: NOW, tf="1h")


def test_kronos_score_formula_and_validity_until_next_candle_close():
    assert kronos_score(KR_BEAR) == pytest.approx(0.7 * math.tanh(-0.03 / 0.02))
    assert kronos_valid_until(NOW, "1h") == datetime(2026, 10, 26, 13, 0, tzinfo=MSK)
    assert kronos_valid_until(NOW, "4h") == datetime(2026, 10, 26, 15, 0, tzinfo=MSK)    # границы 4-часовых свечей по UTC
    assert kronos_valid_until(NOW, "1d") == datetime(2026, 10, 27, 0, 0, tzinfo=MSK)


@pytest.mark.parametrize("kr", [{"status": "mock", "error": "модель не загружена"}, {"status": "error", "error": "x"}, {}, None,
                                {"status": "ok", "change": float("nan"), "volatility_norm": 0.01}, {"status": "ok"}])
def test_kronos_without_a_real_forecast_writes_nothing(kr):
    lab, fd = mk()
    assert fd.push_kronos("SBER", kr) is None and lab.gw.analytics.snapshot("SBER", NOW) == {}


def test_bearish_kronos_from_feeder_blocks_a_buy_in_the_gateway():
    lab, fd = mk()
    fd.push_kronos("SBER", KR_BEAR)
    v = buy(lab)
    assert v["decision"] == "deny" and "kronos_view" in v["codes"]


def test_unscored_or_expired_news_are_not_written_and_ttl_source_is_marked():
    lab, fd = mk()
    assert fd.push_news("SBER", {"title": "без оценки модели", "time": "2026-10-26 12:00"}) is None
    assert fd.push_news("SBER", {"title": "старый слух", "time": "2026-10-26 07:00", "sentiment": -0.8, "category": "rumor"}) is None   # слух живёт 2 ч
    row = fd.push_news("SBER", {"title": "иск к компании", "time": "2026-10-26 12:00", "sentiment": -0.7, "materiality": 0.9, "category": "company risk"})
    assert row and row["ttl_hours"] == DEFAULT_TTL_H["company"] and "[ttl: default]" in row["title"]
    parsed = fd.push_news("GAZP", {"title": "решение регулятора", "time": "2026-10-26 12:10", "sentiment": -0.6, "ttl_hours": 3, "category": "макро"})
    assert parsed["ttl_hours"] == 3.0 and "[ttl: parser]" in parsed["title"] and parsed["category"] == "macro"


def test_negative_news_from_feeder_blocks_buy_is_idempotent_and_expires():
    lab, fd = mk()
    row = {"title": "санкции против компании", "time": "2026-10-26 12:00", "sentiment": -0.9, "category": "macro"}
    fd.push_news("SBER", row)
    fd.push_news("SBER", row)
    assert lab.gw.analytics.snapshot("SBER", NOW)["news_n"] == 1
    assert buy(lab)["decision"] == "deny"
    assert "news_score" not in lab.gw.analytics.snapshot("SBER", NOW + timedelta(hours=30))


def test_tick_collects_source_errors_and_continues():
    lab = Lab("analytics")

    def kr(sym, tf):
        if sym == "BAD":
            raise RuntimeError("сеть недоступна")
        return KR_BEAR if sym == "SBER" else {"status": "mock", "error": "заглушка"}

    news = lambda s: [{"title": "t", "time": "2026-10-26 12:00", "sentiment": 0.4, "category": "company"}] if s == "SBER" else []   # noqa: E731
    rep = AnalyticsFeeder(lab.gw.analytics, clock=lambda: NOW, kronos_source=kr, news_source=news).tick(["BAD", "SBER", "GAZP"])
    assert rep["kronos"] == 1 and rep["news"] == 1 and len(rep["errors"]) == 1 and len(rep["skipped"]) == 1


def test_category_mapping():
    assert [map_category(x) for x in ("Дивиденды и отчётность", "ставка ЦБ", "слух", "company", None)] == ["earnings", "macro", "rumor", "company", "other"]


def test_background_runner_ticks_and_stops():
    lab, lines = Lab("analytics"), []
    stop = start_background(lab.gw.analytics, ["SBER"], interval_s=0.05, say=lines.append, kronos_source=lambda s, tf: {"status": "mock"}, news_source=lambda s: [])
    time.sleep(0.3)
    stop.set()
    assert len(lines) >= 2 and "[feeder]" in lines[0]
