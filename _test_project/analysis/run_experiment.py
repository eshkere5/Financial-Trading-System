"""
run_experiment.py — длительный тест полной системы с журналом в experiment.db.

### NEW 2026-08-19 ###
Принцип: тест подстраивается под систему, а не наоборот.
НИ ОДИН файл src/ не меняется: журнал навешивается снаружи обёртками
(monkey-patch) на методы движка/стратегии — боевой код остаётся чистым,
а эксперимент можно удалить одним файлом.

Запуск:
    $env:EXPERIMENT_RUN_ID = "D_full"        # метка прогона (для абляции)
    $env:EXPERIMENT_HOURS  = "72"            # длительность
    $env:SOAK_SYMBOL       = "BTCUSDT"
    python run_experiment.py

Просмотр в любой момент (второй терминал):
    python watch_experiment.py
    python watch_experiment.py --tail 20
    python watch_experiment.py --export equity.csv
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("run_experiment")

RUN_ID = os.environ.get("EXPERIMENT_RUN_ID", "full")
HOURS = float(os.environ.get("EXPERIMENT_HOURS", "72"))
SYMBOLS = tuple(
    s.strip().upper()
    for s in os.environ.get("EXPERIMENT_SYMBOLS", "BTCUSDT").split(",")
    if s.strip()
)
QTY = float(os.environ.get("SOAK_QTY", "0.002"))


# ── сборка системы (как в soak, фаза 2: полный контур) ───────────────

def _build_news_pipeline():
    import yaml
    from src.engine.config_loader import load_llm_shared, merge_llm_section
    from src.news_agent_client.news_deep_dive import NewsDeepDiveAnalyzer
    from src.news_agent_client.news_risk_pipeline import NewsRiskPipeline
    from src.news_agent_client.rss_collector import RSSCollectorConfig

    news_raw = yaml.safe_load(Path("configs/news_agent.yaml").read_text(encoding="utf-8")) or {}
    rss_cfg = RSSCollectorConfig(**(news_raw.get("rss") or {}))
    ds_path = Path("configs/deepseek_news.yaml")
    ds_raw = yaml.safe_load(ds_path.read_text(encoding="utf-8")) or {} if ds_path.exists() else {}
    llm_cfg = merge_llm_section(ds_raw.get("llm", {}), shared=load_llm_shared())
    pipeline = NewsRiskPipeline(
        rss_cfg=rss_cfg,
        analyzer=NewsDeepDiveAnalyzer(llm_cfg),
        on_assessment=lambda ticker, a: None,  # engine перезапишет в __init__
        filter_mode="keyword",
    )
    TERMS = {
        "BTCUSDT": ["BTCUSDT", "BTC", "Bitcoin", "биткоин", "криптовалюта"],
        "ETHUSDT": ["ETHUSDT", "ETH", "Ethereum", "эфириум", "эфир", "Vitalik"],
    }
    pipeline.update_watch_terms(
        {s: TERMS.get(s, [s]) for s in SYMBOLS},
        "experiment@startup",
    )
    return pipeline


def _build_strategist():
    import yaml
    from src.engine.config_loader import load_llm_shared, merge_llm_section

    lscfg = yaml.safe_load(Path("configs/llm_strategist.yaml").read_text(encoding="utf-8")).get("llm_strategist", {})
    lscfg = merge_llm_section(lscfg, load_llm_shared())
    if not lscfg.get("enabled", False):
        return None
    from src.llm_strategist.strategist import LLMStrategist
    return LLMStrategist(lscfg)


# ── обвязка журналом СНАРУЖИ (без правок src/) ───────────────────────

def _attach_experiment(engine, strategy, strategist, exp) -> None:
    """Оборачивает методы живых объектов: после каждого вызова дописывает
    событие в experiment.db. Система о себе ничего не знает."""

    # 1) equity каждый тик
    orig_tick = engine._tick

    async def tick_logged():
        await orig_tick()
        exp.snapshot_equity(
            equity=engine._state.portfolio_value,
            cash=engine._state.cash,
            n_positions=len(engine._state.positions),
            pnl=engine._state.total_pnl,
        )
    engine._tick = tick_logged

    # 2) новостные события
    orig_news = engine._on_news_assessment

    def news_logged(ticker, assessment):
        orig_news(ticker, assessment)
        exp.log_event(
            "news", ticker, getattr(assessment, "action", "pass"),
            payload={"category": getattr(assessment, "risk_category", "none"),
                     "risk": float(getattr(assessment, "effective_risk", 0.0) or 0.0),
                     "horizon": getattr(assessment, "horizon", "short")},
            rationale=str(getattr(assessment, "source_title", ""))[:120],
        )
    engine._on_news_assessment = news_logged

    # 3) решения стратегиста
    if strategist is not None:
        orig_decide = strategist.decide

        def decide_logged(context):
            d = orig_decide(context)
            exp.log_event(
                "strategist", "*", "bias" if getattr(d, "tactical_bias", None) else "hold",
                payload={"regime": getattr(d, "regime", None),
                         "posture": getattr(d, "risk_posture", None),
                         "size": getattr(d, "size_multiplier", 1.0),
                         "bias": getattr(d, "tactical_bias", None)},
                rationale=str(getattr(d, "rationale", ""))[:500],
            )
            return d
        strategist.decide = decide_logged

    # 4) атрибуция: что сказал каждый контур, когда стратегия родила ордер
    orig_on_bar = strategy.on_bar

    def on_bar_logged(state):
        try:
            held_before = {
                t for t, p in (state.positions or {}).items()
                if getattr(p, "qty", 0) != 0
            }
        except Exception:
            held_before = set()
        orders = orig_on_bar(state)
        for o in orders:
            try:                             # журнал не может уронить тик
                kstate = getattr(strategy, "last_kronos_states", {}).get(o.ticker)
                news = state.news.get(o.ticker)
                bias = {}
                if getattr(engine, "_decision", None) is not None:
                    bias = (engine._decision.tactical_bias or {}).get(o.ticker, {})
                exp.log_attribution(
                    o.ticker,
                    kronos_dir=int(getattr(kstate, "direction", 0) or 0),
                    kronos_conf=float(getattr(kstate, "confidence", 0.0) or 0.0),
                    news_risk=float(getattr(news, "effective_risk", 0.0) or 0.0) if news else 0.0,
                    news_action=str(getattr(news, "risk_category", "none")) if news else "none",
                    strat_dir=int(bias.get("direction", 0)),
                    strat_conv=float(bias.get("conviction", 0.0)),
                    size_mult=float(getattr(engine._decision, "size_multiplier", 1.0)) if engine._decision else 1.0,
                    final_action="close" if o.ticker in held_before else "open",
                    final_qty=float(o.qty),
                )
            except Exception as e:
                log.warning("attribution log failed (некритично): %s", e)
        return orders
    strategy.on_bar = on_bar_logged

    # 5) исход отправки ордера (rejected / below_min / filled)
    orig_submit = engine._submit_and_track

    async def submit_logged(order):
        ok = await orig_submit(order)
        if not ok:
            exp.log_event("risk", order.ticker, "rejected",
                          payload={"qty": float(order.qty), "side": str(order.side)})
        return ok
    engine._submit_and_track = submit_logged


# ── main ─────────────────────────────────────────────────────────────

async def main() -> None:
    from src.engine.config_loader import (
        load_bybit_config, load_engine_config, load_kronos_config,
    )
    from src.engine.experiment_logger import ExperimentLogger
    from src.engine.live_engine import LiveEngine
    from src.executors.bybit_client import BybitClient
    from src.kronos_layer.kronos_adapter import KronosAdapter
    from src.signals.ai_tactic_strategy import AITacticConfig, AITacticStrategy

    exp = ExperimentLogger(run_id=RUN_ID)

    kronos = KronosAdapter(load_kronos_config("configs/kronos.yaml"))
    assert not kronos.is_mock, "Kronos в MOCK-режиме — эксперимент бессмысленен"

    bybit_client = BybitClient(load_bybit_config(testnet=True))
    ohlcv = {}
    for sym in SYMBOLS:
        candles = bybit_client.get_candles(sym, interval="1h", limit=240)
        assert len(candles) >= 32, f"мало свечей по {sym}"
        ohlcv[sym] = candles

    news_pipeline = _build_news_pipeline()
    strategist = _build_strategist()
    assert strategist is not None, "Длительный тест идёт со стратегистом — проверь llm_strategist.yaml"

    strategy = AITacticStrategy(
        cfg=AITacticConfig(
            min_conviction=0.35,
            risk_per_trade_pct=float(os.environ.get("EXPERIMENT_RISK_PCT", "0.05")),  # 5% equity на сделку
            max_positions=len(SYMBOLS),
            initial_tickers=SYMBOLS,
        ),
        kronos=kronos,
        ohlcv=ohlcv,
    )

    engine = LiveEngine(
        cfg=load_engine_config("configs/engine.yaml"),
        mode="sandbox",
        token="",
        news_pipeline=news_pipeline,
        strategist=strategist,
        enable_bybit=True,
        bybit_category="linear",
        price_poll_interval=60,
        strategist_poll_interval=int(os.environ.get("SOAK_STRATEGIST_SEC", "300")),
    )
    engine.register_strategy(strategy)

    _attach_experiment(engine, strategy, strategist, exp)

    log.info("=" * 60)
    log.info("EXPERIMENT START | run_id=%s | %s | %.1f ч | qty=%.4f", RUN_ID, SYMBOLS, HOURS, QTY)
    log.info("Просмотр: python watch_experiment.py [--tail N] [--export csv]")
    log.info("=" * 60)

    task = asyncio.create_task(engine._run_async())
    try:
        await asyncio.sleep(HOURS * 3600)
    except asyncio.CancelledError:
        pass
    finally:
        engine._shutdown()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        news_pipeline.stop()
        exp.close()

    log.info("EXPERIMENT END | run_id=%s | final PnL=%.4f", RUN_ID, engine._state.total_pnl)
    log.info("Итоги: python watch_experiment.py --run %s", RUN_ID)


if __name__ == "__main__":
    asyncio.run(main())
