from __future__ import annotations
import argparse
import logging
from pathlib import Path

import yaml

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ── backtest ──────────────────────────────────────────────────────────

def _cmd_backtest(args: argparse.Namespace) -> None:
    from src.engine.config_loader import load_engine_config, merge_llm_section
    from src.engine.backtest_engine import BacktestEngine
    from src.signals.data_collector import DataCollector
    from src.kronos_layer.kronos_adapter import KronosAdapter
    from src.signals.ai_tactic_strategy import AITacticStrategy, AITacticConfig
    import numpy as np
    from src.utils.math_utils import sharpe, max_drawdown, cagr

    cfg = load_engine_config(args.config)
    signal_cfg = cfg.signal

    kronos_cfg_path = Path("configs/kronos.yaml")
    kronos_cfg = yaml.safe_load(kronos_cfg_path.read_text()) if kronos_cfg_path.exists() else {}
    kronos = KronosAdapter(kronos_cfg) if kronos_cfg else None

    collector = DataCollector(historical_dir="data/raw/historical")
    ohlcv = {}
    for ticker in cfg.tickers:
        try:
            ohlcv[ticker] = collector.load_historical(ticker)
        except FileNotFoundError:
            logger.warning("No OHLCV data for %s", ticker)

    strategist = None
    ls_cfg_path = Path("configs/llm_strategist.yaml")
    if ls_cfg_path.exists():
        ls_cfg = yaml.safe_load(ls_cfg_path.read_text(encoding="utf-8")).get("llm_strategist", {})
        ls_cfg = merge_llm_section(ls_cfg)
        if ls_cfg.get("enabled", False):
            from src.llm_strategist.strategist import LLMStrategist
            strategist = LLMStrategist(ls_cfg)
            logger.info("Backtest | LLMStrategist configured | model=%s", ls_cfg.get("model"))

    tactic_cfg = AITacticConfig(
        min_conviction=getattr(signal_cfg, "min_conviction", 0.35),
        qty_per_trade=getattr(signal_cfg, "qty_per_trade", 100.0),
        max_positions=getattr(signal_cfg, "max_positions", 10),
        initial_tickers=tuple(cfg.tickers),
    )
    strategy = AITacticStrategy(cfg=tactic_cfg, kronos=kronos, ohlcv=ohlcv)

    # Фикс К7: BacktestEngine.__init__ принимает только (cfg, strategist,
    # news_by_ticker_provider) — strategist_bar_interval/max_llm_calls
    # не существуют как параметры конструктора. Throttling частоты вызовов
    # стратега — ответственность самого LLMStrategist/AITacticStrategy,
    # а не BacktestEngine, поэтому эти опции здесь просто убраны.
    engine = BacktestEngine(cfg, strategist=strategist)
    engine.register_strategy(strategy)
    _j = getattr(getattr(engine, "_portfolio", None), "_journal", None)
    if _j is not None:
        strategy.set_journal(_j)
    state = engine.run()

    equity = np.array([v for _, v in state.pnl_history])
    returns = np.diff(equity) / equity[:-1]

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS (AITacticStrategy + LLMStrategist)")
    print("=" * 50)
    print(f"Final equity: {state.portfolio_value:>12,.2f} RUB")
    print(f"Total PnL: {state.total_pnl:>12,.2f} RUB")
    print(f"Total trades: {len(state.trades):>12d}")
    print(f"Open positions: {len(state.positions):>12d}")
    # Фикс К7: engine._llm_calls_made не существует в BacktestEngine.
    # Счётчик вызовов стратега доступен только на самом объекте strategist.
    if strategist is not None:
        llm_calls_made = getattr(strategist, "calls_made", None)
        if llm_calls_made is not None:
            print(f"LLM calls made: {llm_calls_made:>12d}")
        print(f"Tokens in/out: {strategist.total_tokens_in:>6d} / {strategist.total_tokens_out:<6d}")
    if len(returns) > 1:
        print(f"Sharpe ratio: {sharpe(returns):>12.3f}")
        print(f"Max drawdown: {max_drawdown(equity):>12.2%}")
        print(f"CAGR: {cagr(equity):>12.2%}")
    print("=" * 50)


# ── live ──────────────────────────────────────────────────────────────

def _cmd_live(args: argparse.Namespace) -> None:
    from src.engine.config_loader import load_engine_config, merge_llm_section, load_tinkoff_config
    from src.engine.live_engine import LiveEngine
    from src.kronos_layer.kronos_adapter import KronosAdapter
    from src.signals.ai_tactic_strategy import AITacticStrategy, AITacticConfig
    from src.signals.universe_selector import UniverseSelector
    from src.news_agent_client.client import NewsAgentClient
    from src.signals.news_signal import NewsSignalExtractor
    from src.executors.tinkoff_client import TinkoffClient

    cfg = load_engine_config(args.config)
    tinkoff_cfg = load_tinkoff_config(sandbox=(args.mode == "sandbox"))
    token = tinkoff_cfg.token

    signal_cfg = cfg.signal

    kronos_cfg_path = Path("configs/kronos.yaml")
    kronos_cfg = yaml.safe_load(kronos_cfg_path.read_text()) if kronos_cfg_path.exists() else {}
    kronos = KronosAdapter(kronos_cfg) if kronos_cfg else None

    # ── News ─────────────────────────────────────────────────────
    news_extractor = None
    news_client = None
    news_pipeline = None

    news_cfg_path = Path("configs/news_agent.yaml")
    if news_cfg_path.exists():
        na_cfg = yaml.safe_load(news_cfg_path.read_text(encoding="utf-8")) or {}

        if "news_agent" in na_cfg and "llm" in na_cfg["news_agent"]:
            na_cfg["news_agent"]["llm"] = merge_llm_section(
                na_cfg["news_agent"]["llm"]
            )

        news_client = NewsAgentClient(na_cfg)
        news_extractor = NewsSignalExtractor(
            client=news_client,
            hours_back=getattr(signal_cfg, "news_hours_back", 6),
        )

        from src.news_agent_client.rss_collector import RSSCollectorConfig
        from src.news_agent_client.news_deep_dive import NewsDeepDiveAnalyzer
        from src.news_agent_client.news_risk_pipeline import NewsRiskPipeline

        risk_cfg = na_cfg.get("news_risk", {})
        rss_cfg = RSSCollectorConfig(
            poll_interval_sec=int(risk_cfg.get("poll_interval_sec", 30)),
        )
        analyzer = NewsDeepDiveAnalyzer(
            cfg=merge_llm_section(risk_cfg.get("llm", {})),
        )

        # LiveEngine перезапишет on_assessment своим thread-safe callback
        # в __init__; placeholder нужен лишь для контракта конструктора.
        news_pipeline = NewsRiskPipeline(
            rss_cfg=rss_cfg,
            analyzer=analyzer,
            on_assessment=lambda ticker, assessment: None,
            filter_mode=risk_cfg.get("filter_mode", "keyword"),
            ttl_days=risk_cfg.get("ttl_days", {"short": 3, "long": 14}),
        )

        # Seed watch_terms через AliasGenerator (DeepSeek) — как KronosDefaultStrategy.seed_watch_terms
        try:
            from src.news_agent_client.ticker_intelligence import AliasGenerator
            _alias_gen = AliasGenerator(merge_llm_section(risk_cfg.get("llm", {})))
            _generated = _alias_gen.generate(["BTCUSDT", "ETHUSDT"])
            _watch_terms = {t: list({t, *terms})[:8] for t, terms in _generated.items()}
            news_pipeline.update_watch_terms(_watch_terms, "startup")
            logger.info("Watch terms seeded | tickers=%d | %s",
                        len(_watch_terms), {t: len(v) for t, v in _watch_terms.items()})
        except Exception as exc:
            logger.warning("AliasGenerator seed failed: %s — контур стартует пустым", exc)

        logger.info(
            "News pipeline configured | filter=%s | poll=%ss",
            risk_cfg.get("filter_mode", "keyword"),
            rss_cfg.poll_interval_sec,
        )
    # ── UniverseSelector ───────────────────────────────────────────
    # Crypto-only sandbox: не создаём TinkoffClient и не выполняем
    # сетевой запрос UniverseSelector с US VM.
    universe_selector = None
    initial_universe = None
    logger.info(
        "UniverseSelector skipped | crypto-only Bybit sandbox"
    )

    # ── AITacticStrategy ────────────────────────────────────────────

    _sig = (lambda k, d: signal_cfg.get(k, d)) if isinstance(signal_cfg, dict) else (lambda k, d: getattr(signal_cfg, k, d))
    _tac = yaml.safe_load(Path("configs/engine.yaml").read_text(encoding="utf-8")).get("tactic", {})
    tactic_cfg = AITacticConfig(
        min_conviction=float(_tac.get("min_conviction", 0.35)),
        qty_per_trade=float(_tac.get("qty_per_trade", 100.0)),
        max_positions=int(_tac.get("max_positions", 10)),
    )
    logger.info("Tactic cfg | qty=%s min_conv=%s", _tac.get("qty_per_trade"), _tac.get("min_conviction"))
    # OHLCV warmup: дневные klines Bybit (публичный эндпоинт) — кормит regime-слой и Kronos
    ohlcv = {}
    try:
        import requests
        import pandas as pd
        for _t in ["BTCUSDT", "ETHUSDT"]:
            _r = requests.get(
                "https://api-testnet.bybit.com/v5/market/kline",
                params={"category": "spot", "symbol": _t, "interval": "D", "limit": 200},
                timeout=15,
            )
            _rows = list(reversed(_r.json()["result"]["list"]))
            ohlcv[_t] = pd.DataFrame({
                "date":   pd.to_datetime([int(x[0]) for x in _rows], unit="ms"),
                "open":   [float(x[1]) for x in _rows],
                "high":   [float(x[2]) for x in _rows],
                "low":    [float(x[3]) for x in _rows],
                "close":  [float(x[4]) for x in _rows],
                "volume": [float(x[5]) for x in _rows],
            }).set_index("date")
        logger.info("OHLCV warmup | %s", {t: len(df) for t, df in ohlcv.items()})
    except Exception as exc:
        logger.warning("OHLCV warmup failed: %s — regime/Kronos спят", exc)
    strategy = AITacticStrategy(
        cfg=tactic_cfg,
        kronos=kronos,
        ohlcv=ohlcv,
    )
    strategy.set_universe(["BTCUSDT", "ETHUSDT"])  # crypto-only soak
    # прогрев _regime_cache до первого стратег-полла
    for _t in list(strategy.tickers):
        try:
            strategy._regime_for(_t)
        except Exception:
            pass
    logger.info("Regime cache primed | %s",
                {t: getattr(v[1], "regime", "?") for t, v in strategy._regime_cache.items()})

    # ── LLMStrategist (DeepSeek V4) ─────────────────────────────────
    strategist = None
    strategist_interval = 900
    ls_cfg_path = Path("configs/llm_strategist.yaml")
    if ls_cfg_path.exists():
        ls_cfg = yaml.safe_load(ls_cfg_path.read_text(encoding="utf-8")).get("llm_strategist", {})
        ls_cfg = merge_llm_section(ls_cfg)
        if ls_cfg.get("enabled", False):
            from src.llm_strategist.strategist import LLMStrategist
            strategist = LLMStrategist(ls_cfg)
            strategist_interval = int(ls_cfg.get("poll_interval", 900))
            logger.info("LLMStrategist configured | model=%s", ls_cfg.get("model"))

    # Фикс К8: параметр называется news_pipeline, не news_extractor.
    engine = LiveEngine(
        cfg=cfg,
        mode=args.mode,
        token=None,  # crypto-only: tinkoff router skipped (geo-block с US-IP)
        enable_bybit=True,
        bybit_category="linear",
        news_pipeline=news_pipeline,
        strategist=strategist,
        strategist_poll_interval=strategist_interval,
        universe_selector=None,  # не дёргать Tinkoff каждые 900с из strategist-loop
    )
    engine.register_strategy(strategy)
    strategy.set_journal(engine._portfolio._journal)
    engine.run()


# ── news ──────────────────────────────────────────────────────────────

def _cmd_news(args: argparse.Namespace) -> None:
    from src.news_agent_client.client import NewsAgentClient

    na_cfg_path = Path("configs/news_agent.yaml")
    if not na_cfg_path.exists():
        raise FileNotFoundError("configs/news_agent.yaml не найден")

    na_cfg = yaml.safe_load(na_cfg_path.read_text(encoding="utf-8"))
    client = NewsAgentClient(na_cfg)

    docs = client.search_news(args.ticker, hours_back=args.hours)
    if not docs:
        print(f"No news for {args.ticker} in last {args.hours}h")
        return

    print(f"\nNews for {args.ticker} (last {args.hours}h): {len(docs)} items\n")
    for d in docs:
        bar = "█" * int(max(0, d.sentiment + 1) * 5)
        print(f"  [{d.published_at}] {d.source}")
        print(f"  {d.title}")
        print(f"  sentiment={d.sentiment:+.2f} {bar}")
        print()

    if args.summarize:
        summary = client.summarize_events(args.ticker, hours_back=args.hours)
        print("─" * 50)
        print("SUMMARY:")
        print(summary)


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="trading_agent",
        description="Kronos + AITacticStrategy + LLM Strategist + News trading system",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bt = sub.add_parser("backtest", help="Исторический бэктест")
    p_bt.add_argument("--config", default="configs/engine.yaml")

    p_lv = sub.add_parser("live", help="Live/sandbox торговля")
    p_lv.add_argument("--config", default="configs/engine.yaml")
    p_lv.add_argument("--mode", choices=["sandbox", "live"], default="sandbox")

    p_nw = sub.add_parser("news", help="Новости по тикеру")
    p_nw.add_argument("--ticker", required=True)
    p_nw.add_argument("--hours", type=int, default=24)
    p_nw.add_argument("--summarize", action="store_true", help="LLM-саммари")

    args = parser.parse_args()
    dispatch = {
        "backtest": _cmd_backtest,
        "live": _cmd_live,
        "news": _cmd_news,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()