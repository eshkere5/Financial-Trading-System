from __future__ import annotations

import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pytest

import logging

logging.basicConfig(level=logging.INFO)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_REAL_E2E = os.getenv("RUN_REAL_E2E") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_REAL_E2E,
    reason=(
        "Реальный E2E выключен. Запуск: "
        "$env:RUN_REAL_E2E='1'; pytest -q -s tests/test_e2e_news_real.py"
    ),
)


def _as_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, dict):
        return dict(value)

    return {
        name: getattr(value, name)
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name))
    }


def _load_news_components():
    import yaml

    from src.engine.config_loader import load_llm_shared, merge_llm_section
    from src.news_agent_client.news_deep_dive import NewsDeepDiveAnalyzer
    from src.news_agent_client.rss_collector import RSSCollector, RSSCollectorConfig

    shared = load_llm_shared("configs/llm_shared.yaml")

    # RSS-конфиг — из news_agent.yaml (секция rss)
    news_raw = yaml.safe_load(
        (ROOT / "configs" / "news_agent.yaml").read_text(encoding="utf-8")
    ) or {}
    rss_cfg = RSSCollectorConfig(**(news_raw.get("rss") or {}))

    # LLM-конфиг анализатора — из deepseek_news.yaml (секция llm)
    ds_path = ROOT / "configs" / "deepseek_news.yaml"
    if ds_path.exists():
        ds_raw = yaml.safe_load(ds_path.read_text(encoding="utf-8")) or {}
        local_llm = ds_raw.get("llm", {})
    else:
        # fallback: секция из news_agent.yaml
        news_agent_raw = news_raw.get("news_agent", {})
        local_llm = news_agent_raw.get("llm", {})

    llm_cfg = merge_llm_section(local_llm, shared=shared)

    # Нормализуем имя ключа: analyzer ждёт max_tokens (или свой дефолт),
    # а в deepseek_news.yaml ключ называется max_tokens_analyze
    if "max_tokens_analyze" in llm_cfg:
        llm_cfg["max_tokens"] = llm_cfg["max_tokens_analyze"]

    llm_cfg.setdefault("max_tokens", 1500)

    if not llm_cfg.get("api_key"):
        raise RuntimeError(
            "DeepSeek api_key не найден в configs/llm_shared.yaml "
            "или переменной DEEPSEEK_API_KEY."
        )

    print(f"[DEBUG] analyzer max_tokens: {llm_cfg.get('max_tokens')}")

    collector = RSSCollector(rss_cfg)
    print(f"[DEBUG] llm_cfg keys: {list(llm_cfg.keys())}")
    print(f"[DEBUG] llm_cfg max_tokens_analyze: {llm_cfg.get('max_tokens_analyze')}")
    print(f"[DEBUG] llm_cfg max_tokens: {llm_cfg.get('max_tokens')}")
    analyzer = NewsDeepDiveAnalyzer(llm_cfg)

    
    return collector, analyzer


def _print_assessment(label: str, ticker: str, assessment: Any) -> None:
    print("\n" + "=" * 88)
    print(f"[E2E][NEWS][{label}]")
    print(f"ticker              : {ticker}")
    print(f"risk_category       : {assessment.risk_category}")
    print(f"severity            : {assessment.severity}")
    print(f"materiality         : {assessment.materiality}")
    print(f"effective_risk      : {assessment.effective_risk}")
    print(f"sentiment           : {assessment.sentiment}")
    print(f"horizon             : {assessment.horizon}")
    print(f"action              : {assessment.action}")
    print(f"ttl_days            : {assessment.ttl_days}")
    print(f"rationale           : {assessment.rationale}")
    print("=" * 88)

def _build_real_market_state(symbol: str, price: float):
    from src.engine.market_state import MarketState
    from src.utils.time_utils import utcnow

    state = MarketState(cash=100_000.0, timestamp=utcnow())
    state.prices[symbol] = price
    return state


def test_real_rss_deepseek_news_changes_tactic_decision() -> None:
    """
    Реальная цепочка новостей:
      RSS -> реальная свежая публикация -> DeepSeek NewsDeepDiveAnalyzer ->
      NewsSnapshot -> AITacticStrategy.

    Тест не подделывает RSS, LLM или NewsSnapshot.
    Он ищет реальную новость, связанную с тикером, и доказывает,
    что она попадает в MarketState и участвует в вычислении позиции.
    """
    from src.engine.live_engine import LiveEngine
    from src.engine.market_state import NewsSnapshot
    from src.engine.orders import OrderSide
    from src.engine.config_loader import load_engine_config
    from src.news_agent_client.keyword_filter import build_filter
    from src.news_agent_client.news_risk_pipeline import NewsRiskPipeline
    from src.signals.ai_tactic_strategy import AITacticConfig, AITacticStrategy

    symbol = os.getenv("E2E_NEWS_SYMBOL", "BTCUSDT").upper()
    last_price = float(os.getenv("E2E_NEWS_PRICE", "100000.0"))
    max_backlog_days = int(os.getenv("E2E_NEWS_BACKLOG_DAYS", "3"))

    collector, analyzer = _load_news_components()

    # 1. Реальное получение RSS backlog.
    items = collector.fetch_backlog(max_backlog_days)
    from collections import Counter

    sources = Counter(item.source for item in items)
    print("\n[E2E][RSS SOURCES]")
    for name, count in sources.most_common():
        print(f"  {name}: {count}")

    for item in items:
        text = f"{item.title} {getattr(item, 'summary', '')}".lower()
        if any(w in text for w in ("bitcoin", "btc", "биткоин")):
            print(f"[HIT] {item.source}: {item.title[:120]}")
    assert items, (
        "RSS collector не вернул ни одной новости. "
        "Проверь configs/news_agent.yaml, сеть и источники."
    )

    print("\n" + "=" * 88)
    print("[E2E][RSS]")
    print(f"backlog_days        : {max_backlog_days}")
    print(f"items_received      : {len(items)}")
    print("=" * 88)

    # 2. Ставим реальные watch terms и выбираем реально совпавшую новость.
    watch_terms = {
        symbol: [
            symbol,
            "BTC",
            "Bitcoin",
            "биткоин",
            "криптовалюта",
            "крипторынок",
        ]
    }

    from src.news_agent_client.keyword_filter import WatchTermsStore, build_filter

    store = WatchTermsStore()
    store.update(watch_terms, "e2e-test")
    cheap_filter = build_filter(store, mode="keyword")

    selected_item = None
    for item in items:
        matched = cheap_filter.match(item.title, item.summary)
        if symbol in matched:
            selected_item = item
            break

    if selected_item is None:
        pytest.skip(
            f"За {max_backlog_days} дней RSS не дал новости для {symbol}. "
            "Это не ошибка системы: увеличь E2E_NEWS_BACKLOG_DAYS."
        )

    print("\n" + "=" * 88)
    print("[E2E][RSS MATCH]")
    print(f"source              : {selected_item.source}")
    print(f"published_at        : {selected_item.published_at}")
    print(f"title               : {selected_item.title}")
    print(f"link                : {selected_item.link}")
    print("=" * 88)

    # 3. Реальный DeepSeek-анализ реальной RSS-публикации.
    assessment = analyzer.analyze(
        ticker=symbol,
        item=selected_item,
    )

    assert assessment is not None
    assert not getattr(assessment, "is_fallback", False), (
        "NewsDeepDiveAnalyzer ушёл в fallback. "
        f"assessment={assessment!r}"
    )

    _print_assessment("REAL_RSS", symbol, assessment)

    # 4. Реальный LiveEngine callback: assessment -> NewsSnapshot.
    engine_cfg = load_engine_config("configs/engine.yaml")
    engine = LiveEngine(
        cfg=engine_cfg,
        mode="sandbox",
        token="",
        enable_bybit=False,
    )

    engine._on_news_assessment(symbol, assessment)

    snapshot = engine._state.news.get(symbol)
    assert isinstance(snapshot, NewsSnapshot)
    assert snapshot.ticker == symbol
    assert snapshot.headline_count == 1

    print("\n" + "=" * 88)
    print("[E2E][LIVE_ENGINE NEWS SNAPSHOT]")
    print(f"ticker              : {snapshot.ticker}")
    print(f"risk_category       : {snapshot.risk_category}")
    print(f"severity            : {snapshot.severity}")
    print(f"materiality         : {snapshot.materiality}")
    print(f"effective_risk      : {snapshot.effective_risk}")
    print(f"sentiment           : {getattr(snapshot, 'sentiment', None)}")
    print(f"rationale           : {getattr(snapshot, 'rationale', '')[:1400]}")
    print("=" * 88)

    # 5. Одни и те же bias/price, но без news и с реальной news.
    # Так измеряется фактическое влияние news-risk на решение стратегии.
    tactic_cfg = AITacticConfig(
        min_conviction=0.35,
        qty_per_trade=float(os.getenv("E2E_NEWS_QTY", "0.001")),
        max_positions=1,
        kronos_confirmation_required=False,
        initial_tickers=(symbol,),
    )

    tactical_bias = {
        symbol: {
            "direction": 1,
            "conviction": 0.80,
            "horizon": "short",
        }
    }

    no_news_state = _build_real_market_state(symbol, last_price)
    no_news_strategy = AITacticStrategy(cfg=tactic_cfg, kronos=None)
    no_news_strategy.set_tactical_bias(tactical_bias)
    no_news_orders = no_news_strategy.on_bar(no_news_state)

    with_news_state = _build_real_market_state(symbol, last_price)
    with_news_state.news[symbol] = snapshot

    with_news_strategy = AITacticStrategy(cfg=tactic_cfg, kronos=None)
    with_news_strategy.set_tactical_bias(tactical_bias)
    with_news_orders = with_news_strategy.on_bar(with_news_state)

    no_news_qty = sum(
        order.qty for order in no_news_orders if order.side == OrderSide.BUY
    )
    with_news_qty = sum(
        order.qty for order in with_news_orders if order.side == OrderSide.BUY
    )

    assert no_news_qty > 0.0

    action = AITacticStrategy.risk_action(snapshot)

    print("\n" + "=" * 88)
    print("[E2E][NEWS WEIGHT IMPACT]")
    print(f"baseline_buy_qty    : {no_news_qty}")
    print(f"news_buy_qty        : {with_news_qty}")
    print(f"news_policy_action  : {action}")
    print(f"risk_category       : {snapshot.risk_category}")
    print(f"effective_risk      : {snapshot.effective_risk}")
    print(f"baseline_orders     : {no_news_orders}")
    print(f"news_orders         : {with_news_orders}")
    print("=" * 88)

    if action == "block":
        assert with_news_qty == 0.0, (
            "Policy=block, но стратегия всё равно открывает позицию."
        )
    elif action == "dampen":
        assert 0.0 <= with_news_qty < no_news_qty, (
            "Policy=dampen, но размер позиции не уменьшился."
        )
    else:  # pass
        assert with_news_qty == pytest.approx(no_news_qty), (
            "Policy=pass, но размер позиции неожиданно изменился."
        )