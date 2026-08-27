"""
config_loader.py
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.engine.risk_manager import RiskConfig


# ── вспомогательное ──────────────────────────────────────────────────────────

def _find_config_dir() -> Path:
    for name in ("config", "configs"):
        p = Path(name)
        if p.exists():
            return p
    raise FileNotFoundError(
        "Папка с конфигами не найдена. Ожидается config/ или configs/ в корне проекта."
    )


# ── TinkoffConfig ─────────────────────────────────────────────────────────────

@dataclass
class TinkoffConfig:
    token:   str
    sandbox: bool


def load_tinkoff_config(sandbox: bool = True) -> TinkoffConfig:
    cfg_dir = _find_config_dir()
    key     = "sandbox_key.yaml" if sandbox else "prod_key.yaml"
    path    = cfg_dir / key

    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")

    raw   = yaml.safe_load(path.read_text(encoding="utf-8"))
    token = raw.get("tinkoff", {}).get("token", "")

    if not token:
        raise ValueError(f"Токен не найден в {path}")
    if not isinstance(token, str):
        raise TypeError(f"Токен должен быть строкой, получен {type(token)}")
    if not token.startswith("t."):
        raise ValueError(f"Некорректный токен: {token[:8]}…")

    return TinkoffConfig(token=token, sandbox=sandbox)

# ── BybitConfig ───────────────────────────────────────────────────────────────

@dataclass
class BybitConfig:
    api_key: str
    api_secret: str
    testnet: bool
    category: str = "linear"


def load_bybit_config(testnet: bool = True) -> BybitConfig:
    cfg_dir = _find_config_dir()
    key = "bybit_testnet_key.yaml" if testnet else "bybit_mainnet_key.yaml"
    path = cfg_dir / key
    if not path.exists():
        raise FileNotFoundError(f"Не найден конфиг ключей Bybit: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    bybit_raw = (raw or {}).get("bybit", {})
    api_key = bybit_raw.get("api_key", "")
    api_secret = bybit_raw.get("api_secret", "")
    category = bybit_raw.get("category", "linear")
    if not api_key or not api_secret:
        raise ValueError(f"api_key/api_secret не заданы в {path}")
    if api_key.startswith("YOUR_") or api_secret.startswith("YOUR_"):
        raise ValueError(f"Заполни реальные ключи в {path}, сейчас там placeholder")
    return BybitConfig(api_key=api_key, api_secret=api_secret, testnet=testnet, category=category)

# ── RiskConfig ────────────────────────────────────────────────────────────────
# ПРИМЕЧАНИЕ: собственный класс RiskConfig удалён (см. фикс в докстринге модуля).
# Канонический RiskConfig импортируется из src.engine.risk_manager выше —
# его поля (max_position_pct, max_drawdown, stop_loss_pct, take_profit_pct,
# max_open_positions, daily_loss_limit_pct, min_cash_pct) полностью совпадают
# с секцией risk: в engine.yaml.


# ── SignalConfig ──────────────────────────────────────────────────────────────

@dataclass
class SignalConfig:
    """Параметры сигналов. Читается из engine.yaml → signal:"""
    pvalue_threshold:      float = 0.05
    min_half_life:         int   = 5
    max_half_life:         int   = 126
    entry_threshold_sigma: float = 2.0
    exit_threshold_sigma:  float = 0.5
    kronos_horizon:        int   = 10
    min_confidence:        float = 0.55
    news_alpha:            float = 0.30
    news_hours_back:       int   = 6


# ── StrategyConfig ────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    # stat-arb
    zscore_entry:    float = 2.0
    zscore_exit:     float = 0.5
    # signal generator — адаптивные пороги
    threshold_sigma: float = 1.0    # |ER| > sigma * vol → сигнал
    min_confidence:  float = 0.3
    min_strength:    float = 0.005  # минимальный |ER| (0.5%)


# ── EngineConfig ──────────────────────────────────────────────────────────────

@dataclass
class UniverseConfig:
    """Настройки для UniverseSelector (src/signals/universe_selector.py).

    Заменяет статичный список EngineConfig.tickers в live-режиме:
    тикеры подбираются динамически из ликвидного пула + новостной
    активности + ватчлиста LLM-стратега.
    """
    max_universe_size:      int   = 40
    min_lot_price_rub:      float = 50.0
    require_short_enabled:  bool  = False
    refresh_interval_sec:   int   = 3600
    news_boost_weight:      float = 0.3
    hard_exclude:           List[str] = field(default_factory=list)
    fallback_tickers:       List[str] = field(default_factory=lambda: ["SBER", "GAZP", "LKOH", "YNDX", "GMKN"])


@dataclass
class EngineConfig:
    mode:            str            # backtest | sandbox | live
    broker:          str            # tinkoff
    commission:      float
    currency:        str
    timeframe:       str            # 1d | 1h | 5m | 1m
    start_date:      Optional[str]
    end_date:        Optional[str]
    initial_capital: float
    tickers:         List[str]      = field(default_factory=list)   # только backtest
    risk:            RiskConfig     = field(default_factory=RiskConfig)
    strategy:        StrategyConfig = field(default_factory=StrategyConfig)
    signal:          SignalConfig   = field(default_factory=SignalConfig)
    universe:        UniverseConfig = field(default_factory=UniverseConfig)


def load_llm_shared(path: str = "configs/llm_shared.yaml") -> Dict[str, Any]:
    """
    Грузит единую точку настройки LLM (provider/base_url/model/api_key).

    Используется во всех трёх местах, где раньше была своя копия
    provider/model/api_key (было источником "двоения" ключей):
      - configs/news_agent.yaml        (llm: summarize)
      - configs/llm_strategist.yaml    (llm_strategist: стратег)
      - news_block_patches/config/deepseek_news.yaml (llm: анализатор)

    ENV DEEPSEEK_API_KEY всегда итог приоритетнее api_key в файле.
    """
    p = Path(path)
    if not p.exists():
        return {"provider": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash", "api_key": ""}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw["api_key"] = os.environ.get("DEEPSEEK_API_KEY", raw.get("api_key", ""))
    return raw


def merge_llm_section(local_section: Dict[str, Any], shared: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Сливает локальный llm:-блок конфига с общим llm_shared.yaml.
    Локальные provider/base_url/model/api_key игнорируются (чтобы не
    воскрешались вручную и снова расходились) — остаются только
    специфичные параметры (temperature, max_tokens, timeout_seconds и т.д.).
    """
    shared = shared if shared is not None else load_llm_shared()
    merged = dict(local_section or {})
    for key in ("provider", "base_url", "model", "api_key"):
        merged[key] = shared.get(key, merged.get(key, ""))
    return merged


def load_engine_config(path: str = "configs/engine.yaml") -> EngineConfig:
    raw      = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    risk_raw = raw.get("risk", {})
    # ФИКС: раньше здесь читался несуществующий ключ "stat_arb" — в engine.yaml
    # такой секции нет, есть только "signal". zscore_entry/zscore_exit в
    # StrategyConfig этого файла больше не используются реальным кодом
    # (main.py строит src.signals.strategy.StrategyConfig, а не этот класс) —
    # оставлены как совместимость со старым API, но теперь тоже читаются из signal:.
    sig_raw  = raw.get("signal", {})

    return EngineConfig(
        mode            = raw.get("mode",             "backtest"),
        broker          = raw.get("broker",           "tinkoff"),
        commission      = float(raw.get("commission", 0.0005)),
        currency        = raw.get("currency",         "RUB"),
        timeframe       = raw.get("timeframe",        "1d"),
        start_date      = raw.get("start_date"),
        end_date        = raw.get("end_date"),
        initial_capital = float(raw.get("initial_capital", 1_000_000)),
        tickers         = raw.get("tickers", []),
        universe=UniverseConfig(
            max_universe_size     = int(raw.get("universe", {}).get("max_universe_size", 40)),
            min_lot_price_rub     = float(raw.get("universe", {}).get("min_lot_price_rub", 50.0)),
            require_short_enabled = bool(raw.get("universe", {}).get("require_short_enabled", False)),
            refresh_interval_sec  = int(raw.get("universe", {}).get("refresh_interval_sec", 3600)),
            news_boost_weight     = float(raw.get("universe", {}).get("news_boost_weight", 0.3)),
            hard_exclude          = raw.get("universe", {}).get("hard_exclude", []),
            fallback_tickers      = raw.get("universe", {}).get("fallback_tickers", ["SBER", "GAZP", "LKOH", "YNDX", "GMKN"]),
        ),
        risk=RiskConfig(
            max_position_pct     = float(risk_raw.get("max_position_pct",     0.20)),
            max_open_positions   = int(risk_raw.get("max_open_positions",     10)),
            stop_loss_pct        = float(risk_raw.get("stop_loss_pct",        0.05)),
            max_drawdown         = float(risk_raw.get("max_drawdown",         0.20)),
            take_profit_pct      = float(risk_raw.get("take_profit_pct",      0.15)),
            daily_loss_limit_pct = float(risk_raw.get("daily_loss_limit_pct", 0.05)),
            min_cash_pct         = float(risk_raw.get("min_cash_pct",         0.05)),
        ),
        strategy=StrategyConfig(
            zscore_entry    = float(sig_raw.get("entry_threshold_sigma", 2.0)),
            zscore_exit     = float(sig_raw.get("exit_threshold_sigma",  0.5)),
            threshold_sigma = float(sig_raw.get("entry_threshold_sigma", 1.0)),
            min_confidence  = float(sig_raw.get("min_confidence",        0.3)),
            min_strength    = float(sig_raw.get("min_strength",        0.005)),
        ),
        signal=SignalConfig(
            pvalue_threshold      = float(sig_raw.get("pvalue_threshold",      0.05)),
            min_half_life         = int(sig_raw.get("min_half_life",           5)),
            max_half_life         = int(sig_raw.get("max_half_life",           126)),
            entry_threshold_sigma = float(sig_raw.get("entry_threshold_sigma", 2.0)),
            exit_threshold_sigma  = float(sig_raw.get("exit_threshold_sigma",  0.5)),
            kronos_horizon        = int(sig_raw.get("kronos_horizon",          10)),
            min_confidence        = float(sig_raw.get("min_confidence",        0.55)),
            news_alpha            = float(sig_raw.get("news_alpha",            0.30)),
            news_hours_back       = int(sig_raw.get("news_hours_back",         6)),
        ),
    )


# ── KronosConfig ──────────────────────────────────────────────────────────────

@dataclass
class KronosConfig:
    model_name:        str   = "NeoQuasar/Kronos-small"
    tokenizer_name:    str   = "NeoQuasar/Kronos-Tokenizer-base"
    device:            str   = "cpu"
    max_context:       int   = 512
    prediction_length: int   = 120
    sample_count:      int   = 8
    temperature:       float = 1.0
    top_p:             float = 0.9


def load_kronos_config(path: str = "configs/kronos.yaml") -> KronosConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return KronosConfig(
        model_name        = raw.get("model_name",        "NeoQuasar/Kronos-small"),
        tokenizer_name    = raw.get("tokenizer_name",    "NeoQuasar/Kronos-Tokenizer-base"),
        device            = raw.get("device",            "cpu"),
        max_context       = int(raw.get("max_context",       512)),
        prediction_length = int(raw.get("prediction_length", 120)),
        sample_count      = int(raw.get("sample_count",      8)),
        temperature       = float(raw.get("temperature",     1.0)),
        top_p             = float(raw.get("top_p",           0.9)),
    )
