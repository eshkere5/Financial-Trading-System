"""Сборка шлюза: живой режим (ProviderHub проекта) и демо без сети."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from finvex.core.models import MSK

from .context import PaperPortfolio
from .gateway import Gateway
from .prices import HubPriceSource, StaticPriceSource


class DemoClock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


# агент, тикер, сторона, объём, тип, лимит, обоснование, уверенность, минут вперёд
SCENARIOS = [
    (0, "SBER", "buy", 20, "market", None, "Пробой SMA50 на росте объёма, стоп под минимумом недели", 0.72, 0),
    (0, "SBER", "buy", 20, "market", None, "Повторный вход по тому же сигналу", 0.70, 2),
    (0, "GAZP", "buy", 2000, "market", None, "Отскок от поддержки, ожидаю возврат к средней", 0.66, 18),
    (1, "LKOH", "buy", 5, "market", None, "Новостной фон по дивидендам позитивный", 0.61, 25),
    (1, "SBER", "buy", 10, "limit", 3550.0, "Лимитная заявка ниже рынка на откате", 0.64, 22),
    (1, "ROSN", "buy", 40, "market", None, "Ждём реакцию на отчётность", 0.40, 20),
    (2, "VTBR", "buy", 800, "market", None, "Перепроданность по RSI, ожидаю разворот", 0.58, 15),
    (2, "GMKN", "buy", 50, "market", None, "", 0.62, 12),
    (0, "BTCUSDT", "buy", 0.05, "market", None, "Импульс выше локального максимума, риск ограничен стопом", 0.69, 25),
    (0, "SBER", "sell", 500, "market", None, "Фиксирую прибыль по цели", 0.70, 30),
    (2, "GAZP", "sell", 300, "market", None, "Возврат к средней достигнут, сокращаю позицию", 0.63, 14),
    (1, "YNDX", "buy", 30, "market", None, "Позитивная новость по сервисам, вход на ожидании роста", 0.67, 21),
    (0, "MTSS", "buy", 100, "market", None, "Тренд вверх, ADX выше 25, вход по рынку", 0.71, 16),
    (2, "ETHUSDT", "buy", 1.5, "market", None, "Отклонение от средней, ожидаю возврат к ней в течение дня", 0.60, 19),
    (1, "SBER", "buy", 10, "limit", 35.5, "Лимитная заявка на откате, опечатка в цене возможна", 0.65, 10),
    (0, "LKOH", "sell", 5, "market", None, "Сокращаю позицию перед сезоном отчётов", 0.62, 17),
    (2, "ZZZZ", "buy", 10, "market", None, "Бумага без данных, проверка поведения шлюза", 0.60, 5),
    (1, "MOEX", "buy", 400, "market", None, "Рост оборотов биржи, ожидаю улучшение отчёта", 0.68, 12),
    (0, "GAZP", "buy", 2000, "market", None, "Дополнительная покупка по сигналу пробоя", 0.66, 3),
    (2, "ROSN", "buy", 20, "market", None, "Вход на коррекции к поддержке, риск ограничен стопом", 0.57, 18),
    (1, "SBER", "buy", 15, "market", None, "Ночная новость, вход после закрытия сессии", 0.66, 0),
    (0, "SBER", "buy", 10, "market", None, "Короткий тест связи шлюза", 0.60, 8),
]
_DEMO_AGENTS = [("momentum-bot", "shadow", "standard"), ("news-trader", "shadow", "full"), ("mean-reversion", "enforce", "minimal")]


def seed(gw: Gateway, clock: DemoClock) -> None:
    agents = [gw.agents.create(n, m, p, created=clock().isoformat(timespec="seconds"))[0] for n, m, p in _DEMO_AGENTS]
    for k, (ai, sym, side, qty, ot, lim, why, conf, mins) in enumerate(SCENARIOS):
        if k == 20:
            clock.t = clock.t.replace(hour=19, minute=30)
        elif mins:
            from datetime import timedelta
            clock.t += timedelta(minutes=mins)
        raw = dict(client_id=f"demo-{k}", symbol=sym, side=side, quantity=qty, order_type=ot, limit_price=lim,
                   rationale=why, confidence=conf)
        try:
            gw.propose(agents[ai], raw)
        except Exception:
            pass


def build(root: "str | Path" = ".", demo: bool = False) -> Gateway:
    root = Path(root)
    cfg_path = root / "configs" / "gateway.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg = cfg or {}
    if demo:
        clock = DemoClock(datetime(2026, 10, 21, 10, 20, tzinfo=MSK))
        pf = PaperPortfolio(700_000, {"SBER": 300.0, "GAZP": 500.0, "LKOH": 20.0})
        gw = Gateway(None, StaticPriceSource(), pf, clock=clock)
        gw.demo = True
        seed(gw, clock)
        clock.t = clock.t.replace(hour=12, minute=40)
        return gw
    from finvex.core.registry import ProviderHub
    from finvex.core.providers_config import merged_config   # providers.yaml + источники, добавленные в интерфейсе
    hub = ProviderHub.from_config(merged_config(root))
    pf_cfg = cfg.get("portfolio") or {}
    gw = Gateway(root / cfg.get("data_dir", "data/gateway"), HubPriceSource(hub),
                 PaperPortfolio(pf_cfg.get("cash", 1_000_000), pf_cfg.get("positions") or {}),
                 executor=hub.execution, execution_enabled=bool((cfg.get("execution") or {}).get("enabled", False)))
    gw.demo = False
    return gw
