"""
Контроль качества рядов. Выполняется до подачи данных в Kronos и тактику.
Проверки детерминированы и не зависят от модели.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from finvex.core.models import Candle, DataQuality, QualityIssue, Timeframe, MSK


def check_candles(
    candles: Sequence[Candle],
    timeframe: Timeframe,
    *,
    now: Optional[datetime] = None,
    max_intraday_gap_bars: int = 3,
    stale_after_bars: int = 6,
    zero_volume_warn_share: float = 0.2,
) -> DataQuality:
    issues = []
    n = len(candles)
    if n == 0:
        return DataQuality(0, (QualityIssue("empty", "error", "Нет данных за запрошенный период"),))

    bad_nan = bad_price = bad_ohlc = dup = unordered = zero_vol = 0
    prev: Optional[Candle] = None
    for c in candles:
        vals = (c.open, c.high, c.low, c.close, c.volume)
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
            bad_nan += 1
            prev = c
            continue
        if min(c.open, c.high, c.low, c.close) <= 0:
            bad_price += 1
        if c.high < max(c.open, c.close, c.low) or c.low > min(c.open, c.close, c.high):
            bad_ohlc += 1
        if c.volume <= 0:
            zero_vol += 1
        if prev is not None:
            if c.ts_open == prev.ts_open:
                dup += 1
            elif c.ts_open < prev.ts_open:
                unordered += 1
        prev = c

    def add(code, sev, msg, cnt):
        if cnt:
            issues.append(QualityIssue(code, sev, msg, cnt))

    add("nan", "error", "Пустые или NaN значения OHLCV", bad_nan)
    add("non_positive_price", "error", "Нулевые или отрицательные цены", bad_price)
    add("ohlc_inconsistent", "error", "high/low не согласованы с open/close", bad_ohlc)
    add("duplicate_ts", "error", "Повторяющиеся метки времени", dup)
    add("unordered", "error", "Нарушен порядок времени", unordered)

    # Для индексов объём часто равен 0 — это нормально, поэтому только warning по доле.
    if zero_vol and zero_vol / n >= zero_volume_warn_share:
        issues.append(QualityIssue("zero_volume", "warning", "Много баров с нулевым объёмом", zero_vol))

    # Внутридневные пропуски: между торговыми днями разрывы ожидаемы, внутри дня — нет.
    if timeframe.seconds < 86400:
        limit = timedelta(seconds=timeframe.seconds * max_intraday_gap_bars)
        gaps = 0
        for a, b in zip(candles, candles[1:]):
            same_day = a.ts_open.astimezone(MSK).date() == b.ts_open.astimezone(MSK).date()
            if same_day and b.ts_open - a.ts_close > limit:
                gaps += 1
        add("intraday_gap", "warning", f"Внутридневные пропуски больше {max_intraday_gap_bars} баров", gaps)

    # Свежесть: информируем, а не блокируем — рынок может быть закрыт.
    now = now or datetime.now(timezone.utc)
    age = now - candles[-1].ts_close
    if age > timedelta(seconds=timeframe.seconds * stale_after_bars):
        issues.append(QualityIssue(
            "stale", "info",
            f"Последний бар закрылся {int(age.total_seconds() // 60)} мин назад (рынок может быть закрыт)",
        ))
    return DataQuality(n, tuple(issues))
