"""
lot_utils.py — общая логика нормализации размера ордера под требования брокера.
Используется и синхронным, и асинхронным роутерами, чтобы устранить
дублирование и расхождение поведения (см. К3/Н1 в код-ревью).
"""
from __future__ import annotations
import logging
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)


def shares_to_lots(qty_shares: float, lot_size: int) -> int:
    """
    Конвертирует количество акций в количество лотов Tinkoff.
    Округление ВНИЗ (никогда не отправляем больше, чем попросила стратегия).
    Возвращает 0, если итоговое количество лотов меньше 1 (ордер не отправлять).
    """
    if lot_size <= 0:
        lot_size = 1
    lots = int(qty_shares // lot_size)
    return max(0, lots)


def round_qty_to_step(qty: float, qty_step: float, min_qty: float = 0.0) -> float:
    """
    Округляет количество вниз до кратного qty_step (Bybit qtyStep), используя
    Decimal, чтобы избежать ошибок бинарного float (0.1 + 0.2 != 0.3).
    Возвращает 0.0, если итоговое количество меньше min_qty (ордер не отправлять).
    """
    if qty_step <= 0:
        return qty
    d_qty = Decimal(str(qty))
    d_step = Decimal(str(qty_step))
    steps = (d_qty / d_step).to_integral_value(rounding=ROUND_DOWN)
    result = float(steps * d_step)
    if result < min_qty:
        return 0.0
    return result