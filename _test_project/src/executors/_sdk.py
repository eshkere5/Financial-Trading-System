"""
_sdk.py — единая точка совместимого импорта Tinkoff Invest SDK.
Разные версии/форки пакета называются either `t_tech.invest`, либо `tinkoff.invest`.
Все модули должны импортировать отсюда, а не дублировать try/except.
"""
try:
    from t_tech.invest.schemas import OrderDirection, OrderType
    from t_tech.invest import AsyncClient
except ImportError:
    from tinkoff.invest.schemas import OrderDirection, OrderType  # type: ignore
    from tinkoff.invest import AsyncClient  # type: ignore

try:
    from t_tech.invest.utils import quotation_to_decimal
except ImportError:
    from tinkoff.invest.utils import quotation_to_decimal  # type: ignore

__all__ = ["OrderDirection", "OrderType", "AsyncClient", "quotation_to_decimal"]