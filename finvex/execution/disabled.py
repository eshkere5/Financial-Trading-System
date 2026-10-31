"""Отключённый контур исполнения — значение по умолчанию для банковского профиля."""
from __future__ import annotations

from finvex.core.base import Capability, ExecutionProvider, ProviderInfo
from finvex.core.errors import ExecutionDisabledError
from finvex.core.models import OrderAck, OrderRequest
from finvex.core.registry import register


@register("disabled", kind="execution")
class DisabledExecutionProvider(ExecutionProvider):
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(name="disabled", venue="-", capabilities=Capability.NONE)

    def submit(self, order: OrderRequest) -> OrderAck:
        raise ExecutionDisabledError(
            "Исполнение отключено (профиль аналитический). "
            f"Заявка по плану {order.plan_id} не отправлена."
        )

    def cancel(self, client_order_id: str) -> OrderAck:
        raise ExecutionDisabledError("Исполнение отключено")
