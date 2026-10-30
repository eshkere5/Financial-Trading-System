"""Единая иерархия ошибок адаптеров. Код выше адаптеров не должен знать про httpx/pybit/SDK брокера."""
from __future__ import annotations


class AdapterError(Exception):
    """Базовая ошибка слоя адаптеров."""


class ConfigError(AdapterError):
    """Некорректная конфигурация провайдера или реестра."""


class UnknownProviderError(ConfigError):
    """Тип провайдера не найден в реестре."""


class UnsupportedCapability(AdapterError):
    """Провайдер не умеет запрошенную операцию (см. ProviderInfo.capabilities)."""


class UnsupportedTimeframe(UnsupportedCapability):
    """Провайдер не поддерживает запрошенный таймфрейм."""


class ProviderError(AdapterError):
    """Ошибка источника: сеть, HTTP-статус, неожиданный формат ответа."""


class DataFormatError(ProviderError):
    """Ответ источника не соответствует ожидаемой схеме."""


class ExecutionDisabledError(AdapterError):
    """Попытка исполнения при отключённом контуре исполнения."""
