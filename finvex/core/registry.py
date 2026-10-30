"""
Реестр провайдеров и сборка из YAML — подключение новой биржи без правки ядра.

Три способа добавить провайдера:
  1. Декоратор @register("имя") внутри finvex/venues/ или finvex/execution/.
  2. Точка входа Python-пакета в группе `finvex.providers` (внешний плагин).
  3. Полный путь в конфиге: `type: "my_pkg.my_mod:MyProvider"`.
Конструктор провайдера получает params из конфига как именованные аргументы.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Type

from finvex.core.base import ExecutionProvider, MarketDataProvider, PortfolioProvider
from finvex.core.errors import ConfigError, UnknownProviderError

logger = logging.getLogger(__name__)

KINDS = ("market_data", "portfolio", "execution")
ENTRY_POINT_GROUP = "finvex.providers"

_REGISTRY: Dict[str, Dict[str, Callable[..., Any]]] = {k: {} for k in KINDS}
_plugins_loaded = False


def register(name: str, kind: str = "market_data") -> Callable[[Any], Any]:
    if kind not in KINDS:
        raise ConfigError(f"Неизвестный вид провайдера {kind!r}; допустимо: {KINDS}")

    def deco(factory):
        _REGISTRY[kind][name] = factory
        return factory

    return deco


def load_plugins() -> None:
    """Подхватывает внешние провайдеры из entry points; ошибки плагина не валят систему."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - зависит от окружения
        return
    for ep in eps:
        try:
            ep.load()
        except Exception as exc:
            logger.warning("Плагин провайдера %s не загружен: %s", ep.name, exc)


def available(kind: str = "market_data") -> list[str]:
    _ensure_builtin()
    load_plugins()
    return sorted(_REGISTRY[kind])


_BUILTIN_PACKAGES = ("finvex.venues", "finvex.execution")
_builtin_loaded = False


def _ensure_builtin() -> None:
    """Импортирует все модули в finvex.venues.* и finvex.execution.*: их @register срабатывает сам.

    Новый провайдер = новый файл в venues/<площадка>/; правок реестра не требуется.
    Модуль, которому не хватает SDK (pybit, t_tech), пропускается, остальные работают.
    """
    global _builtin_loaded
    if _builtin_loaded:
        return
    _builtin_loaded = True
    import pkgutil

    for pkg_name in _BUILTIN_PACKAGES:
        pkg = importlib.import_module(pkg_name)
        for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg_name + "."):
            try:
                importlib.import_module(info.name)
            except Exception as exc:
                logger.debug("Модуль провайдера %s пропущен: %s", info.name, exc)


def _resolve(kind: str, type_name: str) -> Callable[..., Any]:
    _ensure_builtin()
    if ":" in type_name:
        mod, _, attr = type_name.partition(":")
        try:
            return getattr(importlib.import_module(mod), attr)
        except (ImportError, AttributeError) as exc:
            raise UnknownProviderError(f"Не удалось импортировать {type_name!r}: {exc}") from exc
    if type_name not in _REGISTRY[kind]:
        load_plugins()
    try:
        return _REGISTRY[kind][type_name]
    except KeyError:
        raise UnknownProviderError(
            f"Провайдер {type_name!r} ({kind}) не зарегистрирован. Доступно: {sorted(_REGISTRY[kind])}"
        ) from None


def create(kind: str, type_name: str, **params: Any):
    factory = _resolve(kind, type_name)
    try:
        return factory(**params)
    except TypeError as exc:
        raise ConfigError(f"Неверные параметры для {type_name!r}: {exc}") from exc


class ProviderHub:
    """
    Набор провайдеров профиля. Аналитический код получает market_data(),
    исполнение — только через execution, который по умолчанию отключён.
    """

    def __init__(
        self,
        market_data: Mapping[str, MarketDataProvider],
        default_market_data: str,
        portfolio: Optional[PortfolioProvider] = None,
        execution: Optional[ExecutionProvider] = None,
        profile: str = "default",
    ) -> None:
        if default_market_data not in market_data:
            raise ConfigError(f"default_market_data={default_market_data!r} не описан в market_data")
        self._md = dict(market_data)
        self._default = default_market_data
        self.portfolio = portfolio
        self.profile = profile
        self.execution = execution if execution is not None else create("execution", "disabled")

    def market_data(self, name: Optional[str] = None) -> MarketDataProvider:
        key = name or self._default
        try:
            return self._md[key]
        except KeyError:
            raise ConfigError(f"Источник данных {key!r} не настроен: {sorted(self._md)}") from None

    @property
    def market_data_names(self) -> list[str]:
        return sorted(self._md)

    def close(self) -> None:
        for p in (*self._md.values(), self.portfolio, self.execution):
            if p is not None:
                p.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ProviderHub":
        if not isinstance(cfg, Mapping) or "market_data" not in cfg:
            raise ConfigError("В конфиге нужен раздел market_data")

        md: Dict[str, MarketDataProvider] = {}
        for name, spec in cfg["market_data"].items():
            md[name] = _build("market_data", name, spec)
        default = cfg.get("default_market_data") or next(iter(md))

        portfolio = None
        if cfg.get("portfolio"):
            portfolio = _build("portfolio", "portfolio", cfg["portfolio"])

        execution = None
        exec_cfg = cfg.get("execution") or {}
        # Исполнение включается только явным enabled: true; иначе всегда disabled.
        if exec_cfg.get("enabled") is True and exec_cfg.get("type", "disabled") != "disabled":
            execution = _build("execution", "execution", exec_cfg)
        elif exec_cfg.get("type", "disabled") != "disabled":
            logger.warning(
                "execution.type=%s указан, но enabled != true — используется disabled",
                exec_cfg.get("type"),
            )
        return cls(md, default, portfolio, execution, profile=str(cfg.get("profile", "default")))

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "ProviderHub":
        import yaml
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_config(yaml.safe_load(text) or {})


def _build(kind: str, name: str, spec: Mapping[str, Any]):
    if not isinstance(spec, Mapping) or "type" not in spec:
        raise ConfigError(f"{kind}.{name}: нужно поле type")
    params = dict(spec.get("params") or {})
    return create(kind, spec["type"], **params)
