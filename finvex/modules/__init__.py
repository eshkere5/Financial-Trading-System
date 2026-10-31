"""Модульный слой Finvex: манифесты, зависимости, включение и выключение."""
from .registry import ModuleDisabled, ModuleError, ModuleRegistry, MODULES

__all__ = ["ModuleDisabled", "ModuleError", "ModuleRegistry", "MODULES"]
