"""Реестр модулей. Состояние хранится в data/gateway/modules.json.

Модуль включается только при включённых обязательных зависимостях и выключается, только если от него не зависят
другие включённые модули. По умолчанию все модули включены. Выключенный модуль закрывает свои API-маршруты;
защитный слой при выключении отказывает (fail-closed), а не пропускает заявки.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

MODULES = [
    {"id": "data", "title": "Адаптеры данных и исполнения", "version": "1.0",
     "summary": "Источники котировок и адаптер исполнения. Исполнение включается отдельно в конфигурации.",
     "requires": [], "optional": [],
     "api": ["GET /api/adapters", "PUT /api/adapters", "POST /api/adapters/check"], "ui": ["Данные рынка для проверок: источники задаются в configs/providers.yaml"]},
    {"id": "analytics", "title": "Аналитика", "version": "1.0",
     "summary": "Новостной фон и прогноз Kronos как входные данные для правил. Без модуля правила аналитики пропускаются.",
     "requires": [], "optional": ["data"],
     "api": ["GET /api/analytics", "PUT /api/analytics"], "ui": ["Правила «Новостной фон» и «Прогноз Kronos»"]},
    {"id": "guard", "title": "Защитный слой", "version": "1.0",
     "summary": "Правила, проверка намерений агентов, ключи агентов, журнал вердиктов. При выключении заявки отклоняются.",
     "requires": ["data"], "optional": ["analytics"],
     "api": ["GET /gateway/v1/context", "GET /gateway/v1/limits", "POST /gateway/v1/propose", "GET /gateway/v1/verdicts/{id}",
             "GET /api/verdicts", "GET /api/rules", "PUT /api/rules", "GET /api/agents"], "ui": ["Вердикты","Правила","Агенты","Контекст и проверка намерения в боковой панели"]},
    {"id": "hub", "title": "Хаб агентов и чат", "version": "1.0",
     "summary": "Чат с аналитическим агентом и песочница для проверки своих агентов через защитный слой.",
     "requires": ["guard"], "optional": ["analytics"],
     "api": ["POST /api/chat", "GET /api/sandbox/presets", "POST /api/sandbox/run"], "ui": ["Чат","Песочница"]},
]


class ModuleError(Exception):
    pass


class ModuleDisabled(ModuleError):
    def __init__(self, module_id: str) -> None:
        super().__init__(f"модуль «{module_id}» выключен")
        self.module_id = module_id


class ModuleRegistry:
    def __init__(self, path: "str | Path | None" = None, manifests: Optional[list] = None) -> None:
        self.path = Path(path) if path else None
        self._m = {m["id"]: m for m in (manifests or MODULES)}
        self._lock = threading.RLock()
        self._enabled = {k: True for k in self._m}
        if self.path and self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                for k, v in (saved or {}).items():
                    if k in self._enabled and isinstance(v, bool):
                        self._enabled[k] = v
            except (OSError, ValueError, AttributeError):
                pass
        changed = True
        while changed:     # зависимый модуль не может остаться включённым без основы
            changed = False
            for k, m in self._m.items():
                if self._enabled[k] and any(not self._enabled.get(r, False) for r in m["requires"]):
                    self._enabled[k], changed = False, True

    def is_enabled(self, module_id: str) -> bool:
        with self._lock:
            return bool(self._enabled.get(module_id, False))

    def require(self, module_id: str) -> None:
        if not self.is_enabled(module_id):
            raise ModuleDisabled(module_id)

    def set_enabled(self, module_id: str, enabled: bool) -> list:
        with self._lock:
            if module_id not in self._m:
                raise ModuleError("неизвестный модуль")
            if enabled:
                missing = [r for r in self._m[module_id]["requires"] if not self._enabled[r]]
                if missing:
                    raise ModuleError("сначала включите: " + ", ".join(missing))
            else:
                users = [k for k, m in self._m.items() if self._enabled[k] and module_id in m["requires"]]
                if users:
                    raise ModuleError("от модуля зависят включённые: " + ", ".join(users))
            old = self._enabled[module_id]
            self._enabled[module_id] = enabled
            try:
                self._save()
            except OSError:
                self._enabled[module_id] = old
                raise ModuleError("не удалось сохранить состояние модулей")
            return self.list()

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._enabled, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def list(self) -> list:
        with self._lock:
            return [{**m, "enabled": self._enabled[k],
                     "can_enable": all(self._enabled[r] for r in m["requires"]),
                     "can_disable": not any(self._enabled[o] and k in x["requires"] for o, x in self._m.items())}
                    for k, m in self._m.items()]

    def capabilities(self) -> dict:
        with self._lock:
            return {"modules": [{"id": k, "version": m["version"], "api": m["api"]} for k, m in self._m.items() if self._enabled[k]]}
