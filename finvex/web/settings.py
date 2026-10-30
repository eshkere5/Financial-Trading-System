"""Ключ LLM для аналитики и чата: чтение, безопасное сохранение, проверка.

Ключ пишется в configs/llm_shared.yaml (в .gitignore, права 0600) и в браузер не возвращается: наружу уходят
источник и последние 4 символа. Переменная окружения DEEPSEEK_API_KEY приоритетнее файла.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from urllib.parse import urlparse

import yaml

DEFAULTS = {"provider": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash"}
ENV_KEY = "DEEPSEEK_API_KEY"


class SettingsError(ValueError):
    pass


class LlmSettings:
    def __init__(self, path: "str | Path" = "configs/llm_shared.yaml") -> None:
        self.path, self._lock = Path(path), threading.Lock()

    def _raw(self) -> dict:
        try:
            return (yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}) if self.path.exists() else {}
        except Exception:
            return {}

    def api_key(self) -> str:
        return os.environ.get(ENV_KEY) or str(self._raw().get("api_key") or "")

    def has_key(self) -> bool:
        return bool(self.api_key())

    def public(self) -> dict:
        raw, key = self._raw(), self.api_key()
        return {"provider": raw.get("provider") or DEFAULTS["provider"], "base_url": raw.get("base_url") or DEFAULTS["base_url"],
                "model": raw.get("model") or DEFAULTS["model"], "has_key": bool(key),
                "key_hint": ("…" + key[-4:]) if len(key) >= 8 else ("задан" if key else ""),
                "source": "env" if os.environ.get(ENV_KEY) else ("file" if raw.get("api_key") else "none")}

    def save(self, base_url: str, model: str, api_key: str = "", clear: bool = False) -> dict:
        base_url, model = (base_url or "").strip().rstrip("/"), (model or "").strip()
        u = urlparse(base_url)
        if u.scheme != "https" or not u.netloc:
            raise SettingsError("адрес API должен начинаться с https://")
        if not model or len(model) > 80:
            raise SettingsError("укажите модель (до 80 символов)")
        api_key = (api_key or "").strip()
        if api_key and (len(api_key) < 8 or len(api_key) > 300 or any(c.isspace() for c in api_key)):
            raise SettingsError("ключ выглядит неверно: проверьте, что он скопирован целиком и без пробелов")
        with self._lock:
            raw = self._raw()
            raw.update({"provider": raw.get("provider") or DEFAULTS["provider"], "base_url": base_url, "model": model})
            if clear:
                raw.pop("api_key", None)
            elif api_key:
                raw["api_key"] = api_key
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, self.path)
        return self.public()

    def check(self, timeout: float = 10.0) -> dict:
        import requests
        key, base = self.api_key(), self.public()["base_url"]
        if not key:
            return {"ok": False, "message": "ключ не задан"}
        try:
            r = requests.get(base + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        except Exception as exc:
            return {"ok": False, "message": f"нет связи с {urlparse(base).netloc}: {type(exc).__name__}. Нужен интернет или VPN"}
        if r.status_code == 200:
            return {"ok": True, "message": "ключ принят"}
        return {"ok": False, "message": "провайдер отклонил ключ" if r.status_code in (401, 403) else f"провайдер ответил {r.status_code}"}
