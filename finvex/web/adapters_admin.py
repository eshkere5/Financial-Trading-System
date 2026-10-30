"""Управление источниками данных из интерфейса.

Добавленные источники пишутся в configs/providers.local.yaml (основной providers.yaml и его комментарии не трогаются)
и подхватываются при следующем запуске. Исполнение отсюда включить нельзя: это сознательно делается вручную в конфиге.
Тип провайдера берётся только из реестра, импорт по произвольному пути из интерфейса запрещён.
"""
from __future__ import annotations

import importlib.util
import json
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from finvex.core import registry
from finvex.core.providers_config import load_local, local_path as _lp, merged_config

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
DESC = {"moex_iss": "Мосбиржа, ISS: свечи и котировки (данные с задержкой)", "csv_file": "Свечи из CSV-файлов в папке",
        "bybit_public": "Bybit: публичные свечи и инструменты, ключ не нужен", "disabled": "Исполнение выключено"}
SAMPLE = {"BYBIT": "BTCUSDT", "MOEX": "SBER"}
MAX_PARAMS = 2048


class AdminError(ValueError):
    pass


class AdaptersAdmin:
    def __init__(self, root: "str | Path" = ".") -> None:
        self.root, self._lock = Path(root), threading.Lock()

    def describe(self) -> dict:
        base = yaml.safe_load((self.root / "configs" / "providers.yaml").read_text(encoding="utf-8")) or {}
        local = load_local(self.root).get("market_data") or {}
        rows = [{"name": n, "type": s.get("type"), "params": s.get("params") or {}, "source": "providers.yaml", "removable": False}
                for n, s in (base.get("market_data") or {}).items()]
        rows += [{"name": n, "type": s.get("type"), "params": s.get("params") or {}, "source": "добавлен в интерфейсе", "removable": True}
                 for n, s in local.items()]
        ex = base.get("execution") or {}
        return {"market_data": rows,
                "types": [{"type": t, "desc": DESC.get(t, "")} for t in registry.available("market_data")],
                "execution": {"type": ex.get("type", "disabled"), "enabled": bool(ex.get("enabled")),
                              "available": registry.available("execution")},
                "sdk": {"pybit": importlib.util.find_spec("pybit") is not None},
                "note": "Изменения источников вступают в силу после перезапуска. Исполнение включается только вручную в конфиге."}

    def add(self, name: str, type_: str, params: dict) -> dict:
        name, type_ = (name or "").strip().lower(), (type_ or "").strip()
        if not NAME_RE.match(name):
            raise AdminError("имя: латиница, цифры и _, от 2 до 31 символа, с буквы")
        if ":" in type_ or type_ not in registry.available("market_data"):
            raise AdminError(f"неизвестный тип «{type_}». Доступно: {', '.join(registry.available('market_data'))}")
        if not isinstance(params, dict):
            raise AdminError("параметры должны быть объектом")
        try:
            blob = json.dumps(params)
        except (TypeError, ValueError):
            raise AdminError("параметры должны быть JSON")
        if len(blob) > MAX_PARAMS:
            raise AdminError("слишком много параметров")
        for k, v in params.items():
            if not isinstance(k, str) or isinstance(v, (dict, list)) and k.endswith("url"):
                raise AdminError(f"параметр {k}: неверный вид")
            if k.endswith("url") and not str(v).startswith("https://"):
                raise AdminError(f"{k}: адрес должен начинаться с https://")
        base = yaml.safe_load((self.root / "configs" / "providers.yaml").read_text(encoding="utf-8")) or {}
        if name in (base.get("market_data") or {}):
            raise AdminError("такое имя занято основным конфигом, выберите другое")
        with self._lock:
            local = load_local(self.root)
            local.setdefault("market_data", {})[name] = {"type": type_, "params": params}
            p = _lp(self.root)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(yaml.safe_dump(local, allow_unicode=True, sort_keys=False), encoding="utf-8")
            tmp.replace(p)
        return self.describe()

    def remove(self, name: str) -> dict:
        with self._lock:
            local = load_local(self.root)
            if name not in (local.get("market_data") or {}):
                raise AdminError("удалять можно только источники, добавленные в интерфейсе")
            del local["market_data"][name]
            _lp(self.root).write_text(yaml.safe_dump(local, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return self.describe()

    def check(self, name: str) -> dict:
        """Создаёт провайдер из конфига и берёт несколько свечей. Нужен интернет (для Bybit из РФ ещё и VPN)."""
        spec = (merged_config(self.root).get("market_data") or {}).get(name)
        if not spec:
            raise AdminError("источник не найден")
        try:
            prov = registry.create("market_data", spec["type"], **(spec.get("params") or {}))
        except Exception as exc:
            return {"ok": False, "message": f"не удалось создать: {exc}"[:200]}
        try:
            sym = SAMPLE.get(getattr(prov.info, "venue", ""), "")
            if not sym:
                return {"ok": True, "message": "провайдер создан; проверка свечей для этой площадки не задана"}
            now = datetime.now(timezone.utc)
            s = prov.get_candles(sym, "1h", now - timedelta(days=3), now)
            if not len(s):
                return {"ok": False, "message": f"{sym}: свечей нет (рынок закрыт или нет доступа)"}
            return {"ok": True, "message": f"{sym}: получено {len(s)} свечей, последняя {s.candles[-1].close:g}"}
        except Exception as exc:
            return {"ok": False, "message": f"нет данных: {type(exc).__name__}: {str(exc)[:150]}. Проверьте интернет или VPN"}
        finally:
            try:
                prov.close()
            except Exception:
                pass
