"""Чтение конфигурации провайдеров: основной providers.yaml плюс источники, добавленные из интерфейса."""
from __future__ import annotations

from pathlib import Path

import yaml


def local_path(root: Path) -> Path:
    return Path(root) / "configs" / "providers.local.yaml"


def load_local(root: Path) -> dict:
    p = local_path(root)
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


def merged_config(root: Path) -> dict:
    cfg = yaml.safe_load((Path(root) / "configs" / "providers.yaml").read_text(encoding="utf-8")) or {}
    cfg.setdefault("market_data", {}).update(load_local(root).get("market_data") or {})
    return cfg
