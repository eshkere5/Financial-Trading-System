"""Сериализация / десериализация dataclass-объектов."""

from __future__ import annotations
import json
import dataclasses
from datetime import datetime


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def to_json(obj) -> str:
    return json.dumps(obj, cls=_Encoder, ensure_ascii=False, indent=2)


def from_json(s: str, cls):
    return cls(**json.loads(s))
