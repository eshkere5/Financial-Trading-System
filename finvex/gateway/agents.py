"""Реестр агентов. Ключ показывается один раз, хранится только его хэш."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

MODES = ("shadow", "enforce")
PROFILES = ("minimal", "standard", "full")


@dataclass
class Agent:
    id: str
    name: str
    key_hash: str
    mode: str = "shadow"
    context_profile: str = "standard"
    daily_quota: int = 200
    revoked: bool = False
    created: str = ""
    overrides: dict = field(default_factory=dict)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("key_hash")
        return d


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class AgentStore:
    def __init__(self, path: "str | Path | None") -> None:
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self._agents: dict = {}
        if self.path and self.path.exists():
            for d in json.loads(self.path.read_text(encoding="utf-8")).get("agents", []):
                self._agents[d["id"]] = Agent(**d)

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {"agents": [asdict(a) for a in self._agents.values()]}
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def create(self, name: str, mode: str = "shadow", profile: str = "standard", created: str = "",
               key: Optional[str] = None, daily_quota: int = 200) -> "tuple[Agent, str]":
        name = (name or "").strip()[:60]
        if not name:
            raise ValueError("имя агента обязательно")
        if mode not in MODES or profile not in PROFILES:
            raise ValueError("недопустимый режим или профиль контекста")
        key = key or "fvx_" + secrets.token_urlsafe(24)
        with self._lock:
            ag = Agent(id="ag_" + uuid.uuid4().hex[:8], name=name, key_hash=_hash(key), mode=mode,
                       context_profile=profile, created=created, daily_quota=int(daily_quota))
            self._agents[ag.id] = ag
            self._save()
        return ag, key

    def authenticate(self, key: str) -> Optional[Agent]:
        if not key:
            return None
        h = _hash(key)
        found = None
        for ag in self._agents.values():
            if hmac.compare_digest(ag.key_hash, h):
                found = ag
        return found if found and not found.revoked else None

    def get(self, agent_id: str) -> Optional[Agent]:
        return self._agents.get(agent_id)

    def list(self) -> list:
        return list(self._agents.values())

    def update(self, agent_id: str, **kw) -> Agent:
        with self._lock:
            ag = self._agents[agent_id]
            if "mode" in kw:
                if kw["mode"] not in MODES:
                    raise ValueError("режим: shadow или enforce")
                ag.mode = kw["mode"]
            if "context_profile" in kw:
                if kw["context_profile"] not in PROFILES:
                    raise ValueError("профиль: minimal, standard или full")
                ag.context_profile = kw["context_profile"]
            if "revoked" in kw:
                ag.revoked = bool(kw["revoked"])
            if "overrides" in kw:
                ag.overrides = dict(kw["overrides"])
            self._save()
            return ag
