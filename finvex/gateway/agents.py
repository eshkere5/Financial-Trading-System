"""Реестр агентов. Ключ показывается один раз, хранится только его хэш.

Три вида подключения:
  external - агент снаружи (бот, n8n, Dify, приложение с MCP) сам приходит в шлюз с ключом;
  llm      - Finvex сам вызывает модель (адрес, модель, ключ) и пропускает её решения через шлюз;
  http     - Finvex сам вызывает адрес агента, который вы собрали в n8n, Dify или своём коде, и получает решение в JSON.
Секреты (ключ модели, ключ вызова агента) лежат отдельно от списка агентов и наружу не отдаются.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from finvex.core.urlguard import UnsafeUrl, check_llm_url

MODES = ("shadow", "enforce")
PROFILES = ("minimal", "standard", "full")
KINDS = ("external", "llm", "http")
SYMBOL_RE = re.compile(r"^[A-Z0-9_.\-]{1,24}$")
MAX_SYMBOLS = 10
TOUCH_SAVE_S = 60.0


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
    kind: str = "external"
    config: dict = field(default_factory=dict)
    last_seen: str = ""
    calls: int = 0

    def public(self) -> dict:
        d = asdict(self)
        d.pop("key_hash")
        return d


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def parse_symbols(value) -> list:
    items = re.split(r"[,\s;]+", value) if isinstance(value, str) else list(value or [])
    out = []
    for s in items:
        s = str(s).strip().upper()
        if not s:
            continue
        if not SYMBOL_RE.match(s):
            raise ValueError(f"некорректный тикер: {s[:24]}")
        if s not in out:
            out.append(s)
    if len(out) > MAX_SYMBOLS:
        raise ValueError(f"не больше {MAX_SYMBOLS} тикеров")
    return out


def normalize_config(kind: str, cfg: Optional[dict]) -> dict:
    cfg = dict(cfg or {})
    if kind == "external":
        return {}
    symbols = parse_symbols(cfg.get("symbols"))
    prompt = str(cfg.get("system_prompt") or "").strip()[:2000]
    if kind == "llm":
        base = str(cfg.get("base_url") or "").strip()
        if base:
            try:
                base = check_llm_url(base)
            except UnsafeUrl as exc:
                raise ValueError(str(exc))
        model = str(cfg.get("model") or "").strip()[:80]
        if base and not model:
            raise ValueError("для своего адреса модели укажите название модели")
        if not prompt:
            raise ValueError("опишите роль агента (промпт)")
        return {"base_url": base, "model": model, "system_prompt": prompt, "symbols": symbols}
    url = str(cfg.get("url") or "").strip()
    try:
        url = check_llm_url(url)
    except UnsafeUrl as exc:
        raise ValueError(f"адрес агента: {exc}")
    return {"url": url, "system_prompt": prompt, "symbols": symbols}


class AgentStore:
    def __init__(self, path: "str | Path | None") -> None:
        self.path = Path(path) if path else None
        self.secrets_path = self.path.with_name("agent_secrets.json") if self.path else None
        self._lock = threading.RLock()
        self._agents: dict = {}
        self._secrets: dict = {}
        self._saved_at = 0.0
        if self.path and self.path.exists():
            known = Agent.__dataclass_fields__
            for d in json.loads(self.path.read_text(encoding="utf-8")).get("agents", []):
                self._agents[d["id"]] = Agent(**{k: v for k, v in d.items() if k in known})
        if self.secrets_path and self.secrets_path.exists():
            try:
                self._secrets = json.loads(self.secrets_path.read_text(encoding="utf-8")) or {}
            except ValueError:
                self._secrets = {}

    @staticmethod
    def _write(path: Path, data: dict, private: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if private:
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
        tmp.replace(path)

    def _save(self) -> None:
        self._saved_at = time.monotonic()
        if self.path:
            self._write(self.path, {"agents": [asdict(a) for a in self._agents.values()]})
        if self.secrets_path:
            self._write(self.secrets_path, self._secrets, private=True)

    def create(self, name: str, mode: str = "shadow", profile: str = "standard", created: str = "",
               key: Optional[str] = None, daily_quota: int = 200, kind: str = "external",
               config: Optional[dict] = None, api_key: str = "") -> "tuple[Agent, str]":
        name = (name or "").strip()[:60]
        if not name:
            raise ValueError("имя агента обязательно")
        if mode not in MODES or profile not in PROFILES:
            raise ValueError("недопустимый режим или профиль контекста")
        if kind not in KINDS:
            raise ValueError("вид подключения: external, llm или http")
        cfg = normalize_config(kind, config)
        key = key or "fvx_" + secrets.token_urlsafe(24)
        with self._lock:
            ag = Agent(id="ag_" + uuid.uuid4().hex[:8], name=name, key_hash=_hash(key), mode=mode,
                       context_profile=profile, created=created, daily_quota=int(daily_quota), kind=kind, config=cfg)
            self._agents[ag.id] = ag
            if kind in ("llm", "http") and (api_key or "").strip():
                self._secrets.setdefault(ag.id, {})["api_key"] = api_key.strip()
            self._save()
        return ag, key

    def secret(self, agent_id: str, name: str) -> str:
        return str((self._secrets.get(agent_id) or {}).get(name) or "")

    def authenticate(self, key: str) -> Optional[Agent]:
        if not key:
            return None
        h = _hash(key)
        found = None
        for ag in self._agents.values():
            if hmac.compare_digest(ag.key_hash, h):
                found = ag
        # Агенты llm и http не принимают ключ: их вызывает сам Finvex.
        return found if found and not found.revoked and found.kind == "external" else None

    def touch(self, agent_id: str, ts: str) -> None:
        """Отмечает обращение агента (для проверки подключения). Диск не нагружается: запись не чаще раза в минуту."""
        with self._lock:
            ag = self._agents.get(agent_id)
            if ag is None:
                return
            first = not ag.last_seen
            ag.last_seen, ag.calls = ts, ag.calls + 1
            if first or time.monotonic() - self._saved_at > TOUCH_SAVE_S:
                self._save()

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
            if "config" in kw:
                ag.config = normalize_config(ag.kind, {**ag.config, **dict(kw["config"] or {})})
            if kw.get("api_key") and ag.kind in ("llm", "http"):
                self._secrets.setdefault(ag.id, {})["api_key"] = str(kw["api_key"]).strip()
            if kw.get("clear_api_key"):
                (self._secrets.get(ag.id) or {}).pop("api_key", None)
            self._save()
            return ag

    def delete(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id not in self._agents:
                return False
            del self._agents[agent_id]
            self._secrets.pop(agent_id, None)
            self._save()
            return True
