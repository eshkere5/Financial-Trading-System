"""Мост к чат-агенту (market_agent). Если модуля нет, чат честно отключается."""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Optional

SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,80}$")


class Chat:
    def __init__(self, out_dir: "str | Path") -> None:
        self.out_dir = Path(out_dir)
        self.note = ""
        self._ma = None
        self._sessions: dict = {}
        self._lock = threading.Lock()
        try:
            try:
                from finvex import market_agent as ma
            except ImportError:
                import market_agent as ma  # файл лежит в корне проекта
            self._ma = ma
        except Exception as exc:
            self.note = f"чат отключён: не найден или не запускается market_agent.py ({str(exc)[:140]})"

    @property
    def available(self) -> bool:
        return self._ma is not None

    def reset(self) -> None:
        """Сбросить сессии: новый ключ и модель подхватятся при следующем вопросе."""
        with self._lock:
            self._sessions.clear()

    def _session(self, sid: str):
        with self._lock:
            if sid not in self._sessions:
                ma = self._ma
                out = self.out_dir / sid
                out.mkdir(parents=True, exist_ok=True)
                try:
                    sess = ma.Session(out_dir=out, use_kronos=False)
                except TypeError:
                    sess = ma.Session(out)
                ma.Sandbox(sess)
                self._sessions[sid] = (sess, ma.Agent(sess, verbose=False))
            return self._sessions[sid]

    def ask(self, sid: str, text: str) -> dict:
        if not self.available:
            raise RuntimeError(self.note)
        if not SAFE_NAME.match(sid):
            raise ValueError("некорректная сессия")
        sess, agent = self._session(sid)
        with self._lock:
            answer = agent.ask(text)
        imgs = [Path(p).name for p in getattr(sess, "turn_images", []) or []]
        return {"answer": str(answer), "images": imgs}

    def image_path(self, sid: str, name: str) -> Optional[Path]:
        if not (SAFE_NAME.match(sid) and SAFE_NAME.match(name)):
            return None
        p = (self.out_dir / sid / name).resolve()
        return p if p.is_file() and self.out_dir.resolve() in p.parents else None
