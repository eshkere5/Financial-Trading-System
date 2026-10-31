"""Снимки аналитики: новости с сроком жизни от парсера и оценка Kronos, действующая до закрытия следующей свечи.

В контекст операции попадает только то, что ещё актуально. Просроченное удаляется из активного набора;
в вердикт пишется, какие именно новости использованы, чтобы решение можно было объяснить позже.
Хранилище ничего не считает по рынку и не ходит в сеть.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from finvex.core.models import MSK

# Границы срока жизни новости по типу события, часы. Парсер может ошибиться в оценке срока,
# поэтому его ответ всегда ограничивается этими рамками.
TTL_BOUNDS_H = {"rumor": (0.25, 4.0), "company": (1.0, 48.0), "earnings": (6.0, 120.0), "macro": (2.0, 72.0), "other": (0.25, 24.0)}
GLOBAL_TTL_H = (0.25, 168.0)
MAX_ITEMS_PER_SYMBOL = 50


class AnalyticsError(ValueError):
    pass


def _num(v, lo: float, hi: float, name: str) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise AnalyticsError(f"{name}: нужно число")
    if x != x or not lo <= x <= hi:
        raise AnalyticsError(f"{name}: допустимо от {lo:g} до {hi:g}")
    return x


def _dt(v, name: str) -> datetime:
    try:
        t = datetime.fromisoformat(str(v))
    except ValueError:
        raise AnalyticsError(f"{name}: нужен формат ISO")
    return t if t.tzinfo else t.replace(tzinfo=MSK)


def _sym(symbol) -> str:
    sym = str(symbol or "").strip().upper()
    if not sym or len(sym) > 24 or not sym.replace("_", "").isalnum():
        raise AnalyticsError("некорректный тикер")
    return sym


class AnalyticsStore:
    def __init__(self, path: "str | Path | None", clock: Optional[Callable[[], datetime]] = None) -> None:
        self.path, self.clock = (Path(path) if path else None), clock or (lambda: datetime.now(MSK))
        self._lock = threading.Lock()
        self._d: dict = {}
        if self.path and self.path.exists():
            try:
                self._d = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._d = {}

    def add_news(self, symbol: str, item: dict) -> dict:
        """item: id, score (-1..1), materiality (0..1), category, title, published_at, valid_until или ttl_hours."""
        sym, now = _sym(symbol), self.clock()
        score = _num(item.get("score"), -1, 1, "score")
        if score is None:
            raise AnalyticsError("score обязателен")
        mat = _num(item.get("materiality", 1.0), 0, 1, "materiality")
        cat = item.get("category") if item.get("category") in TTL_BOUNDS_H else "other"
        pub = _dt(item["published_at"], "published_at") if item.get("published_at") else now
        if item.get("valid_until"):
            ttl = (_dt(item["valid_until"], "valid_until") - pub).total_seconds() / 3600.0
        elif item.get("ttl_hours") is not None:
            ttl = _num(item["ttl_hours"], 0, 1e6, "ttl_hours")
        else:
            raise AnalyticsError("нужен valid_until или ttl_hours: срок актуальности оценивает парсер")
        lo, hi = TTL_BOUNDS_H[cat]
        ttl = min(max(ttl, lo, GLOBAL_TTL_H[0]), hi, GLOBAL_TTL_H[1])
        row = {"id": str(item.get("id") or f"{sym}-{int(pub.timestamp())}")[:80], "score": score, "materiality": mat if mat is not None else 1.0,
               "category": cat, "title": str(item.get("title") or "")[:200], "published_at": pub.isoformat(timespec="seconds"),
               "valid_until": (pub + timedelta(hours=ttl)).isoformat(timespec="seconds"), "ttl_hours": round(ttl, 2)}
        with self._lock:
            items = [x for x in self._d.setdefault(sym, {}).setdefault("news", []) if x["id"] != row["id"]]
            items.append(row)
            self._d[sym]["news"] = items[-MAX_ITEMS_PER_SYMBOL:]
            self._purge(now)
            self._save()
        return row

    def set_kronos(self, symbol: str, score, valid_until, tightness=None) -> dict:
        sym = _sym(symbol)
        ks = _num(score, -1, 1, "kronos_score")
        if ks is None:
            raise AnalyticsError("kronos_score обязателен")
        until = _dt(valid_until, "valid_until")
        if until <= self.clock():
            raise AnalyticsError("valid_until уже в прошлом")
        row = {"score": ks, "tightness": _num(tightness, 0, 1e6, "kronos_tightness"), "valid_until": until.isoformat(timespec="seconds")}
        with self._lock:
            self._d.setdefault(sym, {})["kronos"] = row
            self._save()
        return row

    def _purge(self, now: datetime) -> None:
        for sym in list(self._d):
            row = self._d[sym]
            row["news"] = [x for x in row.get("news", []) if _dt(x["valid_until"], "valid_until") > now]
            if "kronos" in row and _dt(row["kronos"]["valid_until"], "valid_until") <= now:
                del row["kronos"]
            if not row.get("news") and "kronos" not in row:
                del self._d[sym]

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._d, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)

    def snapshot(self, symbol: str, now: Optional[datetime] = None) -> dict:
        """Только актуальное на момент now. Просроченное удаляется."""
        now = now or self.clock()
        with self._lock:
            self._purge(now)
            row = self._d.get(symbol.upper(), {})
            out: dict = {}
            news = row.get("news", [])
            if news:
                w = [max(x["materiality"], 0.05) for x in news]
                out.update(news_score=sum(x["score"] * k for x, k in zip(news, w)) / sum(w), news_n=len(news), news_ids=[x["id"] for x in news])
            if "kronos" in row:
                out.update(kronos_score=row["kronos"]["score"], kronos_valid_until=row["kronos"]["valid_until"])
            return out

    def all(self, now: Optional[datetime] = None) -> dict:
        now = now or self.clock()
        with self._lock:
            self._purge(now)
            return json.loads(json.dumps(self._d))
