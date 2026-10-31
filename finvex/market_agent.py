"""
market_agent - диалоговый аналитик рынка. LLM получает Python-пространство с данными бирж и модулями проекта
и сама собирает решение под любой запрос (скрининг, коинтеграция, свои метрики, графики, объяснения).

    python -m finvex.market_agent                      # диалог
    python -m finvex.market_agent "какие акции РФ сейчас в бычьем тренде?"

Файлы сессии: reports/session_<время>/ (графики PNG, notes.json, session.html с диалогом).
Код, который пишет LLM, выполняется локально с правами пользователя (простой запрет на удаление файлов и процессы -
это не полноценная песочница). Ордера не отправляются.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import html
import io
import json
import logging
import math
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from finvex import market_analysis as ma
from finvex import market_stats as st

logger = logging.getLogger("finvex.market_agent")
MSK = ma.MSK
MAX_STEPS = 14
TOOL_RESULT_LIMIT = 9000
HISTORY_LIMIT_CHARS = 110_000
UA = {"User-Agent": "finvex-analysis"}
DENY = re.compile(r"subprocess|os\.system|os\.remove|os\.unlink|shutil\.rmtree|os\.rmdir|\.unlink\(|__import__\(['\"]os")


class ToolError(Exception):
    pass


@dataclass
class Inst:
    symbol: str
    market: str
    name: str
    news_terms: list = field(default_factory=list)


@dataclass
class Session:
    out_dir: Path
    use_kronos: bool = True
    instruments: dict[str, Inst] = field(default_factory=dict)
    frames: dict[tuple, pd.DataFrame] = field(default_factory=dict)
    notes: list[dict[str, str]] = field(default_factory=list)
    last_scan: Optional[pd.DataFrame] = None
    chart_n: int = 0
    turn_images: list[Path] = field(default_factory=list)
    analyzer: Any = None
    sandbox: Any = None

    @property
    def now(self) -> datetime:
        return datetime.now(MSK)

    def chart_path(self, name: str) -> Path:
        self.chart_n += 1
        p = self.out_dir / f"{self.chart_n:02d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)[:60]}.png"
        self.turn_images.append(p)
        return p


# ───────────────────────── инструменты и данные ─────────────────────────
def _moex_lookup(ref: str) -> Optional[Inst]:
    sym = ref.upper()
    if re.fullmatch(r"[A-Z0-9]{1,8}", sym):
        d = ma._http_get_json(f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{sym}.json",
                              {"iss.meta": "off", "iss.only": "securities", "securities.columns": "SECID,SHORTNAME"}, timeout=15)
        rows = (d.get("securities") or {}).get("data") or []
        if rows:
            return Inst(rows[0][0], "moex", rows[0][1] or rows[0][0], [rows[0][0], rows[0][1] or rows[0][0]])
    d = ma._http_get_json("https://iss.moex.com/iss/securities.json", {"q": ref, "iss.meta": "off", "limit": 20,
                          "securities.columns": "secid,shortname,primary_boardid,is_traded"}, timeout=15)
    for secid, short, board, traded in (d.get("securities") or {}).get("data") or []:
        if board == "TQBR" and traded:
            return Inst(secid, "moex", short or secid, [secid, short or secid, ref])
    return None


def _bybit_lookup(ref: str) -> Optional[Inst]:
    if not re.fullmatch(r"[A-Za-z0-9/-]{2,15}", ref):
        return None
    sym = ref.upper().replace("-", "").replace("/", "")
    sym = sym if sym.endswith("USDT") else sym + "USDT"
    d = ma._http_get_json("https://api.bybit.com/v5/market/instruments-info", {"category": "linear", "symbol": sym}, timeout=15)
    return Inst(sym, "bybit", sym, [sym, sym[:-4], ref]) if (d.get("result") or {}).get("list") else None


def get_instrument(sess: Session, ref: str, market: Optional[str] = None) -> Inst:
    ref = str(ref).strip()
    key = ref.upper()
    if key in sess.instruments:
        return sess.instruments[key]
    order = [market] if market in ("moex", "bybit") else (["bybit", "moex"] if key.endswith("USDT") else ["moex", "bybit"])
    for mk in order:
        try:
            inst = (_moex_lookup if mk == "moex" else _bybit_lookup)(ref)
        except Exception as exc:  # noqa: BLE001
            logger.warning("поиск %s на %s: %s", ref, mk, exc)
            inst = None
        if inst is not None:
            sess.instruments[inst.symbol] = sess.instruments[key] = inst
            return inst
    raise ToolError(f"инструмент «{ref}» не найден ни на Мосбирже (TQBR), ни среди USDT-перпетуалов Bybit")


def parse_period(spec: Any, now: datetime) -> tuple[Optional[datetime], Optional[datetime], str]:
    if not isinstance(spec, dict) or not spec.get("type"):
        return None, None, ""
    kind = spec["type"]
    if kind == "relative":
        n, unit = float(spec.get("amount") or 1), spec.get("unit", "days")
        delta = {"hours": timedelta(hours=n), "days": timedelta(days=n), "weeks": timedelta(weeks=n),
                 "months": timedelta(days=30.4 * n), "years": timedelta(days=365.25 * n)}.get(unit, timedelta(days=n))
        return now - delta, now, f"последние {n:g} {unit}"
    if kind == "absolute":
        end = _dt(spec["end"]) if spec.get("end") else now
        if spec.get("end") and len(str(spec["end"]).strip()) <= 10:
            end = end.replace(hour=23, minute=59)
        return _dt(spec["start"]), end, "даты из запроса"
    what = spec.get("what", "today")
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if what == "yesterday":
        return day0 - timedelta(days=1), day0 - timedelta(minutes=1), "вчера"
    if what == "this_week":
        return day0 - timedelta(days=day0.weekday()), now, "текущая неделя"
    if what == "this_month":
        return day0.replace(day=1), now, "текущий месяц"
    if what == "month":
        y, m = int(spec.get("year") or now.year), int(spec.get("month") or now.month)
        return datetime(y, m, 1, tzinfo=MSK), min(datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=MSK) - timedelta(minutes=1), now), f"{m:02d}.{y}"
    return day0, now, "сегодня"


def _dt(x: Any, default: Optional[datetime] = None) -> Optional[datetime]:
    if x is None:
        return default
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=MSK)
    return datetime.fromisoformat(str(x).strip().replace("T", " ")).replace(tzinfo=MSK)


def resolve_period(sess: Session, spec: Any, default_days: float = 30) -> tuple[datetime, datetime, str]:
    try:
        start, end, why = parse_period(spec, sess.now)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"не удалось разобрать период {spec}: {exc}") from exc
    if start is None:
        end, why = sess.now, f"период не задан - последние {default_days:g} дн."
        start = end - timedelta(days=default_days)
    if start >= end:
        start = end - timedelta(days=1)
    return start, min(end, sess.now), why


def candles(sess: Session, inst: Inst, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
    key = (inst.symbol, tf, start.strftime("%Y%m%d%H"), end.strftime("%Y%m%d%H"))
    if key in sess.frames:
        return sess.frames[key]
    fn = ma._bybit_candles if inst.market == "bybit" else ma._moex_candles
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            sess.frames[key] = fn(inst.symbol, tf, start, end)
            return sess.frames[key]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.2 * (attempt + 1))
    raise ToolError(f"не удалось получить свечи {inst.symbol}: {last}")


def daily_frames(sess: Session, insts: list[Inst], days: int) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    end = sess.now
    start = end - timedelta(days=days)
    out: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}

    def one(i: Inst):
        try:
            return i.symbol, candles(sess, i, "1d", start, end), None
        except Exception as exc:  # noqa: BLE001
            return i.symbol, None, str(exc)[:120]

    with ThreadPoolExecutor(max_workers=6) as ex:
        for sym, df, err in ex.map(one, insts):
            if df is not None and len(df) >= 30:
                out[sym] = df.reset_index(drop=True)
            else:
                failed[sym] = err or "мало данных"
    return out, failed


def panel(sess: Session, symbols: list[str], tf: str, start: datetime, end: datetime, col: str = "close") -> pd.DataFrame:
    cols = {}
    for s in symbols:
        inst = get_instrument(sess, s)
        ser = candles(sess, inst, tf, start - timedelta(days=2), end).set_index("timestamps")[col].astype(float)
        if tf == "1d":
            ser.index = pd.DatetimeIndex(ser.index).normalize()
        cols[inst.symbol] = ser[~ser.index.duplicated(keep="last")]
    pn = pd.concat(cols, axis=1).sort_index().ffill(limit=3)
    s0, e0 = start.replace(tzinfo=None), end.replace(tzinfo=None)
    return pn[(pn.index >= s0) & (pn.index <= e0)].dropna(how="all")


def moex_universe(top_n: int) -> list[Inst]:
    data = ma._http_get_json("https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json", {
        "iss.meta": "off", "iss.only": "securities,marketdata",
        "securities.columns": "SECID,SHORTNAME,PREVPRICE,ISSUESIZE", "marketdata.columns": "SECID,VALTODAY"}, timeout=25)
    sec = {r[0]: r for r in (data.get("securities") or {}).get("data", [])}
    val = {r[0]: (r[1] or 0) for r in (data.get("marketdata") or {}).get("data", [])}
    weight = lambda s: float(val.get(s, 0)) or float((sec[s][2] or 0) * (sec[s][3] or 0)) * 1e-6  # noqa: E731
    return [Inst(s, "moex", sec[s][1] or s, [s, sec[s][1] or s]) for s in sorted(sec, key=weight, reverse=True)[:top_n]]


def bybit_universe(top_n: int) -> list[Inst]:
    data = ma._http_get_json("https://api.bybit.com/v5/market/tickers", {"category": "linear"}, timeout=25)
    rows = [r for r in (data.get("result") or {}).get("list", []) if str(r.get("symbol", "")).endswith("USDT")]
    rows.sort(key=lambda r: float(r.get("turnover24h") or 0), reverse=True)
    return [Inst(r["symbol"], "bybit", r["symbol"], [r["symbol"], r["symbol"][:-4]]) for r in rows[:top_n]]


def build_universe(market: str, top_n: int) -> list[Inst]:
    out: list[Inst] = []
    if market in ("moex", "both"):
        out += moex_universe(top_n if market == "moex" else max(top_n // 2, 5))
    if market in ("bybit", "both"):
        out += bybit_universe(top_n if market == "bybit" else max(top_n // 2, 5))
    return out


def google_news(terms: list[str], start: datetime, end: datetime, langs: tuple[str, ...] = ("ru",), limit: int = 60) -> list[dict]:
    import feedparser
    import httpx
    terms = [t for t in dict.fromkeys(terms) if t][:6]
    if not terms:
        return []
    q = " OR ".join(f'"{t}"' if " " in t else t for t in terms) + f" after:{start:%Y-%m-%d} before:{(end + timedelta(days=1)):%Y-%m-%d}"
    seen, items = set(), []
    for lang in langs:
        hl, gl, ceid = {"ru": ("ru", "RU", "RU:ru"), "en": ("en", "US", "US:en")}[lang]
        try:
            with httpx.Client(timeout=20, follow_redirects=True, headers=UA) as cl:
                r = cl.get("https://news.google.com/rss/search", params={"q": q, "hl": hl, "gl": gl, "ceid": ceid})
                r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Google News (%s) недоступен: %s", lang, exc)
            continue
        for e in feedparser.parse(r.content).entries:
            title, pp = str(getattr(e, "title", "")).strip(), getattr(e, "published_parsed", None)
            if not title or title in seen or pp is None:
                continue
            dt = datetime(*pp[:6], tzinfo=timezone.utc)
            if not (start.astimezone(timezone.utc) <= dt <= end.astimezone(timezone.utc)):
                continue
            seen.add(title)
            items.append({"dt": dt, "source": getattr(getattr(e, "source", None), "title", "") or "google_news", "title": title,
                          "link": getattr(e, "link", ""), "summary": re.sub(r"<[^>]+>", " ", str(getattr(e, "summary", "")))[:400]})
    items.sort(key=lambda x: x["dt"], reverse=True)
    return items[:limit]


def score_news(sess: Session, inst: Inst, items: list[dict], analyze: bool, max_llm: int) -> dict[str, Any]:
    rows = [{"time": it["dt"].astimezone(MSK).strftime("%Y-%m-%d %H:%M"), "source": it["source"], "title": it["title"], "link": it["link"]} for it in items]
    out: dict[str, Any] = {"status": "ok", "matched": len(rows), "total_feed_items": len(rows), "terms": inst.news_terms, "items": rows, "score": None}
    if not rows:
        out["note"] = "новостей по запросу за период не найдено"
        return out
    if analyze:
        try:
            if sess.analyzer is None:
                from finvex.analysis.news.deep_dive import NewsDeepDiveAnalyzer
                from finvex.engine.config_loader import merge_llm_section
                cfg = ma._read_yaml("configs/news_agent.yaml")
                sess.analyzer = NewsDeepDiveAnalyzer(merge_llm_section((cfg.get("news_risk") or {}).get("llm", {})))
            from finvex.analysis.news.rss_collector import RawNewsItem
            for i in sorted(set(np.linspace(0, len(rows) - 1, min(max_llm, len(rows))).astype(int).tolist())):
                it = items[i]
                try:
                    r = sess.analyzer.analyze(inst.symbol, RawNewsItem(f"gn{i}", it["source"], it["title"], it["summary"], it["link"], it["dt"].isoformat()))
                except Exception as exc:  # noqa: BLE001
                    r = None
                    logger.warning("оценка новости не удалась: %s", exc)
                if r is not None:
                    rows[i].update(sentiment=float(r.sentiment), severity=float(r.severity), materiality=float(r.materiality),
                                   category=str(r.risk_category), rationale=str(r.rationale)[:200], effective_risk=float(r.effective_risk))
        except Exception as exc:  # noqa: BLE001
            out["llm_error"] = str(exc)[:200]
    scored = [x for x in rows if "sentiment" in x]
    if scored:
        w = np.array([max(0.05, x["materiality"]) for x in scored])
        out["avg_sentiment"] = float(np.average([x["sentiment"] for x in scored], weights=w))
        out["max_effective_risk"] = float(max(x["effective_risk"] for x in scored))
        out["score"] = float(np.clip(out["avg_sentiment"], -1, 1))
        out["scored"] = len(scored)
    return out


# ───────────────────────── готовые рецепты (доступны из Python-пространства как analyze/scan/coint) ─────────────────────────
def recipe_analyze(sess: Session, instrument: str, period: Any = None, timeframe: Optional[str] = None, modules: Optional[list[str]] = None,
                   market: Optional[str] = None, news_terms: Optional[list[str]] = None, analyze_news: bool = True) -> dict:
    inst = get_instrument(sess, instrument, market)
    if news_terms:
        inst.news_terms = list(dict.fromkeys([*news_terms, inst.symbol]))[:10]
    start, end, why = resolve_period(sess, period, 30)
    tf = timeframe if timeframe in ma.ALLOWED_TF else ("1h" if (end - start).days <= 45 else "1d")
    modules = [m.lower() for m in (modules or ["technical", "kronos", "news"])]
    df = ma.add_indicators(candles(sess, inst, tf, start - timedelta(days=ma.warmup_days(tf)), end))
    view = df[(df["timestamps"] >= start.replace(tzinfo=None)) & (df["timestamps"] <= end.replace(tzinfo=None))].reset_index(drop=True)
    if len(view) < 3:
        raise ToolError(f"в периоде слишком мало баров ({len(view)}): биржа могла быть закрыта, расширьте период")
    tech = ma.tech_module(df, view, inst.symbol, tf)
    vol = view["volume"].astype(float)
    tech["volume_ratio_last5"] = float(vol.tail(5).mean() / vol.mean()) if vol.mean() else None
    tech.pop("last_volume_ratio", None)
    if "kronos" in modules and sess.use_kronos:
        kr = ma.kronos_module(df, inst.symbol, tf, 3)
        if kr.get("status") == "ok":
            kr["channel_tightness"] = kr.pop("confidence")
            kr["score"] = 0.7 * float(np.tanh(kr["change"] / (2 * max(kr["volatility_norm"], 1e-4))))
    else:
        kr = {"status": "skipped", "error": "модуль не запрашивался"}
    if "news" in modules:
        nw = score_news(sess, inst, google_news(inst.news_terms or [inst.symbol], start, end, ("ru", "en") if inst.market == "bybit" else ("ru",)), analyze_news, 8)
    else:
        nw = {"status": "skipped", "error": "модуль не запрашивался"}
    comb = ma.combine(dict(kr), tech, nw)
    req = SimpleNamespace(text="", instruments=[inst], tickers=[inst.symbol], start=start, end=end, timeframe=tf, notes=[], parsed_by="agent")
    charts = [sess.chart_path(f"{inst.symbol}_price"), sess.chart_path(f"{inst.symbol}_signals")]
    ma.chart_price(view, inst, req, kr, tech, charts[0])
    ma.chart_summary(inst, comb, nw, charts[1])
    kr_out = {k: v for k, v in kr.items() if not k.startswith("forecast_")}
    if kr.get("status") == "ok":
        kr_out["note"] = "channel_tightness - узость прогнозного канала, НЕ вероятность; выборок Kronos всего 3"
    return {"symbol": inst.symbol, "name": inst.name, "market": inst.market, "period": f"{start:%d.%m.%Y %H:%M} - {end:%d.%m.%Y %H:%M} MSK ({why})",
            "timeframe": tf, "technical": tech, "kronos": kr_out, "news": {k: v for k, v in nw.items() if k != "items"} | {"items": (nw.get("items") or [])[:8]},
            "combined": comb, "charts": [str(c) for c in charts], "note": "combined - эвристика, не торговый сигнал"}


def recipe_scan(sess: Session, market: str = "moex", top_n: int = 30, symbols: Optional[list[str]] = None) -> dict:
    top_n = int(min(max(top_n, 5), 60))
    insts = [get_instrument(sess, s) for s in symbols] if symbols else build_universe(market if market in ("moex", "bybit", "both") else "moex", top_n)
    frames, failed = daily_frames(sess, insts, 420)
    rows = []
    for sym, df in frames.items():
        try:
            rows.append(st.trend_snapshot(df, sym))
        except Exception as exc:  # noqa: BLE001
            failed[sym] = f"ошибка расчёта: {exc}"[:120]
    if not rows:
        raise ToolError(f"не удалось посчитать ни одного инструмента: {failed}")
    ranked = st.rank_trends(rows)
    ranked["name"] = ranked["symbol"].map({i.symbol: i.name for i in insts})
    sess.last_scan = ranked
    breadth = {"instruments": int(len(ranked)), "share_above_sma50": float(ranked["above_sma50"].fillna(False).astype(bool).mean()),
               "share_above_sma200": float(ranked["above_sma200"].fillna(False).astype(bool).mean()), "share_bull_regime": float(ranked["is_bull_regime"].mean()),
               "median_ret_1m": float(ranked["ret_1m"].median()), "median_ret_3m": float(ranked["ret_3m"].median()), "regimes": ranked["regime"].value_counts().to_dict()}
    cols = ["symbol", "name", "bull_score", "regime", "ret_5d", "ret_1m", "ret_3m", "rel_3m_vs_median", "vol_ann", "dd_from_high", "adx", "above_sma50", "above_sma200"]
    return {"as_of": f"{sess.now:%d.%m.%Y %H:%M} MSK", "timeframe": "1d", "breadth": breadth, "top": ranked[cols].head(15).to_dict("records"),
            "bottom": ranked[cols].tail(5).to_dict("records"), "failed": failed,
            "note": "bull_score - прозрачная формула (выше SMA50/200, наклон SMA50, ранги доходности 1м/3м); полный DataFrame: last_scan()"}


def recipe_coint(sess: Session, market: Optional[str] = None, symbols: Optional[list[str]] = None, groups: Optional[dict[str, list[str]]] = None,
                 top_n: int = 25, lookback_days: int = 365, corr_threshold: float = 0.5) -> dict:
    group_syms: dict[str, list[str]] = {}
    if groups:
        for gname, refs in list(groups.items())[:12]:
            ok = []
            for ref in refs[:15]:
                try:
                    ok.append(get_instrument(sess, ref).symbol)
                except ToolError:
                    pass
            if len(ok) >= 2:
                group_syms[gname] = ok
        members = [sess.instruments[s] for g in group_syms.values() for s in g]
    elif symbols:
        members = [get_instrument(sess, s) for s in symbols[:40]]
    else:
        members = build_universe(market if market in ("moex", "bybit") else "moex", int(min(top_n, 40)))
    frames, failed = daily_frames(sess, list({i.symbol: i for i in members}.values()), int(lookback_days) + 10)
    px = pd.concat({s: d.set_index(pd.DatetimeIndex(d["timestamps"]).normalize())["close"].astype(float) for s, d in frames.items()}, axis=1)
    px = px.loc[:, px.notna().mean() >= 0.8].ffill(limit=3).dropna()
    if px.shape[1] < 2 or len(px) < 80:
        raise ToolError(f"мало общих данных: {px.shape[1]} инструментов, {len(px)} общих дней")
    logp = np.log(px)
    mode = "заданные группы"
    group_syms = {g: [s for s in v if s in logp.columns] for g, v in group_syms.items()}
    group_syms = {g: v for g, v in group_syms.items() if len(v) >= 2}
    if not group_syms:
        mode = f"авто-кластеры по корреляции (порог {corr_threshold})"
        group_syms = {f"кластер {i + 1}": c for i, c in enumerate(st.cluster_by_corr(logp.diff().dropna(), corr_threshold))}
    if not group_syms:
        raise ToolError("кластеров не нашлось: снизьте corr_threshold или задайте группы")
    res = st.scan_cointegration(logp, group_syms)
    pairs = [{"pair": f"{r['pair'][0]}/{r['pair'][1]}", "group": r["group"], "p_value": r["pvalue"], "p_bonferroni": r["p_bonferroni"], "adf_tau": r["tau"],
              "beta": r["beta"], "half_life_days": r["half_life"], "spread_z_now": r["z_now"], "corr_returns": r["corr_returns"],
              "beta_stable": r["beta_stable"], "n_obs": r["n_obs"]} for r in res["pairs"][:10]]
    return {"as_of": f"{sess.now:%d.%m.%Y}", "timeframe": "1d", "lookback_days": lookback_days, "grouping": mode, "groups_used": group_syms,
            "group_ranking": res["groups"], "top_pairs": pairs, "pairs_tested": res["tested"], "failed": failed,
            "note": "Engle-Granger на логарифмах цен; смотри p_bonferroni, стабильность beta, half-life. Графики строй сам (logp доступен через closes())"}


# ───────────────────────── Python-пространство (главный инструмент) ─────────────────────────
class Sandbox:
    def __init__(self, sess: Session) -> None:
        self.sess = sess
        plt = ma._plt()
        s = sess

        def h_candles(symbol: str, tf: str = "1d", days: Optional[float] = None, start: Any = None, end: Any = None,
                      market: Optional[str] = None, indicators: bool = False) -> pd.DataFrame:
            e = _dt(end, s.now)
            df = candles(s, get_instrument(s, symbol, market), tf, _dt(start, e - timedelta(days=days or 90)), e)
            return ma.add_indicators(df) if indicators else df

        def h_closes(symbols: list[str], tf: str = "1d", days: float = 90, start: Any = None, end: Any = None, col: str = "close") -> pd.DataFrame:
            e = _dt(end, s.now)
            return panel(s, list(symbols), tf, _dt(start, e - timedelta(days=days)), e, col)

        def h_universe(market: str = "moex", top_n: int = 30) -> list[dict]:
            out = build_universe(market, int(top_n))
            s.instruments.update({i.symbol: i for i in out})
            return [{"symbol": i.symbol, "market": i.market, "name": i.name} for i in out]

        def h_resolve(name: str, market: Optional[str] = None) -> dict:
            i = get_instrument(s, name, market)
            return {"symbol": i.symbol, "market": i.market, "name": i.name}

        def h_regime(df: pd.DataFrame, symbol: str) -> dict:
            from finvex.analysis.regime import compute_regime
            r = compute_regime(df[["high", "low", "close"]], symbol)
            return {k: getattr(r, k, None) for k in ("regime", "adx", "vol_pctile", "close_vs_sma50", "dd_local", "overheat")}

        def h_news(terms: list[str], days: float = 7, start: Any = None, end: Any = None, analyze: bool = False, max_llm: int = 8) -> dict:
            e = _dt(end, s.now)
            langs = ("ru", "en") if any(re.search(r"[A-Za-z]{3,}", t) for t in terms) else ("ru",)
            return score_news(s, Inst("NEWS", "moex", "новости", list(terms)), google_news(list(terms), _dt(start, e - timedelta(days=days)), e, langs), analyze, max_llm)

        def h_analyze(symbol: str, period: Any = None, days: Optional[float] = None, **kw: Any) -> dict:
            if period is None and days:
                period = {"type": "relative", "amount": days, "unit": "days"}
            return recipe_analyze(s, symbol, period, **kw)

        self.ns: dict[str, Any] = {
            "pd": pd, "np": np, "plt": plt, "math": math, "datetime": datetime, "timedelta": timedelta, "MSK": MSK, "re": re, "json": json,
            "NOW": s.now, "candles": h_candles, "closes": h_closes, "universe": h_universe, "resolve": h_resolve, "regime": h_regime,
            "kronos": lambda df, symbol, tf="1d", samples=3: ma.kronos_module(df, symbol, tf, samples), "news": h_news, "analyze": h_analyze,
            "scan": lambda **kw: recipe_scan(s, **kw), "coint": lambda **kw: recipe_coint(s, **kw), "indicators": ma.add_indicators, "rsi": ma.rsi,
            "http_get_json": ma._http_get_json, "engle_granger": st.engle_granger, "cluster_by_corr": st.cluster_by_corr,
            "scan_cointegration": st.scan_cointegration, "trend_snapshot": st.trend_snapshot, "rank_trends": st.rank_trends, "last_scan": lambda: s.last_scan}

    def run(self, code: str, timeout: float = 150.0) -> dict[str, Any]:
        if DENY.search(code):
            return {"error": "код отклонён: удаление файлов и запуск процессов запрещены"}
        out, box = io.StringIO(), {}

        def work() -> None:
            try:
                tree = ast.parse(code)
                last = ast.Expression(tree.body.pop().value) if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(tree, "<agent>", "exec"), self.ns)
                    if last is not None:
                        val = eval(compile(last, "<agent>", "eval"), self.ns)
                        if val is not None:
                            print(val.to_string(max_rows=40, max_cols=20) if isinstance(val, (pd.DataFrame, pd.Series)) else repr(val))
            except BaseException as exc:  # noqa: BLE001
                box["error"] = f"{type(exc).__name__}: {exc}\n" + "\n".join(traceback.format_exc().splitlines()[-6:])

        th = threading.Thread(target=work, daemon=True)
        th.start()
        th.join(timeout)
        res: dict[str, Any] = {}
        if th.is_alive():
            res["error"] = f"таймаут {timeout:.0f} c: уменьшите объём (меньше инструментов/период) или разбейте на шаги"
        elif "error" in box:
            res["error"] = box["error"]
        text = out.getvalue()
        res["stdout"] = text[-6000:]
        charts = flush_figs(self.sess)
        if charts:
            res["charts"] = charts
        return res


def flush_figs(sess: Session) -> list[str]:
    plt = ma._plt()
    paths = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        title = next((a.get_title() for a in fig.axes if a.get_title()), "") or (fig._suptitle.get_text() if fig._suptitle else "")
        p = sess.chart_path(f"fig_{title[:30]}" if title else "fig")
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(str(p))
    return paths


def tool_python(sess: Session, code: str) -> dict:
    if sess.sandbox is None:
        sess.sandbox = Sandbox(sess)
    return sess.sandbox.run(code)


def tool_save_note(sess: Session, kind: str, text: str) -> dict:
    sess.notes.append({"kind": kind, "text": text, "at": f"{sess.now:%Y-%m-%d %H:%M}"})
    (sess.out_dir / "notes.json").write_text(json.dumps(sess.notes, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved": True, "total_notes": len(sess.notes)}


TOOLS = [
    {"type": "function", "function": {"name": "python", "description": (
        "Выполнить Python в постоянном рабочем пространстве (переменные сохраняются между вызовами). Решает любую аналитическую задачу: "
        "данные, индикаторы, скрининг, статистика, коинтеграция, свои метрики, графики matplotlib (все открытые фигуры сохраняются автоматически). "
        "print() только нужное. Доступно: candles(symbol, tf, days|start,end, indicators=False)->OHLCV DataFrame; closes([symbols], tf, days)->DataFrame цен; "
        "universe(market, top_n)->ликвидные инструменты; resolve(name); indicators(df); rsi(series); regime(df, symbol); kronos(df, symbol, tf); "
        "news(terms, days|start,end, analyze=False); http_get_json(url, params); engle_granger(log_y, log_x); cluster_by_corr(returns, thr); "
        "scan_cointegration(logp, groups); trend_snapshot(df, symbol); rank_trends(rows); готовые рецепты analyze(symbol, days|period), "
        "scan(market, top_n), coint(market|groups|symbols). pd, np, plt, datetime, NOW тоже доступны."),
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "save_note", "description": "Запомнить тактику, риск-лимиты или предпочтения пользователя.",
        "parameters": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["tactic", "risk", "preference", "other"]}, "text": {"type": "string"}}, "required": ["kind", "text"]}}}]
DISPATCH: dict[str, Callable[..., dict]] = {"python": tool_python, "save_note": tool_save_note}


def compact(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(f"{obj:.4g}")
    if isinstance(obj, (np.floating, np.integer)):
        return compact(obj.item())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): compact(v) for k, v in obj.items() if not str(k).startswith("forecast_")}
    if isinstance(obj, (list, tuple)):
        return [compact(v) for v in obj[:40]]
    return obj[:6000] if isinstance(obj, str) else obj


SYSTEM_PROMPT = """Ты - квант-аналитик в консоли трейдера (Мосбиржа и Bybit). Сейчас {now} (MSK). Говори по-русски, веди живой диалог, помни контекст.
У тебя есть Python-пространство (инструмент python) с данными бирж, модулями проекта (Kronos, режим рынка, новости) и matplotlib. Любую задачу решай, САМА составляя код из этих кирпичиков, а не ожидая готовой функции под задачу.
Как работать:
- Реши, какие данные нужны (инструменты, период, таймфрейм), загрузи, посчитай, построй график, затем объясни. Несколько вызовов подряд - нормально; переменные сохраняются, не перезагружай данные без нужды.
- Универсум не вшит: акции/монеты бери через universe() или публичные API (http_get_json), группы активов (секторы и т.п.) определяй своими знаниями и проверяй через resolve().
- Графики выбирай и оформляй сам под вопрос (подписи и заголовки по-русски, легенда). Просьбы «добавь/убери/другой период/другой график» - пересчитай и перерисуй. Фигуры сохраняются автоматически; пути из charts перечисляй в ответе.
- Рецепты analyze(), scan(), coint() можно брать как есть или как образец, но можно писать своё. Объяснения («почему», «что значит») давай по уже полученным данным, без лишних вызовов.
- Если код упал - прочитай ошибку, исправь, повтори. Код пользователю не показывай, пока не попросит; показывай результаты и выводы.
Правила достоверности:
- Числа о рынке только из результатов кода. Не выдумывай; нет данных - скажи. Всегда указывай период, таймфрейм и дату данных.
- Kronos channel_tightness - не вероятность. Композитные оценки - эвристики. В коинтеграции учитывай множественное тестирование (Бонферрони), стабильность коэффициента и half-life. Новости из RSS неполные.
- Ордера не отправляются: только аналитика. Тактику, риск-лимиты и предпочтения пользователя сохраняй через save_note и учитывай дальше; бэктест и исполнение - следующий этап.
- Отвечай кратко и структурно, в конце предложи 1-2 следующих шага. Не инвестиционная рекомендация - упоминай коротко.
{notes}"""


# ───────────────────────── агент ─────────────────────────
class Agent:
    def __init__(self, sess: Session, verbose: bool = True) -> None:
        self.sess, self.verbose = sess, verbose
        self.messages: list[dict[str, Any]] = []
        self.transcript: list[dict[str, Any]] = []
        self.summary = ""

    def _system(self) -> dict:
        notes = ""
        if self.sess.notes:
            notes = "Заметки пользователя (учитывай): " + "; ".join(f"[{n['kind']}] {n['text']}" for n in self.sess.notes)
        if self.summary:
            notes += f"\nКраткое содержание предыдущей части диалога: {self.summary}"
        return {"role": "system", "content": SYSTEM_PROMPT.format(now=f"{self.sess.now:%d.%m.%Y %H:%M}", notes=notes)}

    def _chat(self, messages: list[dict], tools: Optional[list] = None):
        import openai
        from finvex.engine.config_loader import load_llm_shared
        shared = load_llm_shared()
        if not shared.get("api_key"):
            raise RuntimeError("нет ключа LLM: задайте DEEPSEEK_API_KEY или configs/llm_shared.yaml")
        client = openai.OpenAI(api_key=shared["api_key"], base_url=shared.get("base_url") or "https://api.deepseek.com", timeout=120)
        kw: dict[str, Any] = dict(model=shared.get("model", "deepseek-flash"), messages=messages, temperature=0.3, max_tokens=4000)
        if tools:
            kw["tools"] = tools
        try:
            return client.chat.completions.create(**kw, extra_body={"thinking": {"type": "disabled"}}).choices[0].message
        except openai.BadRequestError:
            return client.chat.completions.create(**kw).choices[0].message

    def _run_tool(self, name: str, raw_args: str) -> str:
        fn = DISPATCH.get(name)
        if fn is None:
            return json.dumps({"error": f"нет инструмента {name}"}, ensure_ascii=False)
        try:
            args = json.loads(raw_args or "{}")
            if self.verbose:
                shown = args.get("code", json.dumps(args, ensure_ascii=False))
                print(f"  [{name}] {str(shown).strip().splitlines()[0][:110] if str(shown).strip() else ''} ...", flush=True)
            t0 = time.time()
            res = fn(self.sess, **args)
            if self.verbose:
                print(f"  [готово] {time.time() - t0:.1f} c" + (" - ошибка, модель исправит" if isinstance(res, dict) and res.get("error") else ""), flush=True)
        except ToolError as exc:
            res = {"error": str(exc)}
        except TypeError as exc:
            res = {"error": f"неверные аргументы: {exc}"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("tool %s failed", name)
            res = {"error": f"{type(exc).__name__}: {exc}"}
        text = json.dumps(compact(res), ensure_ascii=False)
        return text if len(text) <= TOOL_RESULT_LIMIT else text[:TOOL_RESULT_LIMIT] + '..."[обрезано]"'

    def _compact_history(self) -> None:
        if sum(len(str(m.get("content") or "")) for m in self.messages) <= HISTORY_LIMIT_CHARS:
            return
        head, tail = self.messages[:-12], self.messages[-12:]
        while tail and tail[0].get("role") == "tool":
            head.append(tail.pop(0))
        digest = "\n".join(f"{m['role']}: {str(m.get('content') or '')[:600]}" for m in head if m["role"] in ("user", "assistant") and m.get("content"))
        try:
            msg = self._chat([{"role": "system", "content": "Сожми диалог в краткую сводку (до 1200 символов): вопросы пользователя, инструменты/периоды, важные числа и выводы, тактика."},
                              {"role": "user", "content": (self.summary + "\n" + digest)[-12000:]}])
            self.summary = (msg.content or "")[:1500]
        except Exception as exc:  # noqa: BLE001
            logger.warning("сжатие истории не удалось: %s", exc)
            self.summary = (self.summary + " " + digest)[-1500:]
        self.messages = tail

    def ask(self, text: str) -> str:
        self.sess.turn_images = []
        self.messages.append({"role": "user", "content": text})
        answer = ""
        for _ in range(MAX_STEPS):
            msg = self._chat([self._system(), *self.messages], TOOLS)
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]
            self.messages.append(entry)
            if not msg.tool_calls:
                answer = msg.content or ""
                break
            for tc in msg.tool_calls:
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": self._run_tool(tc.function.name, tc.function.arguments)})
        else:
            answer = "Достигнут лимит шагов на один вопрос. Уточните или разбейте запрос - продолжу с уже полученными данными."
            self.messages.append({"role": "assistant", "content": answer})
        self.transcript.append({"q": text, "a": answer, "images": list(self.sess.turn_images)})
        self._compact_history()
        self.save_html()
        return answer

    def save_html(self) -> Path:
        e, body = html.escape, []
        for t in self.transcript:
            body.append(f"<div class='q'>{e(t['q'])}</div><div class='a'>{e(t['a'])}</div>")
            body += [f"<img src='data:image/png;base64,{base64.b64encode(Path(p).read_bytes()).decode()}'>" for p in t["images"] if Path(p).exists()]
        css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:20px auto;padding:0 14px;color:#222}"
               ".q{background:#e8f0fe;padding:8px 12px;border-radius:8px;margin-top:22px;font-weight:600}"
               ".a{white-space:pre-wrap;padding:8px 12px;line-height:1.45}img{max-width:100%;margin:6px 0}")
        path = self.sess.out_dir / "session.html"
        path.write_text(f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>Диалог</title><style>{css}</style></head><body>{''.join(body)}</body></html>", encoding="utf-8")
        return path


def main(argv: Optional[list[str]] = None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description="Диалоговый аналитик рынка")
    ap.add_argument("query", nargs="*", help="первый вопрос (необязательно)")
    ap.add_argument("--no-kronos", action="store_true")
    ap.add_argument("--open", action="store_true", help="открывать графики сразу (Windows)")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--once", action="store_true", help="ответить на один вопрос и выйти")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("finvex.market_analysis").setLevel(logging.WARNING)
    out = Path(a.out) / f"session_{datetime.now(MSK):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    agent = Agent(Session(out_dir=out, use_kronos=not a.no_kronos))
    print(f"Диалог с аналитиком. Команды: /new - очистить контекст, /html - открыть session.html, /exit - выход.\nФайлы: {out}\n")
    pending = " ".join(a.query).strip()
    while True:
        if pending:
            q, pending = pending, ""
            print(f"Вы: {q}")
        else:
            try:
                q = input("\nВы: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        if not q:
            continue
        if q.lower() in ("/exit", "/quit", "exit", "выход"):
            break
        if q.lower() == "/new":
            agent.messages, agent.summary = [], ""
            print("Контекст очищен (заметки о тактике и загруженные данные сохранены).")
            continue
        if q.lower() == "/html":
            p = agent.save_html()
            os.startfile(str(p)) if hasattr(os, "startfile") else print(p)  # type: ignore[attr-defined]
            continue
        try:
            ans = agent.ask(q)
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка: {exc}")
            continue
        print(f"\nАналитик: {ans}")
        for p in agent.sess.turn_images:
            print(f"  [график] {p}")
            if a.open and hasattr(os, "startfile"):
                os.startfile(str(p))  # type: ignore[attr-defined]
        if a.once:
            break
    print(f"\nДиалог сохранён: {agent.save_html()}")


if __name__ == "__main__":
    main()
