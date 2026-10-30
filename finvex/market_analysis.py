"""
Полный анализ рынка по текстовому запросу пользователя.

    python -m finvex.market_analysis "проанализируй Сбер и Газпром за последние 2 недели"
    python -m finvex.market_analysis "что с биткоином с 01.10.2026 по 15.10.2026"
    python -m finvex.market_analysis            # интерактивный режим

Как понимается запрос (никаких зашитых списков тикеров):
  1. LLM извлекает инструменты (тикер или название компании/монеты), период и поисковые термы новостей.
  2. Каждый инструмент ПРОВЕРЯЕТСЯ по живым источникам: MOEX ISS (поиск по названию/тикеру)
     и Bybit instruments-info. Выдуманный LLM тикер отбрасывается.
  3. Период LLM отдаёт структурой (N единиц / даты / календарный месяц), даты считает код.
  4. Без ключа LLM работает запасной разбор: период по правилам, инструменты через поиск MOEX ISS.
Три модуля на инструмент: Kronos (прогноз), режим и техника, новости (RSS + LLM-оценка).
Итог: графики PNG + report.html / report.md / result.json в reports/<тикеры>_<время>/.
Ордера НЕ отправляются: это только аналитика.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("finvex.market_analysis")
MSK = timezone(timedelta(hours=3))
ALLOWED_TF = ("15m", "1h", "4h", "1d")
MAX_INSTRUMENTS = 5
UA = {"User-Agent": "finvex-analysis"}


# ───────────────────────── модели ─────────────────────────
@dataclass
class Instrument:
    symbol: str
    market: str                      # "moex" | "bybit"
    name: str = ""
    news_terms: list[str] = field(default_factory=list)
    verified: bool = True


@dataclass
class Request:
    text: str
    instruments: list[Instrument]
    start: datetime
    end: datetime
    timeframe: str
    notes: list[str] = field(default_factory=list)
    parsed_by: str = "rules"

    @property
    def tickers(self) -> list[str]:
        return [i.symbol for i in self.instruments]


def is_crypto(symbol: str) -> bool:
    return symbol.upper().endswith(("USDT", "USDC"))


# ───────────────────────── LLM ─────────────────────────
def llm_json(system: str, user: str, max_tokens: int = 1500, timeout: int = 60) -> Optional[dict]:
    """JSON-ответ LLM (ключ и модель из configs/llm_shared.yaml / DEEPSEEK_API_KEY). None — LLM недоступен."""
    try:
        import openai
        from finvex.engine.config_loader import load_llm_shared
        shared = load_llm_shared()
        key = shared.get("api_key")
        if not key:
            return None
        client = openai.OpenAI(api_key=key, base_url=shared.get("base_url") or "https://api.deepseek.com", timeout=timeout)
        kw = dict(model=shared.get("model", "deepseek-flash"), max_tokens=max_tokens, temperature=0.1,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                  response_format={"type": "json_object"})
        try:
            r = client.chat.completions.create(**kw, extra_body={"thinking": {"type": "disabled"}})
        except openai.BadRequestError:
            r = client.chat.completions.create(**kw)
        txt = (r.choices[0].message.content or "").strip()
        txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
        return json.loads(txt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM недоступен: %s", exc)
        return None


UNDERSTAND_SYSTEM = """Ты разбираешь запрос трейдера на анализ рынка. Верни ТОЛЬКО json такого вида:
{"instruments": [{"query": "как назвал пользователь", "symbol": "тикер или null", "market": "moex" | "bybit",
                  "news_terms": ["4-8 названий и псевдонимов для поиска в новостях, рус. и англ."]}],
 "period": null
   | {"type": "relative", "amount": число, "unit": "hours"|"days"|"weeks"|"months"|"years"}
   | {"type": "absolute", "start": "YYYY-MM-DD HH:MM", "end": "YYYY-MM-DD HH:MM" или null}
   | {"type": "calendar", "what": "today"|"yesterday"|"this_week"|"this_month"|"month", "year": число или null, "month": 1-12 или null},
 "timeframe": "15m"|"1h"|"4h"|"1d" или null}
Правила: акции Мосбиржи - market "moex", symbol = secid (SBER, GAZP...); криптовалюты - market "bybit",
symbol вида BTCUSDT (линейные USDT-перпетуалы Bybit). Если не уверен в тикере - symbol null, в query название:
тикер найдут поиском по бирже. Не выдумывай инструменты, которых нет в запросе. Если периода нет - period null.
Относительные периоды НЕ переводи в даты, возвращай amount и unit. Сейчас: {now} (MSK)."""


def llm_understand(text: str, now: datetime) -> Optional[dict]:
    data = llm_json(UNDERSTAND_SYSTEM.replace("{now}", f"{now:%Y-%m-%d %H:%M}"), text, max_tokens=700, timeout=40)
    return data if isinstance(data, dict) else None


# ───────────────────────── проверка инструментов по живым источникам ─────────────────────────
def _http_get_json(url: str, params: dict, timeout: float = 15.0) -> Any:
    import httpx
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout, headers=UA) as cl:
                r = cl.get(url, params=params)
                r.raise_for_status()
                return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(str(last))


_ISS_CACHE: dict[str, list[dict]] = {}


def iss_search(query: str) -> list[dict]:
    """Поиск бумаг на Мосбирже по тикеру или названию (MOEX ISS /securities?q=)."""
    q = query.strip()
    if q in _ISS_CACHE:
        return _ISS_CACHE[q]
    data = _http_get_json("https://iss.moex.com/iss/securities.json", {
        "q": q, "iss.meta": "off", "iss.only": "securities", "limit": 20,
        "securities.columns": "secid,shortname,name,is_traded,primary_boardid,group"})
    block = data.get("securities") or {}
    cols, rows = block.get("columns") or [], block.get("data") or []
    out = [dict(zip(cols, r)) for r in rows]
    _ISS_CACHE[q] = out
    return out


def _is_tradable_share(row: dict) -> bool:
    return bool(row.get("is_traded")) and row.get("group") == "stock_shares" and row.get("primary_boardid") == "TQBR"


def resolve_moex(*queries: Optional[str]) -> Optional[Instrument]:
    """Первая найденная торгуемая акция TQBR: сначала точное совпадение secid, затем совпадение по названию."""
    for q in queries:
        if not q:
            continue
        try:
            rows = iss_search(q)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MOEX ISS поиск «%s» не удался: %s", q, exc)
            continue
        low = q.lower().replace("ё", "е")
        for row in rows:
            if _is_tradable_share(row) and str(row.get("secid", "")).upper() == q.upper():
                return Instrument(row["secid"], "moex", row.get("shortname") or "")
        for row in rows:
            names = f"{row.get('shortname', '')} {row.get('name', '')}".lower().replace("ё", "е")
            if _is_tradable_share(row) and low in names:
                return Instrument(row["secid"], "moex", row.get("shortname") or "")
        shares = [r for r in rows if _is_tradable_share(r)]
        if len(shares) == 1:
            return Instrument(shares[0]["secid"], "moex", shares[0].get("shortname") or "")
    return None


def resolve_bybit(symbol: str) -> Optional[Instrument]:
    sym = re.sub(r"[^A-Za-z0-9]", "", symbol or "").upper()
    if not sym:
        return None
    for cand in dict.fromkeys([sym, sym + "USDT"] if not sym.endswith("USDT") else [sym]):
        try:
            data = _http_get_json("https://api.bybit.com/v5/market/instruments-info", {"category": "linear", "symbol": cand})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bybit instruments-info не ответил (%s) - символ %s принят без проверки", exc, cand)
            return Instrument(cand, "bybit", cand, verified=False) if cand.endswith("USDT") else None
        lst = (data.get("result") or {}).get("list") or []
        if lst and lst[0].get("status") == "Trading":
            return Instrument(cand, "bybit", cand)
    return None


def resolve_instrument(entry: dict) -> Optional[Instrument]:
    symbol = (entry.get("symbol") or "").strip() or None
    query = (entry.get("query") or "").strip() or None
    market = str(entry.get("market") or "").lower()
    if market not in ("moex", "bybit"):
        market = "bybit" if (symbol or "").upper().endswith("USDT") else "moex"
    inst = resolve_bybit(symbol or query or "") if market == "bybit" else resolve_moex(symbol, query)
    if inst is None and market == "moex" and symbol and symbol.upper().endswith("USDT"):
        inst = resolve_bybit(symbol)
    if inst is None:
        return None
    terms = [str(t).strip() for t in (entry.get("news_terms") or []) if str(t).strip()]
    inst.news_terms = list(dict.fromkeys([*terms, inst.symbol, inst.name, query or ""]))
    inst.news_terms = [t for t in inst.news_terms if t][:10]
    return inst


# ───────────────────────── период ─────────────────────────
def _start_of_day(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def period_from_spec(spec: Any, now: datetime) -> tuple[Optional[datetime], Optional[datetime], str]:
    """Структура периода от LLM -> даты. Арифметику делает код, не LLM."""
    if not isinstance(spec, dict):
        return None, None, ""
    kind = spec.get("type")
    try:
        if kind == "relative":
            n = float(spec["amount"])
            days = {"hours": 1 / 24, "days": 1, "weeks": 7, "months": 30, "years": 365}[spec["unit"]]
            if n <= 0:
                return None, None, ""
            return now - timedelta(days=n * days), now, f"последние {spec['amount']} {spec['unit']}"
        if kind == "absolute":
            def p(s: str) -> datetime:
                s = str(s).strip()
                return datetime.fromisoformat(s if len(s) > 10 else s + " 00:00").replace(tzinfo=MSK)
            start = p(spec["start"])
            if spec.get("end"):
                end = p(spec["end"])
                if len(str(spec["end"]).strip()) <= 10:
                    end = end + timedelta(days=1) - timedelta(seconds=1)
            else:
                end = now
            return min(start, end), min(max(start, end), now), "даты из запроса"
        if kind == "calendar":
            what = spec.get("what")
            if what == "today":
                return _start_of_day(now), now, "сегодня"
            if what == "yesterday":
                s = _start_of_day(now) - timedelta(days=1)
                return s, _start_of_day(now) - timedelta(seconds=1), "вчера"
            if what == "this_week":
                return _start_of_day(now - timedelta(days=now.weekday())), now, "текущая неделя"
            if what == "this_month":
                return _start_of_day(now).replace(day=1), now, "текущий месяц"
            if what == "month":
                mon = int(spec["month"])
                year = int(spec["year"]) if spec.get("year") else (now.year if mon <= now.month else now.year - 1)
                s = datetime(year, mon, 1, tzinfo=MSK)
                e = datetime(year + (mon == 12), mon % 12 + 1, 1, tzinfo=MSK) - timedelta(seconds=1)
                return s, min(e, now), f"календарный месяц {mon:02d}.{year}"
    except (KeyError, ValueError, TypeError):
        return None, None, ""
    return None, None, ""


_MONTH_STEMS = {"январ": 1, "феврал": 2, "март": 3, "апрел": 4, "мая": 5, "май": 5, "июн": 6, "июл": 7,
                "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12}
_UNITS = [(r"час\w*|ч\b", 1 / 24), (r"дн\w*|день|сут\w*", 1), (r"недел\w*|нед\b", 7),
          (r"месяц\w*|мес\b", 30), (r"квартал\w*", 91), (r"год\w*|лет\b", 365)]


def parse_period_rules(text: str, now: datetime) -> tuple[Optional[datetime], Optional[datetime], str, str]:
    """Запасной разбор периода без LLM. Возвращает (start, end, пояснение, текст без фразы периода)."""
    low = text.lower().replace("ё", "е")
    dates: list[datetime] = []

    def add(y: int, mo: int, d: int, explicit_year: bool) -> None:
        try:
            dt = datetime(y, mo, d, tzinfo=MSK)
            if not explicit_year and dt > now:
                dt = dt.replace(year=dt.year - 1)
            dates.append(dt)
        except ValueError:
            pass

    for m in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", low):
        add(int(m.group(1)), int(m.group(2)), int(m.group(3)), True)
    stripped = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", " ", low)
    for m in re.finditer(r"(?<![\d.])(\d{1,2})[./](\d{1,2})(?:[./](\d{4}|\d{2}))?(?!\d)", stripped):
        y = int(m.group(3)) if m.group(3) else now.year
        add(y + 2000 if y < 100 else y, int(m.group(2)), int(m.group(1)), bool(m.group(3)))
    if len(dates) >= 2:
        a, b = min(dates), max(dates)
        return a, min(b + timedelta(days=1) - timedelta(seconds=1), now), "явные даты в запросе", low
    if len(dates) == 1:
        d = dates[0]
        if re.search(r"\b(с|от|начиная)\b", low):
            return d, now, "с указанной даты до сейчас", low
        return d, min(d + timedelta(days=1) - timedelta(seconds=1), now), "один указанный день", low

    m = re.search(r"(?:за|последн\w*|прошл\w*|ближайш\w*)?\s*(\d+(?:[.,]\d+)?)\s*(" + "|".join(u for u, _ in _UNITS) + ")", low)
    if m:
        n = float(m.group(1).replace(",", "."))
        for pat, days in _UNITS:
            if re.fullmatch(pat, m.group(2)):
                return now - timedelta(days=n * days), now, f"последние {m.group(1)} {m.group(2)}", low.replace(m.group(0), " ")
    if re.search(r"полгода|полугод", low):
        return now - timedelta(days=182), now, "последние полгода", re.sub(r"полгода|полугод\w*", " ", low)
    m = re.search(r"за\s+(?:последн\w+\s+|прошл\w+\s+)?(час|день|сутки|неделю|месяц|квартал|год)\b", low)
    if m:
        days = {"час": 1 / 24, "день": 1, "сутки": 1, "неделю": 7, "месяц": 30, "квартал": 91, "год": 365}[m.group(1)]
        return now - timedelta(days=days), now, f"за {m.group(1)}", low.replace(m.group(0), " ")
    for word, fn, why in (
        (r"\bсегодня\b", lambda: (_start_of_day(now), now), "сегодня"),
        (r"\bвчера\b", lambda: (_start_of_day(now) - timedelta(days=1), _start_of_day(now) - timedelta(seconds=1)), "вчера"),
        (r"(?:на этой|эту|за эту|текущ\w+) недел\w*", lambda: (_start_of_day(now - timedelta(days=now.weekday())), now), "текущая неделя"),
        (r"(?:в этом|этот|за этот|текущ\w+) месяц\w*", lambda: (_start_of_day(now).replace(day=1), now), "текущий месяц"),
    ):
        if re.search(word, low):
            s, e = fn()
            return s, e, why, re.sub(word, " ", low)
    for stem, mon in _MONTH_STEMS.items():
        m = re.search(r"\b(?:за|в|на)\s+" + stem + r"\w*(?:\s+(\d{4}))?", low)
        if m:
            year = int(m.group(1)) if m.group(1) else (now.year if mon <= now.month else now.year - 1)
            s = datetime(year, mon, 1, tzinfo=MSK)
            e = datetime(year + (mon == 12), mon % 12 + 1, 1, tzinfo=MSK) - timedelta(seconds=1)
            return s, min(e, now), f"календарный месяц {mon:02d}.{year}", low.replace(m.group(0), " ")
    return None, None, "", low


# слова-связки: не названия бумаг (служебная лексика запроса, не список тикеров)
_GLUE = {"проанализируй", "анализ", "анализируй", "покажи", "дай", "что", "как", "рынок", "рынка", "акции", "акция",
         "акций", "цена", "цены", "график", "прогноз", "новости", "последние", "последних", "для", "про", "мне",
         "пожалуйста", "и", "или", "по", "на", "за", "с", "от", "до", "в", "к", "о", "об", "же", "ли"}


def fallback_instruments(text_wo_period: str, original: str) -> list[Instrument]:
    """Без LLM: латинские токены-тикеры и русские названия проверяются поиском MOEX ISS."""
    found: list[Instrument] = []
    seen: set[str] = set()
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", original)
    tokens.sort(key=lambda t: (not t.isupper(), t.lower()))
    low_text = text_wo_period.lower()
    lookups = 0
    for tok in tokens:
        if tok.lower() in _GLUE or tok.lower() not in low_text or lookups >= 8 or len(found) >= MAX_INSTRUMENTS:
            continue
        lookups += 1
        if tok.upper().endswith("USDT"):
            inst = resolve_bybit(tok)
        else:
            inst = None
            for q in dict.fromkeys([tok, tok[:-2] if len(tok) > 5 else "", tok[:-1] if len(tok) > 4 else ""]):
                if q and (inst := resolve_moex(q)) is not None:
                    low = f"{inst.name} {inst.symbol}".lower().replace("ё", "е")
                    if q.lower().replace("ё", "е") in low or q.upper() == inst.symbol:
                        break
                    inst = None
        if inst and inst.symbol not in seen:
            seen.add(inst.symbol)
            inst.news_terms = list(dict.fromkeys([tok, inst.symbol, inst.name]))
            found.append(inst)
    return found


def choose_timeframe(span_days: float) -> str:
    return "1h" if span_days <= 45 else "1d"


def parse_request(text: str, now: Optional[datetime] = None, timeframe: Optional[str] = None,
                  use_llm: bool = True) -> Request:
    now = now or datetime.now(MSK)
    notes: list[str] = []
    parsed_by = "rules"
    instruments: list[Instrument] = []
    start = end = None
    why = ""
    tf_llm: Optional[str] = None

    data = llm_understand(text, now) if use_llm else None
    if data:
        parsed_by = "llm"
        seen: set[str] = set()
        for entry in (data.get("instruments") or [])[:MAX_INSTRUMENTS]:
            if not isinstance(entry, dict):
                continue
            inst = resolve_instrument(entry)
            if inst is None:
                notes.append(f"инструмент «{entry.get('query') or entry.get('symbol')}» не найден на бирже - пропущен")
            elif inst.symbol not in seen:
                seen.add(inst.symbol)
                instruments.append(inst)
                if not inst.verified:
                    notes.append(f"{inst.symbol}: Bybit не ответил, символ принят без проверки")
        start, end, why = period_from_spec(data.get("period"), now)
        tf_llm = data.get("timeframe") if data.get("timeframe") in ALLOWED_TF else None
    elif use_llm:
        notes.append("LLM недоступен - запрос разобран запасными правилами и поиском MOEX ISS")

    if start is None:
        start, end, why, rest = parse_period_rules(text, now)
    else:
        rest = text
    if not instruments and not data:
        instruments = fallback_instruments(rest, text)
        if instruments:
            parsed_by = "rules+iss"
    if start is None:
        start, end, why = now - timedelta(days=30), now, "период не указан - взяты последние 30 дней"
        notes.append(why)
    end = min(end, now)
    if start >= end:
        start = end - timedelta(days=1)
        notes.append("начало периода было позже конца - взят последний день")
    if not instruments:
        raise ValueError("Не нашёл ни одного торгуемого инструмента в запросе. Пример: «проанализируй Сбер за 2 недели» или «биткоин за месяц».")
    tf = timeframe or tf_llm or choose_timeframe((end - start).total_seconds() / 86400)
    notes.insert(0, f"период: {start:%d.%m.%Y %H:%M} - {end:%d.%m.%Y %H:%M} MSK ({why}); таймфрейм {tf}")
    return Request(text, instruments[:MAX_INSTRUMENTS], start, end, tf, notes, parsed_by)


# ───────────────────────── данные ─────────────────────────
def warmup_days(tf: str) -> int:
    return 420 if tf == "1d" else 60


def _bybit_candles(symbol: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
    interval = {"1h": "60", "1d": "D", "15m": "15", "4h": "240"}[tf]
    cur_end = int(end.timestamp() * 1000)
    start_ms = int(start.timestamp() * 1000)
    rows: list = []
    for _ in range(30):
        data = _http_get_json("https://api.bybit.com/v5/market/kline", {
            "category": "linear", "symbol": symbol, "interval": interval,
            "start": f"{start_ms:d}", "end": f"{cur_end:d}", "limit": 1000}, timeout=20)
        lst = (data.get("result") or {}).get("list") or []
        if not lst:
            break
        rows.extend(lst)
        earliest = int(lst[-1][0])
        if earliest <= start_ms or len(lst) < 1000:
            break
        cur_end = earliest - 1
    if not rows:
        raise RuntimeError(f"Bybit не вернул свечи по {symbol}")
    df = pd.DataFrame(rows, columns=["ms", "open", "high", "low", "close", "volume", "amount"]).drop_duplicates("ms")
    df["timestamps"] = pd.to_datetime(df["ms"].astype("int64"), unit="ms", utc=True).dt.tz_convert(MSK).dt.tz_localize(None)
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = df[c].astype(float)
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].sort_values("timestamps").reset_index(drop=True)


def _moex_candles(symbol: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
    from finvex.analysis.kronos.bridge import to_kronos_frame
    from finvex.venues.moex import MoexISSProvider
    prov = MoexISSProvider()
    try:
        series = prov.get_candles(symbol, tf, start.astimezone(timezone.utc), end.astimezone(timezone.utc))
    finally:
        try:
            prov.close()
        except Exception:  # noqa: BLE001
            pass
    return to_kronos_frame(series, require_quality_ok=False)


def load_market_data(inst: Instrument, req: Request) -> pd.DataFrame:
    fetch_start = req.start - timedelta(days=warmup_days(req.timeframe))
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            fn = _bybit_candles if inst.market == "bybit" else _moex_candles
            return fn(inst.symbol, req.timeframe, fetch_start, req.end)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"не удалось получить свечи {inst.symbol}: {last}")


# ───────────────────────── модуль 2: режим и техника ─────────────────────────
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["sma20"] = d["close"].rolling(20).mean()
    d["sma50"] = d["close"].rolling(50).mean()
    sd = d["close"].rolling(20).std()
    d["bb_up"], d["bb_lo"] = d["sma20"] + 2 * sd, d["sma20"] - 2 * sd
    d["rsi"] = rsi(d["close"])
    d["atr"] = atr(d)
    return d


def bars_per_year(tf: str, crypto: bool) -> float:
    per_day = {"15m": 96, "1h": 24, "4h": 6, "1d": 1}[tf]
    if tf != "1d" and not crypto:
        per_day = {"15m": 56, "1h": 14, "4h": 3}[tf]
    return per_day * (365.0 if crypto else 252.0)


def tech_module(full: pd.DataFrame, view: pd.DataFrame, symbol: str, tf: str) -> dict[str, Any]:
    from finvex.analysis.regime import compute_regime, signal_multiplier
    c = view["close"]
    ret = c.pct_change().dropna()
    last_ok = lambda col: float(view[col].iloc[-1]) if pd.notna(view[col].iloc[-1]) else None  # noqa: E731
    out: dict[str, Any] = {
        "status": "ok", "bars": int(len(view)), "first_open": float(view["open"].iloc[0]), "last_close": float(c.iloc[-1]),
        "period_return": float(c.iloc[-1] / view["open"].iloc[0] - 1), "high": float(view["high"].max()),
        "low": float(view["low"].min()), "max_drawdown": float((c / c.cummax() - 1).min()),
        "volatility_ann": float(np.log1p(ret).std() * math.sqrt(bars_per_year(tf, is_crypto(symbol)))) if len(ret) > 2 else float("nan"),
        "avg_volume": float(view["volume"].mean()),
        "last_volume_ratio": float(view["volume"].iloc[-1] / view["volume"].mean()) if view["volume"].mean() else float("nan"),
        "rsi": float(view["rsi"].iloc[-1]), "atr_pct": float(view["atr"].iloc[-1] / c.iloc[-1]),
        "sma20": last_ok("sma20"), "sma50": last_ok("sma50")}
    info = compute_regime(full[["high", "low", "close"]], symbol)
    ml, ms = float(signal_multiplier(info.regime, +1, symbol)), float(signal_multiplier(info.regime, -1, symbol))
    out.update(regime=info.regime, regime_long_mult=ml, regime_short_mult=ms,
               adx=float(info.adx) if pd.notna(info.adx) else None, vol_pctile=float(info.vol_pctile),
               close_vs_sma50=float(info.close_vs_sma50), dd_local=float(info.dd_local),
               overheat=None if pd.isna(info.overheat) else float(info.overheat))
    out["score"] = float(np.clip((ml - ms) / (ml + ms), -1, 1)) if ml + ms > 0 else 0.0
    return out


# ───────────────────────── модуль 1: Kronos ─────────────────────────
_KRONOS: dict[str, Any] = {}


def kronos_module(full: pd.DataFrame, symbol: str, tf: str, samples: int) -> dict[str, Any]:
    try:
        if "ad" not in _KRONOS:
            from finvex.analysis.kronos.adapter import KronosAdapter
            from finvex.engine.config_loader import load_kronos_config
            _KRONOS["ad"] = KronosAdapter(load_kronos_config("configs/kronos.yaml"))
        ad = _KRONOS["ad"]
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"Kronos не загрузился: {exc}"}
    if getattr(ad, "is_mock", False):
        return {"status": "mock", "error": "Kronos в mock-режиме (модель не загружена) - прогноз пропущен"}
    horizon = 10 if tf == "1d" else 24
    try:
        ad.cfg["prediction_length"] = horizon
        ad.cfg["sample_count"] = samples
        cols = [c for c in ("timestamps", "open", "high", "low", "close", "volume", "amount") if c in full.columns]
        ctx = full[cols].tail(min(400, len(full))).reset_index(drop=True)
        t0 = time.time()
        st = ad.encode_prices(ctx, symbol)
        fc = np.asarray(st.forecast_close, dtype=float)
        if len(fc) == 0 or not np.isfinite(fc).all():
            return {"status": "error", "error": "Kronos вернул некорректный прогноз"}
        last = float(ctx["close"].iloc[-1])
        change, voln = float(fc[-1] / last - 1), float(st.volatility_norm)
        return {"status": "ok", "horizon": horizon, "last": last, "target": float(fc[-1]), "change": change,
                "expected_return": float(st.expected_return), "confidence": float(st.confidence),
                "direction": int(st.direction), "volatility_norm": voln, "seconds": round(time.time() - t0, 1),
                "forecast_close": fc.tolist(), "forecast_high": np.asarray(st.forecast_high, dtype=float).tolist(),
                "forecast_low": np.asarray(st.forecast_low, dtype=float).tolist(),
                "score": float(np.tanh(change / max(voln, 1e-4)))}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"ошибка прогноза Kronos: {exc}"}


# ───────────────────────── модуль 3: новости ─────────────────────────
def _parse_dt(v: Any) -> Optional[datetime]:
    s = str(v or "").strip()
    if not s:
        return None
    for f in (lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")), parsedate_to_datetime):
        try:
            d = f(s)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except Exception:  # noqa: BLE001
            continue
    return None


def _read_yaml(path: str) -> dict:
    try:
        import yaml
        p = Path(path)
        return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


_NEWS_CACHE: dict[str, Any] = {}


def news_module(inst: Instrument, req: Request, use_llm: bool, max_llm: int) -> dict[str, Any]:
    try:
        from finvex.analysis.news.keyword_filter import WatchTermsStore, build_filter
        from finvex.analysis.news.rss_collector import RSSCollector, RSSCollectorConfig
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"новостной модуль недоступен: {exc}"}
    try:
        cfg = _read_yaml("configs/news_agent.yaml")
        rss_cfg = cfg.get("rss") or {}
        if "items" not in _NEWS_CACHE:
            ccfg = RSSCollectorConfig(request_timeout=int(rss_cfg.get("request_timeout", 15)))
            if rss_cfg.get("feeds"):
                ccfg.feeds = rss_cfg["feeds"]
            days = max(1, int(math.ceil((datetime.now(MSK) - req.start).total_seconds() / 86400)) + 1)
            _NEWS_CACHE["items"] = RSSCollector(ccfg).fetch_backlog(days=days)
        items = _NEWS_CACHE["items"]
        store = WatchTermsStore()
        store.update({inst.symbol: inst.news_terms or [inst.symbol]}, "analysis")
        flt = build_filter(store, "keyword")
        start_u, end_u = req.start.astimezone(timezone.utc), req.end.astimezone(timezone.utc)
        hits = []
        for it in items:
            dt = _parse_dt(it.published_at)
            if dt is not None and start_u <= dt <= end_u and inst.symbol in flt.match(it.title, it.summary):
                hits.append((dt, it))
        hits.sort(key=lambda x: x[0], reverse=True)
        out: dict[str, Any] = {"status": "ok", "total_feed_items": len(items), "matched": len(hits),
                               "terms": inst.news_terms, "items": [], "score": None}
        if not hits:
            out["note"] = "в RSS-лентах за выбранный период новостей по инструменту не найдено (ленты отдают только свежие записи)"
            return out
        analyzer = None
        if use_llm:
            try:
                from finvex.analysis.news.deep_dive import NewsDeepDiveAnalyzer
                from finvex.engine.config_loader import merge_llm_section
                analyzer = NewsDeepDiveAnalyzer(merge_llm_section((cfg.get("news_risk") or {}).get("llm", {})))
            except Exception as exc:  # noqa: BLE001
                out["llm_error"] = str(exc)
        for k, (dt, it) in enumerate(hits[:30]):
            row = {"time": dt.astimezone(MSK).strftime("%Y-%m-%d %H:%M"), "source": it.source, "title": it.title, "link": it.link}
            if analyzer is not None and k < max_llm:
                try:
                    r = analyzer.analyze(inst.symbol, it)
                except Exception as exc:  # noqa: BLE001
                    r = None
                    logger.warning("LLM-оценка новости не удалась: %s", exc)
                if r is not None:
                    row.update(sentiment=float(r.sentiment), severity=float(r.severity), materiality=float(r.materiality),
                               category=str(r.risk_category), action=str(r.action), rationale=str(r.rationale)[:300],
                               effective_risk=float(r.effective_risk))
            out["items"].append(row)
        scored = [x for x in out["items"] if "sentiment" in x]
        if scored:
            w = np.array([max(0.05, x["materiality"]) for x in scored])
            out["avg_sentiment"] = float(np.average([x["sentiment"] for x in scored], weights=w))
            out["max_effective_risk"] = float(max(x["effective_risk"] for x in scored))
            out["score"] = float(np.clip(out["avg_sentiment"], -1, 1))
        return out
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"ошибка новостного модуля: {exc}"}


# ───────────────────────── сводка ─────────────────────────
def verdict_label(x: Optional[float]) -> str:
    if x is None:
        return "нет данных"
    if x >= 0.45:
        return "бычий"
    if x >= 0.15:
        return "умеренно бычий"
    if x > -0.15:
        return "нейтральный"
    if x > -0.45:
        return "умеренно медвежий"
    return "медвежий"


def combine(kr: dict, te: dict, nw: dict) -> dict[str, Any]:
    parts = {"Kronos": kr.get("score") if kr.get("status") == "ok" else None,
             "Режим/техника": te.get("score") if te.get("status") == "ok" else None,
             "Новости": nw.get("score") if nw.get("status") == "ok" else None}
    vals = [v for v in parts.values() if v is not None]
    comp = float(np.mean(vals)) if vals else None
    signs = {np.sign(v) for v in vals if abs(v) >= 0.1}
    agree = "модули согласны" if len(signs) == 1 and len(vals) >= 2 else ("модули расходятся" if len(signs) > 1 else "сигнал слабый")
    return {"scores": parts, "composite": comp, "label": verdict_label(comp), "agreement": agree, "modules_used": len(vals)}


def llm_commentary(req: Request, inst: Instrument, tech: dict, kr: dict, nw: dict, comb: dict) -> Optional[dict]:
    brief = [{k: x.get(k) for k in ("time", "title", "sentiment", "severity", "category")} for x in (nw.get("items") or [])[:6]]
    payload = {"инструмент": f"{inst.symbol} {inst.name}", "период": f"{req.start:%d.%m.%Y} - {req.end:%d.%m.%Y}", "таймфрейм": req.timeframe,
               "техника": {k: tech.get(k) for k in ("period_return", "high", "low", "max_drawdown", "volatility_ann", "rsi", "regime",
                                                    "adx", "close_vs_sma50", "atr_pct", "last_close")},
               "kronos": {k: kr.get(k) for k in ("status", "horizon", "change", "confidence", "target", "last")},
               "новости": {"avg_sentiment": nw.get("avg_sentiment"), "items": brief}, "сводка_модулей": comb}
    sysmsg = ("Ты рыночный аналитик. По JSON-данным трёх модулей (Kronos, режим/техника, новости) дай краткий вывод на русском. "
              'Верни json: {"summary": 4-6 предложений, "scenarios": [{"name":..., "condition":..., "outlook":...}] (2-3 сценария), '
              '"risks": [до 4 коротких пунктов], "watch": [до 4 уровней/событий для наблюдения]}. '
              "Опирайся только на присланные числа, ничего не выдумывай, не давай гарантий и персональных инвестсоветов.")
    return llm_json(sysmsg, json.dumps(payload, ensure_ascii=False, default=str), max_tokens=1200)


# ───────────────────────── графики ─────────────────────────
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(f"для графиков нужен matplotlib со всеми зависимостями: {exc}. "
                           "pip install matplotlib pillow contourpy cycler fonttools kiwisolver pyparsing python-dateutil packaging") from exc
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _ticks(ax, stamps: pd.Series, extra: int, tf: str, n_ticks: int = 8):
    n = len(stamps)
    idx = np.linspace(0, n - 1, min(n_ticks, n)).astype(int)
    fmt = "%d.%m" if tf == "1d" else "%d.%m %H:%M"
    ax.set_xticks(idx)
    ax.set_xticklabels([stamps.iloc[i].strftime(fmt) for i in idx], fontsize=8)
    ax.set_xlim(-1, n + extra)


def chart_price(view: pd.DataFrame, inst: Instrument, req: Request, kr: dict, tech: dict, path: Path) -> None:
    plt = _plt()
    v = view.reset_index(drop=True)
    n = len(v)
    x = np.arange(n)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [4, 1.2, 1.2]})
    up = (v["close"] >= v["open"]).to_numpy()
    col = np.where(up, "#2e9e5b", "#d64545")
    if n <= 220:
        a1.vlines(x, v["low"], v["high"], color=col, lw=1)
        a1.bar(x, (v["close"] - v["open"]).abs().clip(lower=v["close"].abs() * 1e-5), bottom=np.minimum(v["open"], v["close"]), width=0.65, color=col)
    else:
        a1.plot(x, v["close"], color="#1f4e99", lw=1.3, label="Close")
    a1.plot(x, v["sma20"], color="#e08a00", lw=1.1, label="SMA20")
    a1.plot(x, v["sma50"], color="#7b3fb0", lw=1.1, label="SMA50")
    a1.fill_between(x, v["bb_lo"], v["bb_up"], color="#7a8aa0", alpha=0.12, label="Bollinger 20/2")
    a1.axhline(tech["high"], color="#888", ls=":", lw=0.9)
    a1.axhline(tech["low"], color="#888", ls=":", lw=0.9)
    extra = 0
    if kr.get("status") == "ok":
        fc, hi, lo = (np.array(kr[k]) for k in ("forecast_close", "forecast_high", "forecast_low"))
        fx = np.arange(n, n + len(fc))
        extra = len(fc)
        a1.plot(np.r_[n - 1, fx], np.r_[v["close"].iloc[-1], fc], color="#0b7285", lw=2, ls="--", label=f"Kronos прогноз (+{len(fc)} баров)")
        a1.fill_between(fx, lo, hi, color="#0b7285", alpha=0.2)
        a1.axvline(n - 0.5, color="#0b7285", lw=0.8, ls=":")
    a1.set_title(f"{inst.symbol} {inst.name}: {req.start:%d.%m.%Y %H:%M} - {req.end:%d.%m.%Y %H:%M} MSK, {req.timeframe}  |  "
                 f"{tech['period_return'] * 100:+.2f}% за период, режим {tech['regime']}", fontsize=12)
    a1.legend(loc="upper left", fontsize=8, ncol=3)
    a2.bar(x, v["volume"], color=col, width=0.7, alpha=0.7)
    a2.set_ylabel("Объём")
    a3.plot(x, v["rsi"], color="#1f4e99", lw=1.1)
    a3.axhline(70, color="#d64545", ls="--", lw=0.8)
    a3.axhline(30, color="#2e9e5b", ls="--", lw=0.8)
    a3.set_ylim(0, 100)
    a3.set_ylabel("RSI 14")
    _ticks(a3, v["timestamps"], extra, req.timeframe)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_summary(inst: Instrument, comb: dict, nw: dict, path: Path) -> None:
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    names = list(comb["scores"].keys()) + ["ИТОГ"]
    vals = [comb["scores"][k] for k in comb["scores"]] + [comb["composite"]]
    ys = np.arange(len(names))[::-1]
    for y, nme, val in zip(ys, names, vals):
        if val is None:
            a.barh(y, 0.0, color="#bbb")
            a.text(0.02, y, "нет данных", va="center", fontsize=9, color="#666")
        else:
            a.barh(y, val, color="#2e9e5b" if val >= 0 else "#d64545", alpha=1.0 if nme == "ИТОГ" else 0.8)
            a.text(val + (0.03 if val >= 0 else -0.03), y, f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
    a.set_yticks(ys)
    a.set_yticklabels(names)
    a.set_xlim(-1.2, 1.2)
    a.axvline(0, color="#444", lw=0.8)
    a.set_title(f"{inst.symbol}: сигналы модулей (-1 медвежий ... +1 бычий)\nитог: {comb['label']}, {comb['agreement']}", fontsize=11)
    items = [x for x in (nw.get("items") or []) if "sentiment" in x]
    if items:
        t = pd.to_datetime([x["time"] for x in items])
        s = [x["sentiment"] for x in items]
        b.scatter(t, s, s=[60 + 400 * x["materiality"] for x in items], c=["#2e9e5b" if q >= 0 else "#d64545" for q in s],
                  alpha=0.7, edgecolor="k", lw=0.5)
        b.axhline(0, color="#444", lw=0.8)
        b.set_ylim(-1.1, 1.1)
        b.set_title("Новости: sentiment (размер = materiality)", fontsize=11)
        b.tick_params(axis="x", labelrotation=30, labelsize=8)
    else:
        b.axis("off")
        b.text(0.5, 0.5, "Оценённых новостей за период нет", ha="center", va="center", fontsize=12, color="#666")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_compare(series: dict[str, pd.Series], req: Request, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for t, s in series.items():
        s = s.dropna()
        if len(s):
            ax.plot(np.linspace(0, 1, len(s)), (s / s.iloc[0] - 1) * 100, label=t, lw=1.6)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_title(f"Доходность за период ({req.start:%d.%m} - {req.end:%d.%m.%Y}), %, старт = 0", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{req.start:%d.%m.%Y}", f"{req.end:%d.%m.%Y}"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ───────────────────────── отчёт ─────────────────────────
def pct(x: Optional[float], d: int = 2) -> str:
    return "н/д" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{d}f}%"


def num(x: Optional[float], d: int = 2) -> str:
    return "н/д" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{d}f}".replace(",", " ")


def ticker_blocks(inst: Instrument, tech: dict, kr: dict, nw: dict, comb: dict, comm: Optional[dict], charts: dict[str, Path]) -> list:
    B: list = [("h2", f"{inst.symbol} {inst.name}: итог - {comb['label']}")]
    sc = ", ".join(f"{k} {v:+.2f}" for k, v in comb["scores"].items() if v is not None) or "нет данных"
    B.append(("p", f"Сводный сигнал {comb['composite']:+.2f} ({comb['agreement']}; модулей с данными: {comb['modules_used']} из 3). {sc}."
              if comb["composite"] is not None else "Ни один модуль не дал сигнала."))
    B.append(("img", charts["price"]))
    B.append(("h3", "Модуль 2. Режим и техника"))
    B.append(("li", [
        f"Цена: {num(tech['first_open'])} -> {num(tech['last_close'])} ({pct(tech['period_return'])}), диапазон {num(tech['low'])} - {num(tech['high'])}.",
        f"Режим рынка: {tech['regime']} (вес лонга {tech['regime_long_mult']:.1f}, шорта {tech['regime_short_mult']:.1f}), ADX {num(tech['adx'], 1)}, перцентиль волатильности {tech['vol_pctile'] * 100:.0f}%.",
        f"Макс. просадка за период {pct(tech['max_drawdown'])}, годовая волатильность {pct(tech['volatility_ann'], 1)}, ATR {pct(tech['atr_pct'])} от цены.",
        f"RSI(14) {tech['rsi']:.1f}; цена к SMA50: {pct(tech['close_vs_sma50'])}; объём последнего бара {num(tech['last_volume_ratio'])}x от среднего."]))
    B.append(("h3", "Модуль 1. Kronos"))
    if kr.get("status") == "ok":
        d = "вверх" if kr["direction"] > 0 else "вниз" if kr["direction"] < 0 else "боковик"
        B.append(("li", [f"Прогноз на {kr['horizon']} баров: {num(kr['last'])} -> {num(kr['target'])} ({pct(kr['change'])}).",
                         f"Уверенность {kr['confidence'] * 100:.0f}% (ширина канала {pct(kr['volatility_norm'])}), направление {d}; расчёт {kr['seconds']} c."]))
    else:
        B.append(("p", kr.get("error", "нет данных")))
    B.append(("h3", "Модуль 3. Новости"))
    if nw.get("status") != "ok":
        B.append(("p", nw.get("error", "нет данных")))
    elif not nw.get("items"):
        B.append(("p", nw.get("note", "новостей не найдено")))
    else:
        head = f"Найдено по запросу ({', '.join(nw.get('terms', [])[:6])}): {nw['matched']} из {nw['total_feed_items']} записей лент."
        if nw.get("avg_sentiment") is not None:
            head += f" Средний sentiment {nw['avg_sentiment']:+.2f}, макс. эффективный риск {nw['max_effective_risk']:.2f}."
        B.append(("p", head))
        B.append(("li", [f"{x['time']} [{x['source']}] {x['title']}" + (f" - sentiment {x['sentiment']:+.2f}, severity {x['severity']:.2f}, {x['category']}" if "sentiment" in x else "")
                         for x in nw["items"][:10]]))
    B.append(("img", charts["summary"]))
    if comm:
        B.append(("h3", "Вывод аналитика (LLM по данным модулей)"))
        if comm.get("summary"):
            B.append(("p", str(comm["summary"])))
        scen = [f"{s.get('name', '')}: {s.get('condition', '')} -> {s.get('outlook', '')}" for s in comm.get("scenarios", []) if isinstance(s, dict)]
        if scen:
            B += [("p", "Сценарии:"), ("li", scen)]
        if comm.get("risks"):
            B += [("p", "Риски:"), ("li", [str(x) for x in comm["risks"]])]
        if comm.get("watch"):
            B += [("p", "Что отслеживать:"), ("li", [str(x) for x in comm["watch"]])]
    return B


def render_md(blocks: list, root: Path) -> str:
    out = []
    for kind, val in blocks:
        if kind == "h1":
            out.append(f"# {val}\n")
        elif kind == "h2":
            out.append(f"\n## {val}\n")
        elif kind == "h3":
            out.append(f"\n### {val}\n")
        elif kind == "p":
            out.append(f"{val}\n")
        elif kind == "li":
            out.append("\n".join(f"- {x}" for x in val) + "\n")
        elif kind == "img":
            out.append(f"![]({Path(val).relative_to(root).as_posix()})\n")
    return "\n".join(out)


def render_html(blocks: list) -> str:
    e, body = html.escape, []
    for kind, val in blocks:
        if kind in ("h1", "h2", "h3"):
            body.append(f"<{kind}>{e(val)}</{kind}>")
        elif kind == "p":
            body.append(f"<p>{e(val)}</p>")
        elif kind == "li":
            body.append("<ul>" + "".join(f"<li>{e(x)}</li>" for x in val) + "</ul>")
        elif kind == "img":
            body.append(f'<img src="data:image/png;base64,{base64.b64encode(Path(val).read_bytes()).decode()}" style="max-width:100%">')
    css = "body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}h2{border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:36px}li{margin:3px 0}"
    return f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>Анализ рынка</title><style>{css}</style></head><body>{''.join(body)}</body></html>"


# ───────────────────────── оркестратор ─────────────────────────
def analyze(text: str, *, timeframe: Optional[str] = None, use_kronos: bool = True, use_news: bool = True,
            use_llm: bool = True, kronos_samples: int = 3, out_root: str = "reports") -> dict[str, Any]:
    print("Разбор запроса ...", flush=True)
    req = parse_request(text, timeframe=timeframe, use_llm=use_llm)
    out_dir = Path(out_root) / f"{'_'.join(req.tickers)}_{datetime.now(MSK):%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks: list = [("h1", "Анализ рынка по запросу"), ("p", f"Запрос: «{req.text}»"),
                    ("li", req.notes + [f"инструменты: {', '.join(f'{i.symbol} ({i.market})' for i in req.instruments)}; разбор запроса: {req.parsed_by}"])]
    for n in req.notes:
        print("  " + n, flush=True)
    results: dict[str, Any] = {"request": {"text": req.text, "instruments": [vars(i) for i in req.instruments],
                                           "start": req.start.isoformat(), "end": req.end.isoformat(), "timeframe": req.timeframe,
                                           "parsed_by": req.parsed_by}, "tickers": {}}
    closes: dict[str, pd.Series] = {}
    for inst in req.instruments:
        t = inst.symbol
        print(f"[{t}] загрузка свечей ...", flush=True)
        try:
            full = add_indicators(load_market_data(inst, req))
        except Exception as exc:  # noqa: BLE001
            blocks += [("h2", t), ("p", f"Данные недоступны: {exc}")]
            results["tickers"][t] = {"error": str(exc)}
            continue
        start_n, end_n = req.start.replace(tzinfo=None), req.end.replace(tzinfo=None)
        view = full[(full["timestamps"] >= start_n) & (full["timestamps"] <= end_n)].reset_index(drop=True)
        if len(view) < 3:
            msg = f"в выбранном периоде слишком мало баров ({len(view)}) - биржа могла быть закрыта; расширьте период"
            blocks += [("h2", t), ("p", msg)]
            results["tickers"][t] = {"error": msg}
            continue
        closes[t] = view.set_index("timestamps")["close"]
        print(f"[{t}] баров всего {len(full)}, в периоде {len(view)}; модули ...", flush=True)
        tech = tech_module(full, view, t, req.timeframe)
        kr = kronos_module(full, t, req.timeframe, kronos_samples) if use_kronos else {"status": "skipped", "error": "отключён флагом --no-kronos"}
        nw = news_module(inst, req, use_llm, 8) if use_news else {"status": "skipped", "error": "отключён флагом --no-news"}
        comb = combine(kr, tech, nw)
        comm = llm_commentary(req, inst, tech, kr, nw, comb) if use_llm and comb["modules_used"] else None
        charts = {"price": out_dir / f"{t}_price.png", "summary": out_dir / f"{t}_summary.png"}
        chart_price(view, inst, req, kr, tech, charts["price"])
        chart_summary(inst, comb, nw, charts["summary"])
        blocks += ticker_blocks(inst, tech, kr, nw, comb, comm, charts)
        results["tickers"][t] = {"technical": tech, "kronos": {k: v for k, v in kr.items() if not k.startswith("forecast_")},
                                 "news": nw, "combined": comb, "llm": comm}
    if len(closes) > 1:
        p = out_dir / "compare.png"
        chart_compare(closes, req, p)
        blocks[3:3] = [("h2", "Сравнение инструментов"), ("img", p)]
    blocks.append(("p", "Это аналитическая справка, а не инвестиционная рекомендация: модели ошибаются, результаты прошлого не гарантируют будущего."))
    (out_dir / "report.md").write_text(render_md(blocks, out_dir), encoding="utf-8")
    (out_dir / "report.html").write_text(render_html(blocks), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    results["out_dir"] = str(out_dir)
    return results


def main(argv: Optional[list[str]] = None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description="Полный анализ рынка по текстовому запросу")
    ap.add_argument("query", nargs="*", help="например: проанализируй Сбер за последние 2 недели")
    ap.add_argument("--timeframe", choices=list(ALLOWED_TF), help="принудительный таймфрейм")
    ap.add_argument("--no-kronos", action="store_true")
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="без LLM (запасной разбор запроса, без оценки новостей и вывода)")
    ap.add_argument("--kronos-samples", type=int, default=3)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("finvex.market_analysis").setLevel(logging.WARNING)
    interactive = not a.query
    while True:
        if interactive:
            try:
                text = input("\nЗапрос (пустая строка - выход): ").strip()
            except EOFError:
                break
            if not text:
                break
        else:
            text = " ".join(a.query)
        try:
            res = analyze(text, timeframe=a.timeframe, use_kronos=not a.no_kronos, use_news=not a.no_news,
                          use_llm=not a.no_llm, kronos_samples=a.kronos_samples, out_root=a.out)
            print(f"\nГотово: {Path(res['out_dir']) / 'report.html'} (графики PNG рядом)")
            for t, r in res["tickers"].items():
                c = r.get("combined")
                print(f"  {t}: " + (f"{c['label']} ({c['composite']:+.2f}), {c['agreement']}" if c and c["composite"] is not None else r.get("error", "нет сигнала")))
        except ValueError as exc:
            print(f"Ошибка запроса: {exc}")
        if not interactive:
            break


if __name__ == "__main__":
    main()
