#!/usr/bin/env python3
"""
Finvex healthcheck - проверка НАСТОЯЩИХ подключений (не заглушек), архитектура пакета finvex/.

Запуск из корня репозитория:
    python tests/healthcheck.py                          # всё
    python tests/healthcheck.py --only moex kronos       # выбранные блоки
    python tests/healthcheck.py --only llm news --no-llm-news

Блоки: moex | bybit | tinkoff | kronos | llm | news
Безопасно: Bybit - testnet, T-Invest - sandbox, ордера не отправляются.
Секреты в вывод и отчёт не попадают. Отчёт: data/healthcheck_report.json
Код возврата: 0 - FAIL нет, 1 - есть хотя бы один FAIL.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
for p in (ROOT, ROOT / "src"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

BLOCKS = ["moex", "bybit", "tinkoff", "kronos", "llm", "news"]
_SECRETS: List[str] = []

# Пути модулей в новой архитектуре. Если модуль переехал - правится только здесь.
M = {
    "config": ("finvex.engine.config_loader", "config_loader.py"),
    "moex": ("finvex.venues.moex", "moex.py"),
    "bybit": ("finvex.venues.bybit.client", "bybit_client.py"),
    "tinkoff": ("finvex.venues.tinkoff.client", "tinkoff_client.py"),
    "kronos": ("finvex.analysis.kronos.adapter", "kronos_adapter.py"),
    "bridge": ("finvex.analysis.kronos.bridge", "kronos_bridge.py"),
    "rss": ("finvex.analysis.news.rss_collector", "rss_collector.py"),
    "kwfilter": ("finvex.analysis.news.keyword_filter", "keyword_filter.py"),
    "deepdive": ("finvex.analysis.news.deep_dive", "news_deep_dive.py"),
    "strategist": ("finvex.decision.strategist", "strategist.py"),
}


def mask(text: Any) -> str:
    s = str(text)
    for sec in _SECRETS:
        if sec and len(sec) > 6:
            s = s.replace(sec, "***")
    s = re.sub(r"\bt\.[A-Za-z0-9_\-]{12,}", "t.***", s)
    s = re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}", "sk-***", s)
    s = re.sub(r"(api[_-]?key|secret|token)(=|:)\s*\S+", r"\1\2***", s, flags=re.I)
    return s


@dataclass
class Result:
    block: str
    name: str
    status: str
    detail: str = ""
    seconds: float = 0.0
    hint: str = ""


RESULTS: List[Result] = []
ICON = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}


class _Skip(Exception):
    pass


def record(block: str, name: str, status: str, detail: str = "",
           t0: Optional[float] = None, hint: str = "") -> Result:
    r = Result(block, name, status, mask(detail), round(time.time() - t0, 2) if t0 else 0.0, hint)
    RESULTS.append(r)
    line = f"{ICON[status]} {block:<8} {name:<34} {r.detail}"
    if r.seconds:
        line += f" ({r.seconds}s)"
    print(line, flush=True)
    if hint and status in ("FAIL", "WARN"):
        print(f"           -> {hint}", flush=True)
    return r


def check(block: str, name: str, fn: Callable[[], Any], hint: str = "") -> Any:
    """fn возвращает (status, detail[, value]) или строку (= PASS). Исключение = FAIL."""
    t0 = time.time()
    try:
        out = fn()
        if isinstance(out, tuple):
            record(block, name, out[0], out[1], t0, hint if out[0] != "PASS" else "")
            return out[2] if len(out) > 2 else None
        record(block, name, "PASS", out or "", t0)
        return out
    except _Skip as e:
        record(block, name, "SKIP", str(e), t0)
    except Exception as e:  # noqa: BLE001
        if os.environ.get("FINVEX_TRACE"):
            traceback.print_exc()
        record(block, name, "FAIL", f"{type(e).__name__}: {e}", t0, hint)
    return None


_PY_INDEX: Optional[dict] = None


def _index() -> dict:
    global _PY_INDEX
    if _PY_INDEX is None:
        _PY_INDEX = {}
        skip = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "Kronos",
                "archive", "_archive", "build", "dist", "site-packages"}
        for dp, dn, fn in os.walk(ROOT):
            dn[:] = [d for d in dn if d not in skip and not d.startswith(".")]
            for f in fn:
                if f.endswith(".py"):
                    _PY_INDEX.setdefault(f, []).append(Path(dp) / f)
    return _PY_INDEX


def _dotted(path: Path) -> str:
    src = ROOT / "src"
    base = src if src in path.parents and not (src / "__init__.py").exists() else ROOT
    return ".".join(path.relative_to(base).with_suffix("").parts)


def load(key: str) -> Any:
    """Сначала импорт по новому пути finvex.*, затем запасной поиск файла по имени."""
    dotted, filename = M[key]
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError as e:
        if not (e.name and (dotted == e.name or dotted.startswith(e.name + "."))):
            raise
    cands = [p for p in _index().get(filename, []) if "tests" not in p.parts]
    if not cands:
        raise ImportError(f"Не найден модуль {dotted} (и файл {filename}) в {ROOT}")
    return importlib.import_module(_dotted(sorted(cands, key=lambda p: len(p.parts))[0]))


def pick(mod: Any, *names: str, prefix: str = "") -> Any:
    """Класс по точному имени, иначе первый класс модуля с заданным префиксом."""
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    if prefix:
        for n, c in inspect.getmembers(mod, inspect.isclass):
            if c.__module__ == mod.__name__ and n.startswith(prefix):
                return c
    raise ImportError(f"в {mod.__name__} нет класса {names or prefix}; есть: "
                      f"{[n for n, c in inspect.getmembers(mod, inspect.isclass) if c.__module__ == mod.__name__]}")


def read_yaml(path: str) -> dict:
    import yaml
    for p in (ROOT / path, ROOT / "finvex" / path):
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


SHARED: dict = {}


def moex_provider() -> Any:
    if "prov" not in SHARED:
        SHARED["prov"] = pick(load("moex"), "MoexISSProvider", prefix="Moex")()
    return SHARED["prov"]


def moex_candles(symbol: str, tf: str = "1h", days: int = 90):
    key = (symbol, tf, days)
    if key not in SHARED:
        end = datetime.now(timezone.utc)
        SHARED[key] = moex_provider().get_candles(symbol, tf, end - timedelta(days=days), end)
    return SHARED[key]


# -- MOEX ---------------------------------------------------------------------
def run_moex(a):
    prov = moex_provider()
    sym = a.symbol

    def instruments():
        items = prov.get_instruments()
        ids = {str(getattr(i, "id", getattr(i, "secid", i))) for i in items}
        ok = any(sym in s for s in ids)
        return ("PASS" if len(items) > 50 and ok else "WARN",
                f"{len(items)} инструментов TQBR, {sym} {'найден' if ok else 'НЕ найден'}")

    def candles():
        s = moex_candles(sym, "1h", 30)
        if not len(s):
            return "FAIL", "пустой ряд", None
        q = s.quality
        age_h = (datetime.now(timezone.utc) - s.last_close_time).total_seconds() / 3600
        st = "PASS" if (q is None or q.ok) else "FAIL"
        extra = f"quality={q.worst if q else 'n/a'}" + (
            f" [{', '.join(i.code for i in q.issues)}]" if q and q.issues else "")
        return st, f"{len(s)} часовых баров, last close={s.candles[-1].close}, возраст {age_h:.1f} ч, {extra}", s

    def daily():
        s = moex_candles(sym, "1d", 120)
        return ("PASS" if len(s) >= 50 else "WARN", f"{len(s)} дневных баров")

    def quotes():
        q = prov.get_quotes([sym])
        if not q:
            return "WARN", "котировок нет (возможно, биржа закрыта)"
        x = q[0]
        return "PASS", f"last={getattr(x, 'last', None)} bid={getattr(x, 'bid', None)} ask={getattr(x, 'ask', None)}"

    def trades():
        t = prov.get_trades(sym, limit=20)
        return ("PASS" if t else "WARN", f"{len(t)} обезличенных сделок")

    check("moex", "get_instruments (TQBR)", instruments, "Проверь доступ к iss.moex.com (прокси/VPN/файрвол).")
    check("moex", f"get_candles {sym} 1h", candles, "ISS не отвечает или формат изменился - запусти с FINVEX_TRACE=1.")
    check("moex", f"get_candles {sym} 1d", daily)
    check("moex", f"get_quotes {sym}", quotes)
    check("moex", f"get_trades {sym}", trades)


# -- Bybit --------------------------------------------------------------------
def run_bybit(a):
    import requests

    def public(url):
        def f():
            r = requests.get(f"{url}/v5/market/tickers",
                             params={"category": "linear", "symbol": "BTCUSDT"}, timeout=15)
            if r.status_code == 403:
                return "FAIL", "HTTP 403 - вероятно, блокировка по региону/IP"
            r.raise_for_status()
            return "PASS", f"BTCUSDT last={r.json()['result']['list'][0]['lastPrice']}"
        return f

    hint = "Bybit блокирует часть регионов - нужен разрешённый IP/прокси; сам код в этом не виноват."
    check("bybit", "public testnet tickers", public("https://api-testnet.bybit.com"), hint)
    check("bybit", "public mainnet tickers", public("https://api.bybit.com"), hint)

    cl = {}

    def client():
        try:
            cfg = load("config").load_bybit_config(testnet=True)
        except (FileNotFoundError, ValueError) as e:
            raise _Skip(f"ключи testnet не заданы: {e}")
        _SECRETS.extend([cfg.api_key, cfg.api_secret])
        cl["c"] = pick(load("bybit"), "BybitClient", prefix="Bybit")(cfg)
        return f"клиент создан, category={cfg.category}"

    check("bybit", "BybitClient (testnet keys)", client,
          "Заполни configs/bybit_testnet_key.yaml реальными testnet-ключами.")
    if "c" not in cl:
        return
    c = cl["c"]
    check("bybit", "get_wallet_balance (auth)",
          lambda: f"ответ получен: {type(c.get_wallet_balance()).__name__}",
          "Ключ неверный, без прав Read или с IP-ограничением.")
    check("bybit", "get_prices BTCUSDT", lambda: ("PASS", str(c.get_prices(["BTCUSDT"]))))

    def cndl():
        df = c.get_candles("BTCUSDT", "1h", datetime.now(timezone.utc) - timedelta(days=5), 200)
        miss = {"open", "high", "low", "close", "volume"} - set(map(str.lower, df.columns))
        return ("PASS" if len(df) > 50 and not miss else "FAIL",
                f"{len(df)} баров, нет колонок: {sorted(miss) or '-'}")

    check("bybit", "get_candles BTCUSDT 1h", cndl)
    check("bybit", "get_instrument_spec", lambda: ("PASS", str(c.get_instrument_spec("BTCUSDT"))))
    check("bybit", "get_portfolio (нормализация)",
          lambda: ("PASS", f"ключи: {sorted(c.get_portfolio().keys())}"))


# -- T-Invest -----------------------------------------------------------------
def run_tinkoff(a):
    cl = {}

    def client():
        try:
            cfg = load("config").load_tinkoff_config(sandbox=True)
        except (FileNotFoundError, ValueError, TypeError) as e:
            raise _Skip(f"sandbox-токен не задан: {e}")
        _SECRETS.append(cfg.token)
        cl["c"] = pick(load("tinkoff"), "TinkoffClient", prefix="Tinkoff")(cfg.token, sandbox=True)
        return "клиент создан (sandbox)"

    check("tinkoff", "TinkoffClient (sandbox token)", client,
          "Нужен configs/sandbox_key.yaml с токеном t.xxx и пакет t-tech-investments.")
    if "c" not in cl:
        return
    c = cl["c"]
    sym = a.symbol
    check("tinkoff", f"get_figi {sym}",
          lambda: ("PASS" if c.get_figi(sym) else "FAIL", str(c.get_figi(sym))),
          "Токен недействителен или нет сети до invest-public-api (при SSL-ошибке: SSL_TBANK_VERIFY=True).")
    check("tinkoff", f"get_lot_size {sym}", lambda: ("PASS", f"lot={c.get_lot_size(sym)}"))

    def prices():
        p = c.get_prices([sym])
        SHARED["tink_px"] = p.get(sym)
        return ("PASS" if p.get(sym) else "WARN", str(p))

    check("tinkoff", f"get_prices {sym}", prices)

    def candles():
        from t_tech.invest.schemas import CandleInterval
        df = c.get_candles(sym, datetime.now(timezone.utc) - timedelta(days=10), None,
                           CandleInterval.CANDLE_INTERVAL_HOUR)
        return ("PASS" if len(df) > 10 else "WARN", f"{len(df)} часовых баров")

    check("tinkoff", f"get_candles {sym} 1h", candles)
    check("tinkoff", "get_portfolio (sandbox)",
          lambda: ("PASS", f"ключи: {sorted(c.get_portfolio().keys())}"))

    def cross():
        t = SHARED.get("tink_px")
        s = moex_candles(sym, "1h", 30)
        if not t or not len(s):
            raise _Skip("нет одной из цен")
        m = s.candles[-1].close
        d = abs(t - m) / m * 100
        return ("PASS" if d < 3 else "WARN", f"T-Invest={t} vs MOEX last close={m}, расхождение {d:.2f}%")

    check("tinkoff", "сверка цены с MOEX", cross,
          "Большое расхождение: проверь, что sandbox-цены актуальны и биржа открыта.")


# -- Kronos -------------------------------------------------------------------
class _Grab(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.msgs: List[str] = []

    def emit(self, rec):
        self.msgs.append(self.format(rec))


def run_kronos(a):
    import numpy as np

    def env():
        info = []
        try:
            import torch
            info.append(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
        except ImportError:
            return "FAIL", "torch не установлен"
        try:
            import huggingface_hub  # noqa: F401
            info.append("huggingface_hub OK")
        except ImportError:
            info.append("huggingface_hub НЕТ")
        return "PASS", ", ".join(info)

    check("kronos", "torch / hub", env, "pip install torch huggingface_hub")

    adapter_mod = load("kronos")
    cfg = load("config").load_kronos_config("configs/kronos.yaml")
    if cfg.device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                record("kronos", "device", "WARN",
                       f"в kronos.yaml {cfg.device}, но CUDA недоступна - загрузка упадёт",
                       hint="Поставь device: cpu.")
        except ImportError:
            pass
    grab = _Grab()
    lg = logging.getLogger(adapter_mod.__name__)
    lg.addHandler(grab)
    box = {}

    def loadm():
        t = time.time()
        box["ad"] = pick(adapter_mod, "KronosAdapter", prefix="Kronos")(cfg)
        dt = time.time() - t
        if box["ad"].is_mock:
            why = (grab.msgs[-1] if grab.msgs else "причина не в логах").split("\n")[0]
            return "FAIL", f"MOCK-режим! прогнозы синтетические. Причина: {why[:240]}"
        return "PASS", f"{cfg.model_name} загружена на {cfg.device} за {dt:.1f} c"

    check("kronos", "загрузка модели (не mock)", loadm,
          "Проверь KRONOS_REPO_PATH (клонированный репозиторий Kronos с model/), "
          "интернет для HuggingFace и зависимости.")
    lg.removeHandler(grab)
    ad = box.get("ad")
    if ad is None or ad.is_mock:
        record("kronos", "прогноз на данных MOEX", "SKIP", "модель не загружена - прогноз бессмысленен")
        return
    if a.no_kronos_predict:
        record("kronos", "прогноз на данных MOEX", "SKIP", "--no-kronos-predict")
        return

    def predict():
        s = moex_candles(a.symbol, "1h", 90)
        if len(s) < 64:
            return "FAIL", f"мало баров MOEX: {len(s)}"
        df = load("bridge").to_kronos_frame(s)
        ad.cfg["prediction_length"] = min(int(ad.cfg.get("prediction_length", 120)), a.pred_len)
        ad.cfg["sample_count"] = 1
        t = time.time()
        st = ad.encode_prices(df, a.symbol)
        dt = time.time() - t
        fc = np.asarray(st.forecast_close, dtype=float)
        last = float(df["close"].iloc[-1])
        if len(fc) != ad.cfg["prediction_length"] or not np.isfinite(fc).all():
            return "FAIL", f"некорректный прогноз: len={len(fc)}, finite={bool(np.isfinite(fc).all())}"
        dev = abs(fc[-1] - last) / last * 100
        return ("PASS" if dev < 30 else "WARN",
                f"{a.symbol}: last={last:.2f} -> прогноз +{len(fc)} баров = {fc[-1]:.2f} "
                f"({(fc[-1] / last - 1) * 100:+.2f}%), exp_return={st.expected_return:+.4f}, {dt:.1f} c")

    check("kronos", "MOEX -> bridge -> Kronos -> прогноз", predict,
          "Если упало на данных - проверь формат свечей (колонки/timestamps).")


# -- LLM (DeepSeek) -----------------------------------------------------------
def run_llm(a):
    cfgmod = load("config")
    shared = cfgmod.load_llm_shared()
    key = shared.get("api_key") or ""
    _SECRETS.append(key)

    def cfg():
        if not key:
            raise _Skip("ключ не задан: DEEPSEEK_API_KEY или configs/llm_shared.yaml")
        return (f"provider={shared.get('provider')} model={shared.get('model')} "
                f"base_url={shared.get('base_url')} key=***{key[-4:]}")

    check("llm", "конфигурация и ключ", cfg,
          "setx DEEPSEEK_API_KEY ... (или $env:DEEPSEEK_API_KEY) либо заполни configs/llm_shared.yaml")

    def ping():
        import openai
        client = openai.OpenAI(api_key=key, base_url=shared.get("base_url") or "https://api.deepseek.com",
                               timeout=45)
        kw = dict(model=shared.get("model", "deepseek-flash"),
                  messages=[{"role": "system", "content": "Отвечай только валидным json без пояснений."},
                            {"role": "user", "content": 'Верни json: {"ok": true, "word": "финвекс"}'}],
                  max_tokens=1500, temperature=0, response_format={"type": "json_object"})
        try:
            r = client.chat.completions.create(**kw, extra_body={"thinking": {"type": "disabled"}})
        except openai.BadRequestError:
            r = client.chat.completions.create(**kw)
        ch = r.choices[0]
        txt = (ch.message.content or "").strip()
        txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
        try:
            j = json.loads(txt)
        except json.JSONDecodeError:
            return "FAIL", f"не JSON: finish_reason={ch.finish_reason}, начало ответа: {txt[:80]!r}"
        u = getattr(r, "usage", None)
        return ("PASS" if j.get("ok") is True else "WARN",
                f"ответ разобран как JSON: {j}; finish_reason={ch.finish_reason}; токенов: {getattr(u, 'total_tokens', '?')}")

    if key:
        check("llm", "запрос к модели + разбор JSON", ping,
              "401 - ключ; 402 - баланс; 404 - имя модели; length - поднять max_tokens; "
              "timeout - сеть/прокси; ImportError - pip install openai")

    def strat():
        ls = read_yaml("configs/llm_strategist.yaml").get("llm_strategist", {})
        obj = pick(load("strategist"), "LLMStrategist", prefix="LLM")(cfgmod.merge_llm_section(ls))
        return f"LLMStrategist создан: {type(obj).__name__}"

    check("llm", "LLMStrategist инициализация", strat)


# -- Новости: RSS -> парсинг -> фильтр -> LLM-оценка ---------------------------
def _parse_dt(v: Any) -> Optional[datetime]:
    from email.utils import parsedate_to_datetime
    sv = str(v or "").strip()
    if not sv:
        return None
    for f in (lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")), parsedate_to_datetime):
        try:
            return f(sv)
        except Exception:  # noqa: BLE001
            continue
    return None


def run_news(a):
    cfgmod = load("config")
    rss_mod = load("rss")
    cfg = read_yaml("configs/news_agent.yaml")
    feeds = (cfg.get("rss") or {}).get("feeds") or dict(getattr(rss_mod, "DEFAULT_FEEDS", {}))
    timeout = int((cfg.get("rss") or {}).get("request_timeout", 15))
    col = rss_mod.RSSCollector(rss_mod.RSSCollectorConfig(feeds=feeds, request_timeout=timeout))
    items_all = []

    for src, url in feeds.items():
        def one(src=src, url=url):
            items = col._fetch_feed(src, url)
            if not items:
                return "WARN", "0 записей: лента недоступна, пуста или не разобралась"
            bad = [i for i in items if not i.title.strip() or not i.link]
            ok_dates = sum(1 for i in items if _parse_dt(i.published_at))
            items_all.extend(items)
            st = "PASS" if not bad and ok_dates >= len(items) * 0.8 else "WARN"
            return st, (f"{len(items)} записей, без title/link: {len(bad)}, "
                        f"дат разобрано: {ok_dates}/{len(items)}")

        check("news", f"RSS {src}", one,
              "Лента могла сменить URL/формат, быть недоступной из вашей сети или вернуть 403.")

    kf = load("kwfilter")
    store = kf.WatchTermsStore()
    store.update({"SBER": ["Сбербанк", "Сбер", "SBER", "Греф"], "GAZP": ["Газпром", "GAZP"],
                  "LKOH": ["Лукойл", "LKOH"], "YNDX": ["Яндекс", "YNDX"]}, "healthcheck")
    flt = kf.build_filter(store, "keyword")

    def offline_filter():
        pos = flt.match("Сбербанк объявил о рекордной прибыли", "")
        neg = flt.match("Погода в Москве на выходные", "")
        return ("PASS" if "SBER" in pos and not neg else "FAIL",
                f"позитивный тест -> {pos}, негативный -> {neg}")

    check("news", "KeywordFilter (контрольные примеры)", offline_filter)

    def real_filter():
        if not items_all:
            raise _Skip("нет реальных новостей")
        hits = [(i, flt.match(i.title, i.summary)) for i in items_all]
        hits = [(i, m) for i, m in hits if m]
        SHARED["news_hit"] = hits[0] if hits else None
        return "PASS", f"из {len(items_all)} реальных новостей совпало с watch-terms: {len(hits)}"

    check("news", "KeywordFilter на реальных RSS", real_filter)

    if a.no_llm_news:
        record("news", "NewsDeepDive (LLM-оценка)", "SKIP", "--no-llm-news")
        return
    shared = cfgmod.load_llm_shared()
    if not shared.get("api_key"):
        record("news", "NewsDeepDive (LLM-оценка)", "SKIP", "нет ключа LLM")
        return
    _SECRETS.append(shared["api_key"])

    def deep():
        mod = load("deepdive")
        llm_cfg = cfgmod.merge_llm_section((cfg.get("news_risk") or {}).get("llm", {}))
        an = pick(mod, "NewsDeepDiveAnalyzer", prefix="News")(llm_cfg)
        if SHARED.get("news_hit"):
            item, tick = SHARED["news_hit"][0], SHARED["news_hit"][1][0]
            origin = "реальная новость"
        else:
            item = rss_mod.RawNewsItem(
                "hc1", "healthcheck", "Сбербанк объявил о рекордной прибыли за квартал",
                "Чистая прибыль выросла, совет директоров рекомендовал дивиденды.",
                "http://x", datetime.now(timezone.utc).isoformat())
            tick, origin = "SBER", "контрольная новость"
        r = an.analyze(tick, item)
        if r is None:
            return "FAIL", "анализатор вернул None (ошибка API или невалидный JSON)"
        rng = (all(0.0 <= float(getattr(r, f)) <= 1.0 for f in ("severity", "materiality"))
               and -1.0 <= float(r.sentiment) <= 1.0)
        return ("PASS" if rng else "WARN",
                f"{origin} {tick}: категория={r.risk_category}, severity={r.severity}, "
                f"sentiment={r.sentiment}, действие={r.action}")

    check("news", "NewsDeepDive (LLM -> JSON -> оценка)", deep, "Смотри блок llm: ключ, модель, сеть.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Finvex: проверка реальных подключений")
    ap.add_argument("--only", nargs="*", choices=BLOCKS, help="какие блоки запускать")
    ap.add_argument("--symbol", default="SBER", help="тикер MOEX/T-Invest для проверок")
    ap.add_argument("--pred-len", type=int, default=24, help="горизонт прогноза Kronos в проверке")
    ap.add_argument("--no-kronos-predict", action="store_true")
    ap.add_argument("--no-llm-news", action="store_true", help="не тратить токены на LLM-оценку новости")
    ap.add_argument("--report", default="data/healthcheck_report.json")
    a = ap.parse_args()

    runners = {"moex": run_moex, "bybit": run_bybit, "tinkoff": run_tinkoff,
               "kronos": run_kronos, "llm": run_llm, "news": run_news}
    print(f"Finvex healthcheck | {datetime.now().isoformat(timespec='seconds')} | root={ROOT}\n")
    for b in (a.only or BLOCKS):
        print(f"-- {b} " + "-" * 60)
        try:
            runners[b](a)
        except Exception as e:  # noqa: BLE001
            if os.environ.get("FINVEX_TRACE"):
                traceback.print_exc()
            record(b, "инициализация блока", "FAIL", f"{type(e).__name__}: {e}",
                   hint="Не найден модуль или зависимость: проверь пути в словаре M и pip install -e .")
        print()

    cnt = {k: sum(r.status == k for r in RESULTS) for k in ICON}
    print("ИТОГ: " + ", ".join(f"{k}={v}" for k, v in cnt.items()))
    rp = ROOT / a.report
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "summary": cnt,
                              "results": [r.__dict__ for r in RESULTS]}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"Отчёт: {rp}")
    return 1 if cnt["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
