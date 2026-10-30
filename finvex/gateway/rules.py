"""Правила как данные. Каждое правило проверяется кодом, а не просьбой к модели."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Callable, Optional

from .models import Check, Intent

WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")
CRYPTO_SUFFIXES = ("USDT", "USDC", "PERP")

RULES_META = [
    dict(id="require_rationale", title="Обоснование обязательно", kind="number", unit="симв.", default=20,
         desc="Агент обязан объяснить решение. Минимальная длина текста."),
    dict(id="allowed_symbols", title="Разрешённые бумаги", kind="list", unit="", default=[],
         desc="Пусто значит любые. Иначе только перечисленные через запятую."),
    dict(id="blocked_symbols", title="Запрещённые бумаги", kind="list", unit="", default=[],
         desc="Эти тикеры агенту недоступны."),
    dict(id="no_short", title="Без коротких продаж", kind="flag", unit="", default=True,
         desc="Нельзя продать больше, чем есть в позиции."),
    dict(id="max_order_value", title="Максимальная сумма заявки", kind="number", unit="₽", default=200000,
         desc="Если сумма больше, шлюз предложит уменьшенный объём."),
    dict(id="max_position_pct", title="Максимальная доля бумаги", kind="number", unit="% портфеля", default=20,
         desc="Доля позиции в портфеле после сделки."),
    dict(id="max_orders_per_day", title="Заявок в сутки", kind="number", unit="шт.", default=20,
         desc="Лимит намерений одного агента за календарный день."),
    dict(id="trading_hours", title="Только торговая сессия", kind="flag", unit="", default=True,
         desc="Акции MOEX: будни с 10:00 до 18:40 по Москве. Крипто круглосуточно."),
    dict(id="price_deviation", title="Отклонение лимитной цены", kind="number", unit="%", default=3,
         desc="Защита от опечатки в цене: лимитная цена не дальше от последней."),
    dict(id="data_freshness", title="Свежесть данных", kind="number", unit="мин.", default=180,
         desc="В сессию последняя свеча не должна быть старше."),
    dict(id="regime_block_buy", title="Режимы без покупок", kind="list", unit="", default=["CRASH", "RALLY_FRENZY"],
         desc="Если детектор режима видит эти состояния, покупки запрещены."),
    dict(id="min_confidence", title="Минимальная уверенность", kind="number", unit="0..1", default=0.55,
         desc="Если агент указал уверенность ниже порога, нужна доработка."),
    dict(id="cooldown", title="Пауза по бумаге", kind="number", unit="сек.", default=300,
         desc="Повтор того же действия по той же бумаге раньше срока блокируется."),
]
EFFECTS = {  # что делает правило при нарушении: deny - отказ, revise - на доработку, warn - только предупреждение
    "kill_switch": "deny", "require_rationale": "warn", "max_rationale": "revise", "allowed_symbols": "deny",
    "blocked_symbols": "deny", "no_short": "mixed", "max_order_value": "mixed", "max_position_pct": "mixed",
    "max_orders_per_day": "revise", "trading_hours": "revise", "price_deviation": "revise", "data_freshness": "revise",
    "regime_block_buy": "deny", "min_confidence": "revise", "cooldown": "revise",
    "news_sentiment": "bands", "kronos_view": "bands", "max_revisions": "deny",
}
REGIME_INFO = [
    ("CRASH", "Обвал", "Резкое падение: просадка от максимума 60 баров не меньше 10% при очень высокой волатильности."),
    ("RALLY_FRENZY", "Перегретый рост", "Сильный тренд вверх (ADX от 25), цена заметно выше SMA200. Часто предшествует откату."),
    ("RECOVERY", "Отскок в медвежьем рынке", "Цена выше SMA50 и SMA50 растёт, но цена всё ещё ниже SMA200."),
    ("BEAR_TREND", "Нисходящий тренд", "Цена ниже SMA50, и сама SMA50 направлена вниз."),
    ("BULL_TREND", "Восходящий тренд", "Цена выше SMA50 и SMA200, SMA50 растёт, ADX от 20."),
    ("HIGH_VOL_CHOP", "Хаотичные качели", "Волатильность в верхних 30% за год, чёткого тренда нет."),
    ("RANGE_BOUND", "Боковик", "Цена ходит в диапазоне без выраженного тренда и без экстремальной волатильности."),
    ("RANGE_SQUEEZE", "Сжатие", "Волатильность и ширина полос Боллинджера в нижней четверти за год. Часто перед сильным движением."),
    ("TRANSITION", "Переход", "Мало данных или признаки смешаны: режим определить нельзя."),
]
RULES_META.insert(0, dict(id="kill_switch", title="Аварийная остановка", kind="flag", unit="", default=True, enabled=False,
                          desc="Когда включено, шлюз отклоняет все новые намерения всех агентов."))
RULES_META.insert(2, dict(id="max_rationale", title="Максимальная длина обоснования", kind="number", unit="симв.", default=1500,
                          desc="Слишком длинный текст возвращается на доработку: через обоснование можно протащить чужие инструкции."))
RULES_META.append(dict(id="max_revisions", title="Лимит доработок", kind="number", unit="раз за час", default=3,
                       desc="Сколько раз агент может вернуть ту же бумагу и сторону на доработку. Дальше отказ: так нельзя продавить заявку мелкими правками."))
RULES_META += [
    dict(id="news_sentiment", title="Новостной фон по активу", kind="band", unit="-1..1", default={"revise": -0.2, "deny": -0.5, "enforce": False},
         below=True, desc="Среднее по новостям, которые парсер ещё считает актуальными. Для покупок: не выше порога доработки значит на доработку, не выше порога отказа значит отказ."),
    dict(id="kronos_view", title="Прогноз Kronos", kind="band", unit="-0.7..0.7", default={"revise": -0.2, "deny": -0.5, "enforce": False},
         below=True, desc="Оценка по свечам, обновляется при закрытии свечи и действует до закрытия следующей. Выборок мало, поэтому это мягкий фильтр."),
]
for m in RULES_META:
    m["effect"] = EFFECTS[m["id"]]
    if m["id"] == "regime_block_buy":
        m["options"] = [dict(id=i, title=t, desc=d) for i, t, d in REGIME_INFO]
META = {m["id"]: m for m in RULES_META}


def default_rules() -> dict:
    return {m["id"]: {"enabled": m.get("enabled", True), "value": copy.deepcopy(m["default"])} for m in RULES_META}


def normalize(rules: dict) -> dict:
    out = default_rules()
    for rid, spec in (rules or {}).items():
        if rid not in META:
            raise ValueError(f"неизвестное правило: {rid}")
        meta = META[rid]
        val = spec.get("value", out[rid]["value"])
        if meta["kind"] == "number":
            try:
                val = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"{rid}: нужно число")
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"{rid}: число должно быть >= 0")
            if val == int(val) and rid != "min_confidence":
                val = int(val)
        elif meta["kind"] == "band":
            val = _norm_band(rid, meta, val if isinstance(val, dict) else {})
        elif meta["kind"] == "list":
            if isinstance(val, str):
                val = [s for s in val.replace(";", ",").replace(" ", ",").split(",") if s]
            val = [str(s).strip().upper() for s in val if str(s).strip()]
        else:
            val = bool(val)
        out[rid] = {"enabled": bool(spec.get("enabled", True)), "value": val}
    return out


def _norm_band(rid: str, meta: dict, val: dict) -> dict:
    d = {**meta["default"], **val}
    try:
        rev, den = float(d["revise"]), float(d["deny"])
    except (TypeError, ValueError):
        raise ValueError(f"{rid}: пороги должны быть числами")
    if not (math.isfinite(rev) and math.isfinite(den)):
        raise ValueError(f"{rid}: пороги должны быть конечными числами")
    if meta["below"]:
        if not (-1 <= den <= rev <= 1):
            raise ValueError(f"{rid}: порог отказа не должен быть выше порога доработки, оба в пределах -1..1")
    elif not (0 < rev <= den):
        raise ValueError(f"{rid}: порог доработки должен быть меньше порога отказа, оба больше нуля")
    return {"revise": rev, "deny": den, "enforce": bool(d.get("enforce"))}


def mandate_hash(rules: dict) -> str:
    """Отпечаток действующих правил: любое изменение лимита меняет хеш."""
    live = {k: v["value"] for k, v in sorted(rules.items()) if v.get("enabled")}
    return hashlib.sha256(json.dumps(live, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_rules(path: "str | Path | None") -> dict:
    if path and Path(path).exists():
        return normalize(json.loads(Path(path).read_text(encoding="utf-8")))
    return default_rules()


def save_rules(path: "str | Path | None", rules: dict) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")


def merge(base: dict, overrides: Optional[dict]) -> dict:
    return normalize({**{k: dict(v) for k, v in base.items()}, **(overrides or {})}) if overrides else base


def is_crypto(symbol: str) -> bool:
    return symbol.endswith(CRYPTO_SUFFIXES)


def next_open(now: datetime, symbol: str) -> datetime:
    """Ближайшее открытие сессии акций (10:00 МСК в будний день)."""
    if is_crypto(symbol):
        return now
    d = now
    if d.weekday() < 5 and d.timetz().replace(tzinfo=None) < time(10, 0):
        return d.replace(hour=10, minute=0, second=0, microsecond=0)
    d = (d + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def session_open(now: datetime, symbol: str) -> bool:
    if is_crypto(symbol):
        return True
    return now.weekday() < 5 and time(10, 0) <= now.timetz().replace(tzinfo=None) <= time(18, 40)


@dataclass
class State:
    ctx: dict
    now: datetime
    orders_today: int
    last_same: Optional[datetime]
    position_qty: float
    equity: float
    analytics: dict = field(default_factory=dict)
    revisions: int = 0   # доработок по этой бумаге и стороне за последний час


def _chk(rid, status, msg, severity="block", qty=None, retry=None) -> Check:
    return Check(rid, META[rid]["title"], status, msg, severity, qty, retry)


def _r_kill(i, p, s):
    return _chk("kill_switch", "fail", "Аварийная остановка включена: новые намерения не принимаются")


def _r_rationale(i: Intent, p, s: State) -> Check:
    n = len(i.rationale.strip())
    if n >= p:
        return _chk("require_rationale", "pass", f"Обоснование есть ({n} симв.)")
    return _chk("require_rationale", "warn", f"Обоснование короче {p:g} символов: это не блокирует заявку, но решение хуже объяснимо")


def _r_rat_max(i, p, s):
    n = len(i.rationale)
    if n <= p:
        return _chk("max_rationale", "pass", f"Длина обоснования {n} из {p:g}")
    return _chk("max_rationale", "fail", f"Обоснование {n} симв. при максимуме {p:g}: сократите до сути", "revise")


def _r_allowed(i, p, s):
    if not p:
        return _chk("allowed_symbols", "skip", "Список не задан")
    ok = i.symbol in p
    return _chk("allowed_symbols", "pass" if ok else "fail", "Бумага в списке" if ok else f"{i.symbol} нет в списке разрешённых")


def _r_blocked(i, p, s):
    bad = i.symbol in p
    return _chk("blocked_symbols", "fail" if bad else "pass", f"{i.symbol} запрещена" if bad else "Бумага не запрещена")


def _r_short(i, p, s):
    if i.side == "sell" and i.quantity > s.position_qty + 1e-9:
        return _chk("no_short", "fail", f"Продажа {i.quantity:g} при позиции {s.position_qty:g}: это короткая продажа",
                    "revise" if s.position_qty > 0 else "block", s.position_qty if s.position_qty > 0 else None)
    return _chk("no_short", "pass" if i.side == "sell" else "skip", "Продажа в пределах позиции" if i.side == "sell" else "Покупка")


def _ref_price(i: Intent, s: State) -> Optional[float]:
    return i.limit_price if i.order_type == "limit" and i.limit_price else s.ctx.get("price")


def _r_order_value(i, p, s):
    px = _ref_price(i, s)
    if not px:
        return _chk("max_order_value", "fail", "Нет цены для оценки суммы")
    val = px * i.quantity
    if val <= p:
        return _chk("max_order_value", "pass", f"Сумма {val:,.0f} ₽ в пределах {p:,.0f} ₽".replace(",", " "))
    sug = math.floor(p / px)
    return _chk("max_order_value", "fail", f"Сумма {val:,.0f} ₽ больше лимита {p:,.0f} ₽".replace(",", " "),
                "revise" if sug >= 1 else "block", float(sug) if sug >= 1 else None)


def _r_position(i, p, s):
    px = _ref_price(i, s)
    if i.side != "buy" or not px or s.equity <= 0:
        return _chk("max_position_pct", "skip", "Проверяется только для покупок")
    cap = s.equity * p / 100.0
    after = (s.position_qty + i.quantity) * px
    if after <= cap:
        return _chk("max_position_pct", "pass", f"Доля после сделки {after / s.equity * 100:.1f}% при лимите {p:g}%")
    room = math.floor(max(cap / px - s.position_qty, 0))
    return _chk("max_position_pct", "fail", f"Доля после сделки {after / s.equity * 100:.1f}% больше лимита {p:g}%",
                "revise" if room >= 1 else "block", float(room) if room >= 1 else None)


def _r_per_day(i, p, s):
    if s.orders_today < p:
        return _chk("max_orders_per_day", "pass", f"Сегодня {s.orders_today} из {p:g}")
    tomorrow = (s.now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return _chk("max_orders_per_day", "fail", f"Суточный лимит {p:g} намерений исчерпан", "revise", None, (tomorrow - s.now).total_seconds())


def _r_hours(i, p, s):
    if session_open(s.now, i.symbol):
        return _chk("trading_hours", "pass", "Круглосуточный рынок" if is_crypto(i.symbol) else "Торговая сессия идёт")
    nxt = next_open(s.now, i.symbol)
    return _chk("trading_hours", "fail", f"Вне сессии ({WEEKDAYS[s.now.weekday()]} {s.now:%H:%M} МСК), открытие {WEEKDAYS[nxt.weekday()]} {nxt:%H:%M}",
                "revise", None, max((nxt - s.now).total_seconds(), 0.0))


def _r_deviation(i, p, s):
    last = s.ctx.get("price")
    if i.order_type != "limit" or not last:
        return _chk("price_deviation", "skip", "Рыночная заявка" if i.order_type != "limit" else "Нет цены")
    dev = abs(i.limit_price / last - 1) * 100
    if dev <= p:
        return _chk("price_deviation", "pass", f"Отклонение {dev:.2f}% от последней цены {last:g}")
    return _chk("price_deviation", "fail", f"Лимитная цена на {dev:.1f}% от последней {last:g}: похоже на опечатку", "revise")


def _r_fresh(i, p, s):
    age = s.ctx.get("data_age_min")
    if age is None:
        return _chk("data_freshness", "fail", "Нет времени последней свечи")
    if not session_open(s.now, i.symbol):
        return _chk("data_freshness", "skip", "Рынок закрыт, свежесть не проверяется")
    if age <= p:
        return _chk("data_freshness", "pass", f"Данные {age:.0f} мин. давности")
    return _chk("data_freshness", "fail", f"Данные устарели на {age:.0f} мин. при допуске {p:g}", "revise", None, 300.0)


def _r_regime(i, p, s):
    reg = s.ctx.get("regime")
    if i.side != "buy":
        return _chk("regime_block_buy", "skip", "Проверяется только для покупок")
    if reg is None:
        return _chk("regime_block_buy", "warn", "Режим рынка не определён", "revise")
    if reg in p:
        return _chk("regime_block_buy", "fail", f"Режим {reg}: покупки запрещены")
    return _chk("regime_block_buy", "pass", f"Режим {reg}")


def _r_conf(i, p, s):
    if i.confidence is None:
        return _chk("min_confidence", "warn", "Агент не указал уверенность")
    if i.confidence >= p:
        return _chk("min_confidence", "pass", f"Уверенность {i.confidence:.2f}")
    return _chk("min_confidence", "fail", f"Уверенность {i.confidence:.2f} ниже порога {p:g}", "revise")


def _band(rid, p, s, i, key_score, label):
    """Три зоны по коэффициенту. Нет актуальных данных - оценка нейтральная, это не причина для остановки."""
    if i.side != "buy":
        return _chk(rid, "skip", "Проверяется только для покупок")
    score = (s.analytics or {}).get(key_score)
    if score is None:
        return _chk(rid, "skip", f"{label}: актуальных оценок нет, считается нейтральным")
    note = "" if p["enforce"] else " (режим предупреждения: на решение не влияет)"
    if score <= p["deny"]:
        return _chk(rid, "fail" if p["enforce"] else "warn", f"{label} {score:+.2f} не выше порога отказа {p['deny']:+g}{note}", "block")
    if score <= p["revise"]:
        return _chk(rid, "fail" if p["enforce"] else "warn", f"{label} {score:+.2f} не выше порога доработки {p['revise']:+g}{note}", "revise")
    return _chk(rid, "pass", f"{label} {score:+.2f}")


def _r_news(i, p, s):
    return _band("news_sentiment", p, s, i, "news_score", "Новостной фон")


def _r_kronos(i, p, s):
    return _band("kronos_view", p, s, i, "kronos_score", "Прогноз Kronos")


def _r_revcap(i, p, s):
    return _chk("max_revisions", "skip", f"Доработок за час: {s.revisions} из {p:g}")


def _r_cooldown(i, p, s):
    if s.last_same is None:
        return _chk("cooldown", "pass", "Повторов нет")
    dt = (s.now - s.last_same).total_seconds()
    if dt >= p:
        return _chk("cooldown", "pass", f"Прошло {dt:.0f} сек.")
    return _chk("cooldown", "fail", f"Повтор через {dt:.0f} сек. при паузе {p:g}", "revise", None, max(p - dt, 1.0))


CHECKS: "dict[str, Callable]" = {
    "max_revisions": _r_revcap,
    "kill_switch": _r_kill, "max_rationale": _r_rat_max, "news_sentiment": _r_news, "kronos_view": _r_kronos,
    "require_rationale": _r_rationale, "allowed_symbols": _r_allowed, "blocked_symbols": _r_blocked,
    "no_short": _r_short, "max_order_value": _r_order_value, "max_position_pct": _r_position,
    "max_orders_per_day": _r_per_day, "trading_hours": _r_hours, "price_deviation": _r_deviation,
    "data_freshness": _r_fresh, "regime_block_buy": _r_regime, "min_confidence": _r_conf, "cooldown": _r_cooldown,
}
QTY_RULES = ("max_order_value", "max_position_pct", "no_short")


def evaluate(rules: dict, intent: Intent, state: State) -> "tuple[str, list, Optional[float]]":
    checks, decision, suggested = [], "allow", None
    for meta in RULES_META:
        rid, spec = meta["id"], rules[meta["id"]]
        if not spec["enabled"]:
            continue
        try:
            c = CHECKS[rid](intent, spec["value"], state)
        except Exception as exc:  # правило не смогло посчитаться: закрываем, а не пропускаем
            c = _chk(rid, "fail", f"Правило не удалось проверить: {exc}")
        checks.append(c)
        if c.status == "fail":
            if c.severity == "block":
                decision = "deny"
            elif decision != "deny":
                decision = "revise"
            if c.suggested_quantity is not None:
                suggested = c.suggested_quantity if suggested is None else min(suggested, c.suggested_quantity)
    cap = rules.get("max_revisions", {})
    if decision == "revise" and cap.get("enabled") and state.revisions >= cap["value"]:
        checks.append(_chk("max_revisions", "fail", f"Лимит доработок исчерпан: {state.revisions} за час при допустимых {cap['value']:g}. Пересмотрите идею целиком", "block"))
        decision = "deny"
    if decision != "revise":
        suggested = None
    if suggested is not None:
        probe = Intent(**{**intent.to_dict(), "quantity": suggested})
        if any(rules[r]["enabled"] and CHECKS[r](probe, rules[r]["value"], state).status == "fail" for r in QTY_RULES):
            suggested = None
    return decision, checks, suggested
