"""Решение агента: разбор ответа, аналитик без модели и вызов агента по адресу (n8n, Dify, свой сервис).

Единый формат решения: {"action": "buy|sell|hold", "quantity": число, "order_type": "market|limit", "limit_price": число или null,
"confidence": 0..1, "rationale": "почему"}. Шлюз проверяет любое решение одинаково, откуда бы оно ни пришло.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

from finvex.core.urlguard import UnsafeUrl, check_llm_url

FORMAT = ("Ответь ТОЛЬКО одним JSON-объектом без пояснений вне него: "
          '{"action":"buy|sell|hold","quantity":число,"order_type":"market|limit","limit_price":число или null,'
          '"confidence":число от 0 до 1,"rationale":"2-3 предложения: почему именно это решение"}. '
          "Если нет явного основания, выбери hold. Количество указывай в штуках (для акций) или в монетах (для крипто). "
          "Не придумывай тикеры и данные: используй только то, что дано в контексте. Шлюз проверит лимиты, но не превышай их сам.")
MAX_RESPONSE = 200_000
RESULT_KEYS = ("output", "text", "message", "result", "response", "data", "json")


class DecisionError(ValueError):
    pass


def extract_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        obj = json.loads(t)
    except ValueError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m:
            raise DecisionError("модель не вернула JSON")
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            raise DecisionError("модель вернула некорректный JSON")
    if not isinstance(obj, dict):
        raise DecisionError("ожидался JSON-объект")
    return obj


def decision_to_intent(dec: dict, symbol: str, prefix: str = "run") -> dict:
    return {"client_id": f"{prefix}-" + uuid.uuid4().hex[:12], "symbol": symbol, "side": str(dec.get("action", "")).lower(),
            "quantity": dec.get("quantity"), "order_type": dec.get("order_type", "market"), "limit_price": dec.get("limit_price"),
            "confidence": dec.get("confidence"), "rationale": dec.get("rationale", "")}


def _find_decision(obj: Any, depth: int = 0) -> Optional[dict]:
    """Ответ n8n, Dify и подобных платформ вкладывает результат по-разному: ищем объект с полем action."""
    if depth > 5:
        return None
    if isinstance(obj, dict):
        if "action" in obj:
            return obj
        for key in RESULT_KEYS:
            if key in obj:
                found = _find_decision(obj[key], depth + 1)
                if found:
                    return found
    elif isinstance(obj, list) and obj:
        return _find_decision(obj[0], depth + 1)
    elif isinstance(obj, str):
        try:
            return _find_decision(extract_json(obj), depth + 1)
        except DecisionError:
            return None
    return None


def call_http(url: str, key: str, payload: dict, timeout: float = 30.0) -> dict:
    """Спрашивает решение у агента по адресу. Только https (или localhost), без перенаправлений, ответ не больше 200 КБ."""
    import httpx
    try:
        check_llm_url(url, resolve=True)
    except UnsafeUrl as exc:
        raise DecisionError(f"адрес агента: {exc}")
    headers = {"Content-Type": "application/json", "User-Agent": "finvex-agent-call"}
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            with client.stream("POST", url, content=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"), headers=headers) as resp:
                if 300 <= resp.status_code < 400:
                    raise DecisionError("агент отвечает перенаправлением: укажите конечный адрес")
                if resp.status_code >= 400:
                    raise DecisionError(f"агент ответил HTTP {resp.status_code}")
                data = bytearray()
                for chunk in resp.iter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_RESPONSE:
                        raise DecisionError("ответ агента слишком большой")
    except httpx.TimeoutException:
        raise DecisionError(f"агент не ответил за {timeout:g} с")
    except httpx.HTTPError as exc:
        raise DecisionError(f"агент недоступен: {type(exc).__name__}")
    text = bytes(data).decode("utf-8", "replace")
    try:
        obj = json.loads(text)
    except ValueError:
        obj = text
    dec = _find_decision(obj)
    if dec is None:
        raise DecisionError("в ответе агента нет решения: ожидается JSON с полем action (buy, sell или hold)")
    return dec


def analyst_decision(ctx: dict) -> dict:
    """Аналитик Finvex без модели и ключей: простые прозрачные правила по режиму рынка, ADX и RSI.
    Нужен как эталон для проверки шлюза и как агент, который работает без интернета и платных вызовов."""
    symbol = str(ctx.get("symbol") or "")
    price, equity = float(ctx.get("price") or 0), float(ctx.get("equity") or 0)
    regime, adx, rsi = ctx.get("regime"), float(ctx.get("adx") or 0), ctx.get("rsi14")
    pos = float(ctx.get("position_qty") or 0)

    def hold(why: str) -> dict:
        return {"action": "hold", "quantity": None, "order_type": "market", "limit_price": None, "confidence": 0.5, "rationale": why}

    if price <= 0 or equity <= 0:
        return hold("Нет цены или капитала для расчёта.")
    risk_qty = equity * 0.005 / price
    qty = round(risk_qty, 6) if symbol.upper().endswith(("USDT", "USDC")) else max(1, int(risk_qty))
    state = f"режим {regime}, ADX {adx:g}, RSI {rsi if rsi is not None else 'нет данных'}"
    if pos <= 0 and regime == "BULL_TREND" and adx >= 25 and (rsi is None or rsi < 70):
        return {"action": "buy", "quantity": qty, "order_type": "market", "limit_price": None, "confidence": 0.6,
                "rationale": f"Бычий тренд подтверждён ADX, перекупленности нет ({state}). Вход на 0,5% капитала."}
    if pos > 0 and (regime in ("BEAR_TREND", "CRASH") or (rsi is not None and rsi >= 75)):
        return {"action": "sell", "quantity": pos, "order_type": "market", "limit_price": None, "confidence": 0.6,
                "rationale": f"Тренд сломан или рынок перегрет ({state}). Закрываю позицию."}
    return hold(f"Нет явного сигнала ({state}).")
