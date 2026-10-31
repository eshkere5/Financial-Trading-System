"""Сессия «ИИ-ассистент как агент»: решения принимает не скрипт, а ИИ-агент по контексту шлюза. Без сети.

    python tools/agent_session_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finvex.gateway.factory import build   # noqa: E402

gw = build(".", demo=True)
me, _ = gw.agents.create("Ассистент (агент)", "shadow", "full", daily_quota=1000)
base = dict(order_type="market", limit_price=None)


def show(tag, raw):
    v = gw.propose(me, raw)
    print(f"[{tag}] {raw['side']} {raw['symbol']} {raw['quantity']} -> {v['decision'].upper()} | коды {v['codes']} | {v['agent_message'][:200]}")
    return v


for sym in ("SBER", "BTCUSDT", "VTBR", "GAZP"):
    c = gw.get_context(me, sym)
    print(sym, {k: c.get(k) for k in ("price", "regime", "adx", "rsi14", "drawdown_60d_pct", "position_qty")})
show("перепроданность VTBR", dict(client_id="s1", symbol="VTBR", side="buy", quantity=300, confidence=0.7, rationale="RSI 7 и просадка 31%: перепроданность, ожидаю отскок.", **base))
show("тренд BTC", dict(client_id="s2", symbol="BTCUSDT", side="buy", quantity=0.05, confidence=0.65, rationale="Бычий тренд, ADX 24.6, беру малый объём из-за высокого RSI.", **base))
show("сокращение GAZP", dict(client_id="s3", symbol="GAZP", side="sell", quantity=200, confidence=0.6, rationale="Медвежий тренд, сокращаю позицию до подтверждения разворота.", **base))
v = show("жадный объём SBER", dict(client_id="s4", symbol="SBER", side="buy", quantity=5000, confidence=0.8, rationale="Диапазон, но хочу набрать позицию крупно.", **base))
if v["decision"] == "revise" and v["suggested_quantity"]:
    show("исправил по подсказке", dict(client_id="s4b", symbol="SBER", side="buy", quantity=v["suggested_quantity"], confidence=0.8, rationale="Исправил объём до допустимого шлюзом.", **base))
rules = {k: dict(x) for k, x in gw.rules.items()}
for k in ("news_sentiment", "kronos_view"):
    rules[k]["value"] = {**rules[k]["value"], "enforce": True}
gw.set_rules(rules)
gw.analytics.add_news("ETHUSDT", {"id": "sec", "score": -0.9, "category": "company", "ttl_hours": 2, "title": "Претензии регулятора к бирже (имитация)"})
show("ETH после плохой новости", dict(client_id="s5", symbol="ETHUSDT", side="buy", quantity=0.1, confidence=0.7, rationale="Продолжение тренда, добавляю.", **base))
show("выдуманный тикер", dict(client_id="s6", symbol="DOGE2USDT", side="buy", quantity=1, confidence=0.7, rationale="Слышал в чате про рост на 300%.", **base))
