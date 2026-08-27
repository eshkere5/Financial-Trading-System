"""
smoke_regime.py — офлайн-проверка режимной связки ПЕРЕД soak'ом.

### NEW 2026-08-24 ###
Проверяет без сети и без DeepSeek:
1. Детектор на реальных окнах истории (sber_daily.csv): печатает режимы
   на известных фазах — крах авг-2024, медвежий 2025, восстановление 2026;
2. Стратегия: tactical_bias + candles -> ордер с режимным множителем
   и строка 'taken' в журнале с заполненным regime;
3. Пропуск (низкая conviction) пишется в журнал как 'skipped' с причиной;
4. Стратегист: _apply_price_regime подменяет posture/size по детектору,
   LLM может понизить, но НЕ может повысить;
5. Читаемость: все записанное читается обратно из data/smoke_test.db.

Запуск (из корня проекта):
    python analysis\smoke_regime.py
Ожидание: в конце "SMOKE: все проверки прошли".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "src").is_dir() else _HERE.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.regime.regime_detector import compute_regime, signal_multiplier, RegimeInfo  # noqa: E402

PASS = True


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        PASS = False


print("=" * 60)
print("1. Детектор на реальных окнах истории (SBER)")
print("=" * 60)
csv_path = "analysis/results/sber_daily.csv"
if not os.path.exists(csv_path):
    csv_path = "sber_daily.csv"
df = pd.read_csv(csv_path)
closes = df["close"].astype(float)
times = df["time"] if "time" in df.columns else df.iloc[:, 0]

probe_dates = {"2024-08-05": "carry-крах", "2025-04-07": "тарифный обвал",
               "2025-06-20": "середина 2025", "2026-03-20": "весна 2026",
               "2026-08-21": "последний бар"}
for date, label in probe_dates.items():
    idx = times[times == date].index
    if len(idx) == 0:
        continue
    i = int(idx[0]) + 1
    info = compute_regime(closes.iloc[:i])
    print(f"  {date} ({label}): {info.regime}  dist={info.close_vs_sma50:+.3f} "
          f"vol_p={info.vol_pctile:.2f} dd={info.drawdown:+.3f}")

print("\n" + "=" * 60)
print("2. Стратегия: сигнал проходит через режимный множитель + журнал")
print("=" * 60)

from src.signals.ai_tactic_strategy import AITacticStrategy, AITacticConfig
from src.engine.trade_journal import TradeJournal

journal = TradeJournal("data/smoke_test.db")

# окно: медвежья фаза (должен быть BEAR/CRASH — шорты усилены)
i_bear = int(times[times == "2025-04-07"].index[0]) + 1 if (times == "2025-04-07").any() else 400
win = df.iloc[max(0, i_bear - 400):i_bear].copy()

strat = AITacticStrategy(cfg=AITacticConfig(qty_per_trade=100.0, min_conviction=0.35),
                         kronos=None, ohlcv={"SBER": win})
strat.set_journal(journal)

last_price = float(win["close"].iloc[-1])
state = SimpleNamespace(
    prices={"SBER": last_price},
    positions={},
    get_news=lambda t: None,   # новостей нет — чистый режимный тест
)

strat.set_tactical_bias({"SBER": {"direction": -1, "conviction": 0.9, "horizon": "short"}})
orders = strat.on_bar(state)
regime_now = strat._regime_cache["SBER"][1]
mult_expected = signal_multiplier(regime_now.regime, -1, ticker="SBER")

check("ордер создан", len(orders) == 1, f"(режим={regime_now.regime})")
if orders:
    qty_expected = 100.0 * min(1.0, 0.9) * mult_expected
    check("qty срежимирован", abs(orders[0].qty - qty_expected) < 1e-6,
          f"qty={orders[0].qty:.2f} vs ожидание {qty_expected:.2f} (mult={mult_expected})")

print("\n" + "=" * 60)
print("3. Пропуск пишется в журнал с причиной")
print("=" * 60)
strat2 = AITacticStrategy(cfg=AITacticConfig(qty_per_trade=100.0, min_conviction=0.35),
                          kronos=None, ohlcv={"SBER": win})
strat2.set_journal(journal)
strat2.set_tactical_bias({"SBER": {"direction": 1, "conviction": 0.10, "horizon": "short"}})
orders2 = strat2.on_bar(state)
check("слабый сигнал пропущен", len(orders2) == 0)

rows = journal._db.execute(
    "SELECT ticker, action, skip_reason, regime, direction_multiplier FROM decisions").fetchall()
taken = [r for r in rows if r[1] == "taken"]
skipped = [r for r in rows if r[1] == "skipped"]
check("в журнале есть taken с режимом", any(r[3] for r in taken),
      f"taken={len(taken)}, regime={taken[0][3] if taken else '—'}")
check("в журнале есть skipped с причиной", any(r[2] for r in skipped),
      f"skipped={len(skipped)}, reason={skipped[0][2] if skipped else '—'}")

print("\n" + "=" * 60)
print("4. Стратегист: детектор подменяет режим, LLM может только понижать")
print("=" * 60)
from src.llm_strategist.strategist import LLMStrategist, StrategistDecision

st = LLMStrategist({})   # без ключа — только тестируем _apply_price_regime
crash = {"BTCUSDT": RegimeInfo("CRASH", -0.12, -0.03, 0.95, -0.15, 0.8)}

d1 = StrategistDecision(regime="trend", risk_posture="normal", size_multiplier=1.0, rationale="llm says calm")
out1 = st._apply_price_regime(d1, crash)
check("CRASH форсирует halt", out1.risk_posture == "halt" and out1.size_multiplier == 0.0,
      f"posture={out1.risk_posture} size={out1.size_multiplier}")

d2 = StrategistDecision(regime="unknown", risk_posture="halt", size_multiplier=0.3, rationale="llm panic")
bull = {"BTCUSDT": RegimeInfo("BULL_TREND", 0.05, 0.01, 0.5, -0.02, 0.4)}
out2 = st._apply_price_regime(d2, bull)
check("LLM может понизить (BULL->halt)", out2.risk_posture == "halt",
      f"posture={out2.risk_posture}")

d3 = StrategistDecision(regime="trend", risk_posture="normal", size_multiplier=1.5, rationale="llm greedy")
out3 = st._apply_price_regime(d3, crash)
check("LLM НЕ может повысить (CRASH остался halt)", out3.risk_posture == "halt",
      f"posture={out3.risk_posture}")

d4 = StrategistDecision(risk_posture="normal", size_multiplier=1.0)
out4 = st._apply_price_regime(d4, None)
check("без price_regimes — обратная совместимость", out4.risk_posture == "normal" and out4.regime == "unknown")

journal.close()
print("\n" + ("SMOKE: все проверки прошли" if PASS else "SMOKE: ЕСТЬ ПРОВАЛЫ — чини до soak'а"))
sys.exit(0 if PASS else 1)
