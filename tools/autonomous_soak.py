"""Автономный прогон защитного контура: агенты, поток новостей и оценок Kronos, шок и аварийная остановка. Без сети и ключей.

    python tools/autonomous_soak.py --minutes 5          # пять минут реального времени (1 с = 1 минута модельного времени)
    python tools/autonomous_soak.py --minutes 0.3 --out data/soak_smoke

Фазы: A - обычная работа; B - шок (негативные новости и медвежий прогноз по всем бумагам); C - аварийная остановка; D - восстановление.
Новости и оценки Kronos здесь ИМИТАЦИЯ потока (случайные генераторы; модель и разбор новостей не запускаются): проверяется защитный контур,
а не качество прогнозов. Цены синтетические и неподвижны, поэтому доходность не считается.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finvex.core.models import MSK  # noqa: E402
from finvex.gateway.context import PaperPortfolio  # noqa: E402
from finvex.gateway.gateway import Gateway, GatewayError  # noqa: E402
from finvex.gateway.prices import StaticPriceSource  # noqa: E402

UNIVERSE = ["SBER", "GAZP", "LKOH", "MTSS", "ROSN", "GMKN", "BTCUSDT", "ETHUSDT"]
STYLES = {"cautious": 0.35, "trend": 0.55, "meanrev": 0.55, "aggressive": 0.8}   # доля покупок
PHASES = [("A", "обычная работа", 0.0), ("B", "шок: негативные новости и Kronos", 0.40), ("C", "аварийная остановка", 0.70), ("D", "восстановление", 0.83)]
SHOCK_ID = "shock"


class VClock:
    """Модельное время: speed секунд модельного времени на одну реальную."""

    def __init__(self, speed: float) -> None:
        self.t0, self.speed = time.time(), speed
        self.start = datetime(2026, 10, 26, 10, 30, tzinfo=MSK)

    def __call__(self) -> datetime:
        return self.start + timedelta(seconds=(time.time() - self.t0) * self.speed)

    def real(self) -> float:
        return time.time() - self.t0


def make_intent(rng: random.Random, gw, clock, style: str, k: int, tag: str) -> tuple:
    """Возвращает (заявка, метка, тип изъяна). Метка honest/flawed определена при построении, а не по ответу шлюза."""
    now = clock()
    sym = rng.choice(UNIVERSE)
    side = "buy" if rng.random() < STYLES[style] else "sell"
    px = float(gw.context.prices.candles(sym, "1h", 5, now)["close"].iloc[-1])
    crypto = sym.endswith("USDT")
    qty = max(round(rng.uniform(5_000, 60_000) / px, 4 if crypto else 0), 0.0001 if crypto else 1)
    raw = dict(client_id=f"{tag}-{k}", symbol=sym, side=side, quantity=qty, order_type="market", limit_price=None,
               confidence=round(rng.uniform(0.6, 0.9), 2), rationale="Сигнал по режиму рынка, риск ограничен стопом, объём небольшой.")
    flaw = None
    if rng.random() < 0.30:
        flaw = rng.choice(["oversize", "hallucinated", "price_typo", "low_confidence"])
        if flaw == "oversize":
            raw["quantity"] = round(max(qty * rng.uniform(8, 40), 250_000 / px), 4)   # заведомо больше лимита суммы заявки
        elif flaw == "hallucinated":
            raw["symbol"] = rng.choice(["TESLAUSDT", "SBERX", "ZZZZ"])
        elif flaw == "price_typo":
            raw.update(order_type="limit", limit_price=round(px * rng.choice([0.1, 10, 1.25]), 4))
        else:
            raw["confidence"] = round(rng.uniform(0.05, 0.4), 2)
    return raw, ("flawed" if flaw else "honest"), flaw


class Soak:
    def __init__(self, minutes: float, speed: float, seed: int) -> None:
        self.total, self.seed = minutes * 60.0, seed
        self.clock = VClock(speed)
        pf = PaperPortfolio(700_000, {"SBER": 300.0, "GAZP": 500.0, "LKOH": 20.0, "MTSS": 100.0})
        self.gw = Gateway(None, StaticPriceSource(), pf, clock=self.clock)
        rules = {k: dict(v) for k, v in self.gw.rules.items()}
        for k in ("news_sentiment", "kronos_view"):
            rules[k]["value"] = {**rules[k]["value"], "enforce": True}
        rules["max_orders_per_day"]["value"] = 400   # за модельный день агенты шлют сотню намерений; суточный лимит проверяется отдельно
        self.gw.set_rules(rules)
        self.events, self.errors, self.lock, self.stop = [], [], threading.Lock(), threading.Event()
        self.shocked: set = set()
        self.kronos_state = {s: 0.0 for s in UNIVERSE}

    def phase(self) -> str:
        frac, cur = self.clock.real() / self.total, "A"
        for name, _, start in PHASES:
            if frac >= start:
                cur = name
        return cur

    def guard(self, fn):
        def run():
            try:
                fn()
            except Exception as exc:   # поток не должен молча умирать
                self.errors.append(f"{getattr(fn, '__name__', 'thread')}: {type(exc).__name__}: {exc}")
        return run

    def agent(self, name: str, idx: int) -> None:
        rng = random.Random(self.seed * 100 + idx)
        ag, _ = self.gw.agents.create(name, "shadow", "standard", daily_quota=10**6)
        k = 0
        while not self.stop.is_set():
            k += 1
            raw, label, flaw = make_intent(rng, self.gw, self.clock, name, k, name)
            v = self.submit(ag, name, raw, label, flaw)
            if v["decision"] == "revise" and v.get("suggested_quantity"):   # послушный агент исправляет объём по подсказке шлюза
                self.submit(ag, name, {**raw, "client_id": raw["client_id"] + "r", "quantity": v["suggested_quantity"]}, "corrected", flaw, retry=True)
            self.stop.wait(rng.uniform(1.5, 4.0))

    def submit(self, ag, name, raw, label, flaw, retry=False) -> dict:
        t0 = time.perf_counter()
        ph, vt = self.phase(), self.clock()
        try:
            v = self.gw.propose(ag, raw)
        except GatewayError as exc:
            v = {"decision": "deny", "codes": ["input"], "reasons": [str(exc)], "suggested_quantity": None, "context": {}}
        ms = (time.perf_counter() - t0) * 1000
        used = (v.get("context") or {}).get("analytics_used") or {}
        with self.lock:
            self.events.append(dict(t=round(self.clock.real(), 2), vtime=vt.strftime("%H:%M"), phase=ph, agent=name, symbol=raw["symbol"], side=raw["side"],
                                    label=label, flaw=flaw, retry=retry, decision=v["decision"], codes=v.get("codes", []), ms=round(ms, 2),
                                    news=used.get("news_score"), kronos=used.get("kronos_score")))
        return v

    def news_feed(self) -> None:
        rng, n = random.Random(self.seed + 1), 0
        while not self.stop.is_set():
            n += 1
            sym, r = rng.choice(UNIVERSE), rng.random()
            score = rng.uniform(0.3, 0.8) if r < 0.4 else (rng.uniform(-0.15, 0.15) if r < 0.7 else rng.uniform(-0.7, -0.3))
            self.gw.analytics.add_news(sym, {"id": f"n{n}", "score": round(score, 2), "materiality": round(rng.uniform(0.4, 1), 2),
                                             "category": rng.choice(["rumor", "company", "earnings", "macro"]), "ttl_hours": rng.uniform(0.2, 6),
                                             "title": "новость (имитация)"})
            self.stop.wait(rng.uniform(5, 9))

    def kronos_feed(self) -> None:
        rng = random.Random(self.seed + 2)
        while not self.stop.is_set():
            for s in UNIVERSE:
                self.kronos_state[s] = max(-0.7, min(0.7, 0.6 * self.kronos_state[s] + rng.gauss(0, 0.25)))
                score = -0.65 if (self.phase() == "B" and s in self.shocked) else self.kronos_state[s]
                self.gw.analytics.set_kronos(s, round(score, 3), (self.clock() + timedelta(hours=1)).isoformat())
            self.stop.wait(15)

    def controller(self) -> None:
        marks, done = {name: start * self.total for name, _, start in PHASES}, set()
        while not self.stop.is_set():
            t = self.clock.real()
            if "B" not in done and t >= marks["B"]:
                done.add("B")
                self.shocked = set(UNIVERSE)
                for s in UNIVERSE:
                    self.gw.analytics.add_news(s, {"id": f"{SHOCK_ID}-{s}", "score": -0.9, "materiality": 1.0, "category": "company",
                                                  "ttl_hours": 1.5, "title": "шоковая новость (имитация)"})
            if "C" not in done and t >= marks["C"]:
                done.add("C")
                self.set_kill(True)
            if "D" not in done and t >= marks["D"]:
                done.add("D")
                self.set_kill(False)
                self.shocked = set()
            self.stop.wait(0.2)

    def set_kill(self, on: bool) -> None:
        r = {k: dict(v) for k, v in self.gw.rules.items()}
        r["kill_switch"]["enabled"] = on
        self.gw.set_rules(r)

    def run(self) -> dict:
        threads = [threading.Thread(target=self.guard(self.news_feed), daemon=True), threading.Thread(target=self.guard(self.kronos_feed), daemon=True),
                   threading.Thread(target=self.guard(self.controller), daemon=True)]
        threads += [threading.Thread(target=self.guard(lambda n=n, i=i: self.agent(n, i)), daemon=True) for i, n in enumerate(STYLES)]
        for t in threads:
            t.start()
        while self.clock.real() < self.total and not self.errors:
            time.sleep(0.5)
        self.stop.set()
        for t in threads:
            t.join(timeout=5)
        return self.summarize()

    def summarize(self) -> dict:
        ev, by = self.events, defaultdict(Counter)
        for e in ev:
            by[e["phase"]][e["decision"]] += 1
        buys_in_shock = [e for e in ev if e["phase"] == "B" and e["side"] == "buy" and e["decision"] == "allow"]
        flawed_through = [e for e in ev if e["label"] == "flawed" and e["decision"] == "allow"]
        honest = [e for e in ev if e["label"] == "honest"]
        ms = sorted(e["ms"] for e in ev)
        shock_left = [s for s in UNIVERSE if f"{SHOCK_ID}-{s}" in (self.gw.analytics.snapshot(s).get("news_ids") or [])]
        checks = {
            "в шоке покупки по бумагам с негативной аналитикой не проходят": len(buys_in_shock) == 0,
            "при аварийной остановке не разрешено ничего": by["C"]["allow"] == 0 and sum(by["C"].values()) > 0,
            "ни одна заявка с изъяном не прошла": len(flawed_through) == 0,
            "после восстановления заявки снова проходят": (by["D"]["allow"] > 0) if sum(by["D"].values()) else True,
            "шоковые новости к концу прогона удалены по сроку": (len(shock_left) == 0) if self.total >= 170 else True,
            "потоки отработали без сбоев": not self.errors,
        }
        return {"duration_s": round(self.clock.real(), 1), "speed": self.clock.speed, "decisions": len(ev), "per_phase": {k: dict(v) for k, v in by.items()},
                "honest": len(honest), "honest_allowed": sum(e["decision"] == "allow" for e in honest), "flawed": sum(e["label"] == "flawed" for e in ev),
                "corrected": sum(e["label"] == "corrected" for e in ev), "flawed_through": len(flawed_through), "allowed_buys_in_shock": len(buys_in_shock),
                "latency_ms": {"avg": round(sum(ms) / len(ms), 2) if ms else None, "p95": ms[int(0.95 * len(ms))] if ms else None, "max": ms[-1] if ms else None},
                "codes": Counter(c for e in ev for c in e["codes"]).most_common(8), "checks": checks, "errors": self.errors}


def chart(events: list, total: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    bins = max(int(total // 10), 1)
    data = {d: [0] * bins for d in ("allow", "revise", "deny")}
    for e in events:
        data[e["decision"]][min(int(e["t"] // 10), bins - 1)] += 1
    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor="#0b0d10")
    bottom = [0] * bins
    for d, col, name in (("allow", "#2f80ff", "Разрешено"), ("revise", "#8fb8ff", "На доработку"), ("deny", "#e5484d", "Отклонено")):
        ax.bar([b * 10 + 5 for b in range(bins)], data[d], width=9, bottom=bottom, color=col, label=name)
        bottom = [x + y for x, y in zip(bottom, data[d])]
    top = max(bottom) * 1.3 if bottom else 1
    ax.set_ylim(0, top)
    for (name, label, start), nxt in zip(PHASES, PHASES[1:] + [("", "", 1.0)]):
        ax.axvspan(start * total, nxt[2] * total, alpha=0.14, color={"A": "#2f80ff", "B": "#e5484d", "C": "#f2f4f7", "D": "#2f80ff"}[name])
        ax.text((start + nxt[2]) / 2 * total, top * 0.98, f"{name}: {label}", ha="center", va="top", color="#f2f4f7", fontsize=9)
    ax.set_facecolor("#0b0d10")
    ax.set_xlabel("Секунды прогона", color="#f2f4f7")
    ax.set_ylabel("Заявок за 10 с", color="#f2f4f7")
    ax.set_title("Защитный контур в автономном режиме: решения по времени", color="#f2f4f7")
    ax.tick_params(colors="#f2f4f7")
    leg = ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0, 0.86))
    for t in leg.get_texts():
        t.set_color("#f2f4f7")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#0b0d10")
    plt.close(fig)


def report(s: dict, path: Path) -> None:
    ok = all(s["checks"].values())
    L = ["# Автономный прогон защитного контура", "", f"Итог: **{'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if ok else 'ЕСТЬ НАРУШЕНИЯ'}**", "",
         f"- Длительность: {s['duration_s']} с реального времени, ускорение модельного времени x{s['speed']:g}",
         f"- Решений: {s['decisions']}; честных {s['honest']}, с изъяном {s['flawed']}, исправленных по подсказке шлюза {s['corrected']}",
         f"- Заявок с изъяном прошло: {s['flawed_through']}", f"- Покупок, разрешённых в фазе шока: {s['allowed_buys_in_shock']}",
         f"- Задержка решения шлюза, мс: средняя {s['latency_ms']['avg']}, p95 {s['latency_ms']['p95']}, максимум {s['latency_ms']['max']}", "",
         "| Фаза | Разрешено | На доработку | Отклонено |", "|---|---|---|---|"]
    for ph, label, _ in PHASES:
        d = s["per_phase"].get(ph, {})
        L.append(f"| {ph}: {label} | {d.get('allow', 0)} | {d.get('revise', 0)} | {d.get('deny', 0)} |")
    L += ["", "## Проверки", ""] + [f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in s["checks"].items()]
    L += ["", f"Чаще всего срабатывали правила: {s['codes']}", "", "Новости и оценки Kronos в прогоне имитированы, цены синтетические: проверяется контур, а не качество прогнозов."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=60.0, help="секунд модельного времени за одну реальную")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "data" / "soak" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    soak = Soak(a.minutes, a.speed, a.seed)
    print(f"Автономный прогон на {a.minutes:g} мин. Фазы: " + ", ".join(f"{n} с {int(st * 100)}%" for n, _, st in PHASES), flush=True)
    s = soak.run()
    (out / "events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in soak.events) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    report(s, out / "report.md")
    chart(soak.events, soak.total, out / "chart_timeline.png")
    print((out / "report.md").read_text(encoding="utf-8"))
    sys.exit(0 if all(s["checks"].values()) and not s["errors"] else 1)


if __name__ == "__main__":
    main()
