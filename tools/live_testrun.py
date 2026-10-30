"""Тестовый прогон агентов: ключ DeepSeek имитирует работу агентов, решения проходят шлюз и (с --send) уходят
в Bybit Testnet / песочницу T-Invest. Деньги виртуальные, боевые ключи не читаются.

Примеры (из корня проекта):
    python tools/live_testrun.py                       # пробный прогон без отправки заявок
    python tools/live_testrun.py --send --rounds 3     # заявки в тестовые контуры
    python tools/live_testrun.py --fake --out data/sample_run   # самопроверка без сети и ключей (синтетика)
Результат: папка data/testruns/<время>/ с events.jsonl, summary.json, report.md; затем tools/make_presentation.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finvex.execution.testnet import TestVenues  # noqa: E402
from finvex.execution.testtrade import send_verdict  # noqa: E402
from finvex.web import sandbox as sbx  # noqa: E402

DEFAULT_AGENTS = "cautious,trend,meanrev,aggressive"
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SBER,GAZP"
CAPS = {"BYBIT": 100.0, "MOEX": 5000.0}   # максимальная сумма одной тестовой заявки: USDT и рубли


def _event(**kw) -> dict:
    return {"ts": datetime.now().isoformat(timespec="seconds"), **kw}


def snapshot(venues: TestVenues) -> dict:
    out = {}
    for s in venues.status():
        if s["ready"]:
            try:
                out[s["venue"]] = venues.get(s["venue"]).portfolio()
            except Exception as exc:
                out[s["venue"]] = {"error": str(exc)[:200]}
    return out


def run(gw, sandbox, venues: TestVenues, agents: list, symbols: list, rounds: int, pause: float, send: bool,
        max_orders: int, caps: dict, log_path: Path, fake: bool = False, say=print) -> dict:
    events, sent_total = [], 0
    log = log_path.open("a", encoding="utf-8")

    def emit(ev: dict) -> None:
        events.append(ev)
        log.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        log.flush()

    before = snapshot(venues) if send else {}
    started = time.time()
    try:
        for r in range(rounds):
            for a, agent in enumerate(agents):
                if send and sent_total >= max_orders:
                    say(f"Достигнут лимит {max_orders} заявок, останавливаюсь.")
                    raise StopIteration
                sym = symbols[(r + a) % len(symbols)]
                base = dict(round=r + 1, agent=agent, symbol=sym)
                try:
                    venue_ = venues.venue_for(sym)
                    cap_note = f"Для тестового прогона сумма одной заявки не должна превышать {caps.get(venue_, 0):,.0f} {'USDT' if venue_ == 'BYBIT' else 'руб.'}."
                    res = sandbox.run(agent, sym, cap_note)
                except Exception as exc:
                    emit(_event(**base, outcome="error", note=str(exc)[:300]))
                    say(f"[{r + 1}] {agent:<10} {sym:<8} ошибка: {exc}")
                    continue
                m, v = res.get("model") or {}, res.get("verdict")
                ev = _event(**base, preset=res.get("preset"), latency_ms=res.get("latency_ms"), tokens=res.get("tokens") or {},
                            action=m.get("action"), quantity=m.get("quantity"), confidence=m.get("confidence"),
                            rationale=(m.get("rationale") or "")[:300], note=res.get("note", ""), sent=False)
                if v:
                    ev.update(outcome=v["decision"], codes=v.get("codes", []), agent_message=v.get("agent_message", ""),
                              verdict_id=v["id"], price=(v.get("context") or {}).get("price"), mandate=(v.get("mandate_hash") or "")[:12])
                    if send and v["decision"] == "allow":
                        notional = float(ev["price"] or 0) * float(v["intent"]["quantity"])
                        venue = venues.venue_for(sym)
                        if notional > caps.get(venue, 0):
                            ev.update(venue=venue, order_message=f"не отправлено: сумма {notional:,.0f} больше лимита тестового прогона {caps[venue]:,.0f}")
                        else:
                            s = send_verdict(venues, v)
                            ev.update(sent=bool(s.get("sent")), venue=s.get("venue"), order_status=s.get("status"),
                                      order_id=s.get("order_id"), order_message=s.get("message"))
                            sent_total += 1 if s.get("sent") else 0
                elif m.get("action") == "hold":
                    ev["outcome"] = "hold"
                elif res.get("model") is None and not v:
                    ev["outcome"] = "no_context" if "Контекст" in res.get("note", "") else "model_error"
                else:
                    ev["outcome"] = "rejected_format"
                emit(ev)
                say(f"[{r + 1}] {agent:<10} {sym:<8} {ev['outcome']:<10} {ev.get('order_status') or ev.get('order_message') or ev.get('note') or ''}"[:150])
            if r < rounds - 1 and pause > 0:
                time.sleep(pause)
    except (StopIteration, KeyboardInterrupt):
        say("Прогон остановлен, формирую отчёт.")
    finally:
        log.close()
    after = snapshot(venues) if send else {}
    return summarize(events, before, after, started, send, fake)


def summarize(events: list, before: dict, after: dict, started: float, send: bool, fake: bool) -> dict:
    outcomes = Counter(e.get("outcome", "?") for e in events)
    rules = Counter(c for e in events if e.get("outcome") in ("deny", "revise") for c in e.get("codes", []))
    per_agent: dict = {}
    for e in events:
        d = per_agent.setdefault(e["agent"], Counter())
        d[e.get("outcome", "?")] += 1
        d["sent"] += 1 if e.get("sent") else 0
    lat = [e["latency_ms"] for e in events if e.get("latency_ms")]
    toks = sum((e.get("tokens") or {}).get("prompt") or 0 for e in events), sum((e.get("tokens") or {}).get("completion") or 0 for e in events)
    return {"fake": fake, "send": send, "started": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - started, 1), "decisions": len(events), "outcomes": dict(outcomes),
            "orders_sent": sum(1 for e in events if e.get("sent")), "top_rules": rules.most_common(6),
            "per_agent": {k: dict(v) for k, v in per_agent.items()},
            "avg_latency_ms": round(sum(lat) / len(lat)) if lat else None, "tokens_prompt": toks[0], "tokens_completion": toks[1],
            "balances_before": before, "balances_after": after}


def write_report(summary: dict, events: list, path: Path) -> None:
    L = ["# Тестовый прогон Finvex", ""]
    if summary["fake"]:
        L += ["> СИНТЕТИЧЕСКИЙ ПРОГОН: данные, модель и площадки подставные, это проверка механики, а не результат.", ""]
    L += [f"- Начало: {summary['started']}, длительность {summary['duration_s']} с",
          f"- Режим: {'заявки в тестовые контуры' if summary['send'] else 'без отправки заявок (пробный)'}",
          f"- Решений агентов: {summary['decisions']}, заявок принято тестовыми контурами: {summary['orders_sent']}",
          f"- Исходы: {summary['outcomes']}", f"- Что чаще всего останавливал шлюз: {summary['top_rules']}",
          f"- Средняя задержка модели: {summary['avg_latency_ms']} мс; токены: запрос {summary['tokens_prompt']}, ответ {summary['tokens_completion']}", "",
          "## Сделки в тестовых контурах", ""]
    L += [f"- {e['ts']} {e['agent']} {e['symbol']} {e.get('action')} {e.get('quantity')} -> {e.get('venue')} {e.get('order_status')} {e.get('order_id')}"
          for e in events if e.get("sent")] or ["- нет"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", default=DEFAULT_AGENTS)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--pause", type=float, default=20.0, help="пауза между раундами, сек.")
    ap.add_argument("--send", action="store_true", help="отправлять разрешённые заявки в тестовые контуры")
    ap.add_argument("--max-orders", type=int, default=10)
    ap.add_argument("--cap-bybit", type=float, default=CAPS["BYBIT"], help="лимит суммы одной заявки, USDT")
    ap.add_argument("--cap-moex", type=float, default=CAPS["MOEX"], help="лимит суммы одной заявки, руб.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fake", action="store_true", help="самопроверка на синтетике, без сети и ключей")
    a = ap.parse_args()
    agents, symbols = [x.strip() for x in a.agents.split(",") if x.strip()], [x.strip().upper() for x in a.symbols.split(",") if x.strip()]
    out = Path(a.out) if a.out else ROOT / "data" / "testruns" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    caps = {"BYBIT": a.cap_bybit, "MOEX": a.cap_moex}
    from finvex.gateway.factory import build
    from finvex.web.settings import LlmSettings
    if a.fake:
        sys.path.insert(0, str(ROOT / "tests"))
        from fake_stack import make_fake
        gw, sandbox, venues = make_fake()
        a.pause = 0
    else:
        gw = build(ROOT, demo=False)
        sandbox, venues = sbx.Sandbox(gw, LlmSettings(ROOT / "configs" / "llm_shared.yaml")), TestVenues()
        if not sandbox.settings.has_key():
            sys.exit("Нет ключа DeepSeek: задайте DEEPSEEK_API_KEY или configs/llm_shared.yaml (вкладка «Настройки»).")
        if a.send:
            st = venues.status()
            for s in st:
                print(f"{s['title']}: {'готов' if s['ready'] else s['message']}")
            if not any(s["ready"] for s in st):
                sys.exit("Ни один тестовый контур не готов: заявки отправить некуда.")
    print(f"Тестовый прогон: агенты {agents}, тикеры {symbols}, раундов {a.rounds}, отправка {'ВКЛ (тестовые контуры)' if a.send else 'выкл'}")
    summary = run(gw, sandbox, venues, agents, symbols, a.rounds, a.pause, a.send, a.max_orders, caps, out / "events.jsonl", a.fake)
    events = [json.loads(x) for x in (out / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_report(summary, events, out / "report.md")
    print(f"Готово: {out}\nДальше: python tools/make_presentation.py {out}")


if __name__ == "__main__":
    main()
