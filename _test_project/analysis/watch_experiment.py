"""
watch_experiment.py — просмотр experiment.db во время/после прогона.

### NEW 2026-08-20 ###
Класть в КОРЕНЬ проекта. БД по умолчанию: data/experiment.db
(как DEFAULT_DB_PATH в experiment_logger.py). Переопределить: --db путь

    python watch_experiment.py                 # сводка последнего прогона
    python watch_experiment.py --list          # все run_id в базе
    python watch_experiment.py --tail 20       # последние N событий
    python watch_experiment.py --run D_full    # сводка конкретного прогона
    python watch_experiment.py --export equity.csv
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "experiment.db"


def _con(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def _runs(c) -> list[str]:
    out: list[str] = []
    for table in ("equity_snapshots", "decision_events", "attribution"):
        try:
            rows = c.execute(
                f"SELECT DISTINCT run_id FROM {table} ORDER BY run_id"
            ).fetchall()
            out.extend(r[0] for r in rows)
        except sqlite3.OperationalError:
            pass
    seen, ordered = set(), []
    for r in out:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered


def summary(c, run_id: str) -> None:
    print(f"\n{'='*64}\nСВОДКА ПРОГОНА: {run_id}\n{'='*64}")

    eq = c.execute(
        "SELECT ts, equity, cash, positions, pnl FROM equity_snapshots "
        "WHERE run_id=? ORDER BY ts", (run_id,),
    ).fetchall()
    if eq:
        first, last = eq[0], eq[-1]
        peak = max(r["equity"] for r in eq)
        trough = min(r["equity"] for r in eq)
        mdd, run_peak = 0.0, -1e18
        for r in eq:
            run_peak = max(run_peak, r["equity"])
            if run_peak > 0:
                mdd = max(mdd, (run_peak - r["equity"]) / run_peak)
        hours = (last["ts"] - first["ts"]) / 3600.0
        print(f"\nБЮДЖЕТ  ({hours:.1f} ч, {len(eq)} снапшотов)")
        print(f"  старт equity   : {first['equity']:.2f}")
        print(f"  финал equity   : {last['equity']:.2f}  (PnL {last['pnl']:+.2f})")
        print(f"  пик / дно      : {peak:.2f} / {trough:.2f}")
        print(f"  max drawdown   : {mdd*100:.2f}%")
        print(f"  позиций сейчас : {last['positions']}")
    else:
        print("\nБЮДЖЕТ: снапшотов нет")

    ev = c.execute(
        "SELECT actor, action, COUNT(*) n FROM decision_events "
        "WHERE run_id=? GROUP BY actor, action ORDER BY actor, n DESC", (run_id,),
    ).fetchall()
    if ev:
        print("\nСОБЫТИЯ ПО КОНТУРАМ")
        for r in ev:
            print(f"  {r['actor']:<12} {r['action']:<18} x{r['n']}")

    att = c.execute(
        "SELECT ticker, COUNT(*) n, AVG(kronos_conf) kc, AVG(news_risk) nr, "
        "AVG(strat_conv) sc, AVG(size_mult) sm, AVG(final_qty) q FROM attribution "
        "WHERE run_id=? GROUP BY ticker", (run_id,),
    ).fetchall()
    if att:
        print("\nАТРИБУЦИЯ (среднее по решениям)")
        print(f"  {'тикер':<10} {'n':>4} {'kron_conf':>9} {'news_risk':>9} "
              f"{'strat_conv':>10} {'size_mult':>9} {'avg_qty':>9}")
        for r in att:
            print(f"  {r['ticker']:<10} {r['n']:>4} {r['kc'] or 0:>9.2f} "
                  f"{r['nr'] or 0:>9.2f} {r['sc'] or 0:>10.2f} "
                  f"{r['sm'] or 0:>9.2f} {r['q'] or 0:>9.4f}")
    else:
        print("\nАТРИБУЦИЯ: записей нет (сигналов ещё не было)")
    print()


def tail(c, run_id: str | None, n: int) -> None:
    q = ("SELECT ts, actor, ticker, action, rationale FROM decision_events "
         + ("WHERE run_id=? " if run_id else "")
         + "ORDER BY ts DESC LIMIT ?")
    rows = c.execute(q, (run_id, n) if run_id else (n,)).fetchall()
    print(f"\nПОСЛЕДНИЕ {len(rows)} СОБЫТИЙ" + (f" (run={run_id})" if run_id else ""))
    for r in rows:
        t = datetime.fromtimestamp(r["ts"], tz=timezone.utc).strftime("%m-%d %H:%M:%S")
        rat = (r["rationale"] or "")[:70]
        print(f"  {t}  {r['actor']:<11} {r['ticker']:<8} {r['action']:<14} {rat}")
    print()


def export(c, run_id: str, out: str) -> None:
    rows = c.execute(
        "SELECT ts, equity, cash, positions, pnl FROM equity_snapshots "
        "WHERE run_id=? ORDER BY ts", (run_id,),
    ).fetchall()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "equity", "cash", "positions", "pnl"])
        for r in rows:
            w.writerow([r["ts"], r["equity"], r["cash"], r["positions"], r["pnl"]])
    print(f"Экспортировано {len(rows)} точек → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="путь к experiment.db (по умолчанию data/experiment.db)")
    ap.add_argument("--run", default=None, help="run_id (по умолчанию — последний)")
    ap.add_argument("--tail", type=int, default=0, help="показать N последних событий")
    ap.add_argument("--export", default=None, help="экспорт equity в CSV")
    ap.add_argument("--list", action="store_true", help="список всех run_id")
    args = ap.parse_args()

    db = Path(args.db) if args.db else DEFAULT_DB
    if not db.exists():
        # запасной вариант — вдруг логгер создал в другом месте
        for cand in (ROOT / "src" / "db" / "experiment.db", ROOT / "experiment.db"):
            if cand.exists():
                db = cand
                break
        else:
            sys.exit(f"experiment.db не найден. Искал: {DEFAULT_DB}, src/db/, корень.\n"
                     f"Сначала запусти run_experiment.py.")

    c = _con(db)
    runs = _runs(c)
    if args.list:
        print(f"БД: {db}\nПрогоны в базе:")
        for r in runs:
            print(" ", r)
        if not runs:
            print("  (пусто)")
        return
    if not runs:
        sys.exit("База пуста — ни одного прогона. Запусти run_experiment.py.")

    run_id = args.run or runs[-1]
    if run_id not in runs:
        print(f"run_id '{run_id}' не найден. Есть: {', '.join(runs)}")
        return

    if args.export:
        export(c, run_id, args.export)
    elif args.tail:
        tail(c, run_id if args.run else None, args.tail)
    else:
        summary(c, run_id)


if __name__ == "__main__":
    main()
