"""CLI шлюза: demo, agent-add, agent-list, verify."""
from __future__ import annotations

import argparse
import json

from .factory import build


def main() -> None:
    ap = argparse.ArgumentParser(prog="finvex.gateway")
    ap.add_argument("cmd", choices=["demo", "agent-add", "agent-list", "verify"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--root", default=".")
    ap.add_argument("--mode", default="shadow")
    a = ap.parse_args()
    if a.cmd == "demo":
        gw = build(a.root, demo=True)
        for v in reversed(gw.list_verdicts(100)):
            i = v["intent"]
            print(f"{v['ts'][11:16]} {v['agent_name']:<15} {i['side']:<4} {i['symbol']:<8} {i['quantity']:<8g} -> {v['decision']:<6} {'; '.join(v['reasons'])[:70]}")
        print(json.dumps(gw.stats(), ensure_ascii=False))
        return
    gw = build(a.root)
    if a.cmd == "agent-add":
        ag, key = gw.agents.create(a.name or "agent", a.mode)
        print(f"Агент {ag.id} создан. Ключ показывается один раз:\n{key}")
    elif a.cmd == "agent-list":
        for ag in gw.agents.list():
            print(ag.id, ag.name, ag.mode, "отозван" if ag.revoked else "активен")
    else:
        res = gw.ledger.verify()
        print(json.dumps(res, ensure_ascii=False))
        raise SystemExit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
