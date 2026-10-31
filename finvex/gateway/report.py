"""Отчёт «что остановил шлюз»: сводка по журналу вердиктов."""
from __future__ import annotations

from collections import Counter

from .rules import RULES_META

TITLES = {m["id"]: m["title"] for m in RULES_META}
TITLES.update({"quota": "Суточная квота агента", "context": "Контекст рынка"})


def build_report(verdicts: "list[dict]") -> dict:
    total = len(verdicts)
    by = Counter(v["decision"] for v in verdicts)
    held = [v for v in verdicts if v["decision"] != "allow"]
    by_rule: Counter = Counter()
    for v in held:
        by_rule.update(v.get("codes") or [])
    notional = sum(float((v.get("context") or {}).get("price") or 0) * float(v["intent"]["quantity"]) for v in held)
    return {"total": total, "allow": by.get("allow", 0), "revise": by.get("revise", 0), "deny": by.get("deny", 0),
            "stopped_pct": round(100 * len(held) / total, 1) if total else 0.0, "held_notional": round(notional, 2),
            "top_rules": [{"rule_id": r, "title": TITLES.get(r, r), "count": n} for r, n in by_rule.most_common(8)]}
