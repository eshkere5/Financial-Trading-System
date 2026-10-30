"""Журнал решений: JSONL с цепочкой хэшей. Правка задним числом ломает проверку."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import threading
from pathlib import Path
from typing import Iterable

GENESIS = "0" * 64


def _canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Ledger:
    def __init__(self, path: "str | Path | None") -> None:
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self._rows: list = []
        self._last = GENESIS
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._rows.append(row)
                    self._last = row["hash"]
        elif self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, data: dict) -> str:
        with self._lock:
            body = {"seq": len(self._rows) + 1, "prev": self._last, "data": data}
            digest = hashlib.sha256((self._last + _canon(body)).encode("utf-8")).hexdigest()
            row = {**body, "hash": digest}
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(_canon(row) + "\n")
            self._rows.append(row)
            self._last = digest
            return digest

    def verify(self) -> dict:
        prev = GENESIS
        for row in self._rows:
            body = {"seq": row["seq"], "prev": row["prev"], "data": row["data"]}
            ok = row["prev"] == prev and row["hash"] == hashlib.sha256((prev + _canon(body)).encode("utf-8")).hexdigest()
            if not ok:
                return {"ok": False, "checked": row["seq"], "broken_at": row["seq"]}
            prev = row["hash"]
        return {"ok": True, "checked": len(self._rows), "head": prev}

    def rows(self) -> list:
        with self._lock:
            return list(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    @staticmethod
    def to_csv(verdicts: Iterable[dict]) -> str:
        buf = io.StringIO()
        wr = csv.writer(buf, delimiter=";")
        wr.writerow(["ts", "agent", "mode", "symbol", "side", "qty", "decision", "executed", "reasons", "codes", "mandate", "verdict_id", "hash"])
        for v in verdicts:
            i = v["intent"]
            wr.writerow([v["ts"], v["agent_name"], v["mode"], i["symbol"], i["side"], i["quantity"], v["decision"],
                         int(v["executed"]), " | ".join(v["reasons"]), ",".join(v.get("codes") or []),
                         (v.get("mandate_hash") or "")[:10], v["id"], v.get("ledger_hash", "")])
        return buf.getvalue()
