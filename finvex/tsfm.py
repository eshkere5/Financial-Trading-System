"""Explicit TSFM preparation and diagnostics; never contacts brokers or LLM APIs."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from finvex.analysis.kronos.service import KronosService, config_path, load_config


def prepare(path, cfg, offline=False):
    from huggingface_hub import snapshot_download
    lock = {"schema": 1, "source": "bundled inference source; see _vendor/UPSTREAM.json"}
    for role in ("model", "tokenizer"):
        if cfg.get(role + "_path"):
            folder = Path(cfg[role + "_path"])
            if not folder.is_dir():
                raise ValueError(f"Local {role} directory not found")
            lock[role] = {"local_path": str(folder.resolve())}
            continue
        repo = cfg[role + "_name"]
        folder = Path(snapshot_download(repo_id=repo, revision=cfg.get(role + "_revision"),
                        cache_dir=cfg.get("cache_dir"), local_files_only=offline,
                        allow_patterns=["*.json", "*.safetensors", "*.bin", "README.md", "LICENSE*"]))
        lock[role] = {"repo_id": repo, "revision": folder.name}
    dest = path.with_name("kronos.lock.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    temp.replace(dest)
    return {"status": "prepared", "lock_file": str(dest), "weights": "Hugging Face cache; not included in Git"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["prepare", "doctor", "smoke-test"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--load", action="store_true", help="doctor: load cached weights")
    ap.add_argument("--offline", action="store_true", help="prepare: use cached files only")
    args = ap.parse_args(argv)
    path = config_path(args.root)
    try:
        cfg = load_config(path)
        if args.cmd == "prepare":
            result = prepare(path, cfg, args.offline)
        elif args.cmd == "doctor":
            deps = {m: importlib.util.find_spec(m) is not None for m in
                    ("torch", "einops", "huggingface_hub", "safetensors", "tqdm")}
            svc = KronosService(cfg)
            if args.load:
                svc.load()
            ok = all(deps.values()) and (not args.load or svc.status()["state"] == "ready")
            result = {"status": "ok" if ok else "error", "dependencies": deps,
                      "config": str(path), "runtime": svc.status(),
                      "hint": "pip install -e .[tsfm]; python -m finvex.tsfm prepare"}
        else:
            import numpy as np
            import pandas as pd
            from finvex.core.models import MSK
            from datetime import datetime
            end = pd.Timestamp(datetime.now(MSK).replace(tzinfo=None)).floor("D") - pd.Timedelta(days=2)
            close = 100 + np.linspace(0, 3, 64) + np.sin(np.arange(64) / 4)
            frame = pd.DataFrame({"timestamps": pd.date_range(end=end, periods=64, freq="D"),
                                  "open": close, "high": close + 1, "low": close - 1,
                                  "close": close, "volume": np.full(64, 1000.0)})
            result = KronosService(cfg).forecast(frame, "SMOKE_FIXTURE", "1d", 1)
            result = {k: v for k, v in result.items() if not k.startswith("forecast_")}
            result["input"] = "synthetic fixture, real model; no market/LLM/broker request"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in ("ok", "prepared") else 1
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
