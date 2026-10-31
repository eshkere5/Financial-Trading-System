"""Serialized, lazy TSFM inference. No broker calls and no implicit weight downloads."""
from __future__ import annotations
import copy
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from finvex.core.models import MSK
from .adapter import KronosAdapter

DEFAULTS = dict(enabled=True, model_name="NeoQuasar/Kronos-base",
                tokenizer_name="NeoQuasar/Kronos-Tokenizer-base", device="cpu", max_context=512,
                prediction_length=120, sample_count=3, temperature=1.0, top_p=0.9,
                local_files_only=True, allow_mock=False, cache_ttl_seconds=900, cache_max_entries=128)
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def config_path(root=None):
    return Path(root or os.environ.get("FINVEX_ROOT", ".")).resolve() / "configs" / "kronos.yaml"


def load_config(path=None):
    path = Path(path) if path else config_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("kronos.yaml must contain a mapping")
    cfg = {**DEFAULTS, **(raw or {})}
    lock_path = path.with_name("kronos.lock.json")
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for role in ("model", "tokenizer"):
            entry = lock.get(role) or {}
            if entry.get("repo_id") == cfg.get(role + "_name"):
                cfg.setdefault(role + "_revision", entry.get("revision"))
    for name in ("model_path", "tokenizer_path", "cache_dir"):
        if cfg.get(name) and not Path(cfg[name]).is_absolute():
            cfg[name] = str((path.resolve().parent.parent / cfg[name]).resolve())
    return cfg


class KronosService:
    def __init__(self, cfg=None, adapter_factory=KronosAdapter):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self._factory, self._adapter = adapter_factory, None
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._retry_at = 0.0
        self._state = "not_loaded" if self.cfg.get("enabled") else "disabled"
        self._error = None

    def enabled(self):
        return bool(self.cfg.get("enabled", True))

    def status(self):
        return {"enabled": self.enabled(), "state": self._state, "error": self._error,
                "model": self.cfg.get("model_name"), "device": self.cfg.get("device"),
                "source": "bundled", "local_files_only": self.cfg.get("local_files_only", True),
                "cached_forecasts": len(self._cache)}

    def load(self):
        with self._lock:
            if not self.enabled():
                self._state = "disabled"
                return None
            if self._adapter is None and time.monotonic() >= self._retry_at:
                self._state = "loading"
                try:
                    ad = self._factory(self.cfg)
                    if getattr(ad, "is_mock", True):
                        raise RuntimeError(getattr(ad, "load_error", None) or "real model unavailable")
                    self._adapter, self._state, self._error = ad, "ready", None
                except Exception as exc:
                    self._state, self._error = "unavailable", f"{type(exc).__name__}: {exc}"
                    self._retry_at = time.monotonic() + 30
            return self._adapter

    def forecast(self, full, symbol, tf, samples=3):
        if not self.enabled():
            return {"status": "disabled", "error": "TSFM disabled"}
        try:
            if tf not in TF_SECONDS:
                raise ValueError("Unsupported TSFM timeframe")
            horizon = 10 if tf == "1d" else 24
            samples = int(samples)
            if not 1 <= samples <= 64:
                raise ValueError("sample_count must be 1..64")
            cols = ["timestamps", "open", "high", "low", "close", "volume"]
            if "amount" in full.columns:
                cols.append("amount")
            ctx = full[cols].copy()
            ctx["timestamps"] = pd.to_datetime(ctx["timestamps"])
            if ctx["timestamps"].dt.tz is not None:
                ctx["timestamps"] = ctx["timestamps"].dt.tz_convert(MSK).dt.tz_localize(None)
            now = pd.Timestamp(datetime.now(MSK).replace(tzinfo=None))
            step = pd.Timedelta(seconds=TF_SECONDS[tf])
            ctx = ctx.loc[ctx["timestamps"] + step <= now].sort_values("timestamps").tail(400).reset_index(drop=True)
            if len(ctx) < 2:
                raise ValueError("At least two closed candles required")
            if ctx["timestamps"].duplicated().any() or ctx["timestamps"].isna().any():
                raise ValueError("Duplicate or invalid timestamps")
            values = ctx[[c for c in cols if c != "timestamps"]].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite input values")
            digest = hashlib.sha256(pd.util.hash_pandas_object(ctx, index=False).values.tobytes()).hexdigest()
            key = (str(symbol).upper(), tf, samples, digest)
            with self._lock:
                cached = self._cache.get(key)
                if cached and time.monotonic() - cached[0] < float(self.cfg["cache_ttl_seconds"]):
                    result = copy.deepcopy(cached[1])
                    result["cached"] = True
                    return result
                ad = self.load()
                if ad is None:
                    return {"status": "unavailable", "error": self._error or "TSFM disabled"}
                started = time.monotonic()
                future = pd.date_range(start=ctx["timestamps"].iloc[-1] + step, periods=horizon, freq=step)
                st = ad.encode_prices(ctx, symbol, prediction_length=horizon, sample_count=samples,
                                      future_timestamps=future)
                forecasts = {k: np.asarray(getattr(st, "forecast_" + k), dtype=float) for k in ("open", "high", "low", "close")}
                if any(len(v) != horizon or not np.isfinite(v).all() or (v <= 0).any() for v in forecasts.values()):
                    raise ValueError("Invalid forecast shape or values")
                if (forecasts["high"] < np.maximum(forecasts["open"], forecasts["close"])).any() or (forecasts["low"] > np.minimum(forecasts["open"], forecasts["close"])).any():
                    raise ValueError("Forecast violates OHLC bounds")
                last = float(ctx["close"].iloc[-1])
                change, voln = float(forecasts["close"][-1] / last - 1), float(st.volatility_norm)
                if not np.isfinite([change, voln, st.confidence, st.expected_return]).all() or voln < 0:
                    raise ValueError("Invalid forecast metrics")
                end = ctx["timestamps"].iloc[-1]
                result = {"status": "ok", "horizon": horizon, "last": last, "target": float(forecasts["close"][-1]),
                          "change": change, "expected_return": float(st.expected_return), "confidence": float(st.confidence),
                          "direction": int(st.direction), "volatility_norm": voln,
                          "seconds": round(time.monotonic() - started, 3), "cached": False,
                          "score": float(0.7 * np.tanh(change / (2 * max(voln, 1e-4)))),
                          "as_of": end.replace(tzinfo=MSK).isoformat(),
                          "valid_until": (end + 2 * step).replace(tzinfo=MSK).isoformat(),
                          "timeframe": tf, "model": self.cfg.get("model_name"),
                          "calendar": "regular_step_not_exchange_calendar",
                          "note": "confidence is channel tightness, not probability"}
                result.update({"forecast_" + k: v.tolist() for k, v in forecasts.items()})
                self._cache[key] = (time.monotonic(), copy.deepcopy(result))
                self._cache.move_to_end(key)
                while len(self._cache) > max(1, int(self.cfg["cache_max_entries"])):
                    self._cache.popitem(last=False)
                self._state, self._error = "ready", None
                return result
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            return {"status": "error", "error": self._error}


_SERVICES = {}
_SERVICES_LOCK = threading.Lock()


def get_service(root=None):
    path = config_path(root)
    cfg = load_config(path)
    key = (str(path), json.dumps(cfg, sort_keys=True, default=str))
    with _SERVICES_LOCK:
        if key not in _SERVICES:
            _SERVICES[key] = KronosService(cfg)
            while len(_SERVICES) > 4:
                del _SERVICES[next(iter(_SERVICES))]
        return _SERVICES[key]
