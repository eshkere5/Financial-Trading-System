# TSFM / Kronos: bundled inference source and first run

## What is included

Finvex 0.2.1 contains the three inference source files from the Kronos snapshot supplied in the
project archive: `finvex/analysis/kronos/_vendor/model/`. A second clone, submodule, generic
`model` import, or `sys.path` change is not needed. Training code, upstream examples and weights
are not bundled. The MIT license is preserved; `UPSTREAM.json` stores SHA256 hashes of the original
files. The upstream commit was not in the archive and is recorded as unknown. The only source change
is `from model.module` -> `from .module`.

## Installation on Windows (from the project root)

Keep your working `.venv`; do not recreate it unnecessarily.

```powershell
python -m pip install -e ".[dev,tsfm,crypto]"
python -m finvex.tsfm prepare
python -m finvex.tsfm doctor --load
python -m finvex.tsfm smoke-test
python -m finvex.web --open
```

`& .\scripts\setup-tsfm.ps1` performs install + prepare + load check (without crypto).
`-SkipInstall` keeps the current environment; `-Offline -SkipInstall` forbids downloads.

`prepare` is the explicit network step: Hugging Face model and tokenizer files go to its standard
cache, not Git. It writes `configs/kronos.lock.json` with resolved revisions; commit that small file
for reproducible model selection. Before it exists, weights are not revision-pinned. Python
dependencies are constrained, not fully locked.

Runtime uses `local_files_only: true`; opening the app does not start an unexpected download.
`doctor --load` loads real weights. `smoke-test` uses the real model and a synthetic local OHLCV
fixture (no exchange, LLM or broker). Failure returns nonzero; mock output is never success.
The shipped config keeps Kronos-base on CPU; change `configs/kronos.yaml`, prepare again, restart.
For offline deployment copy a complete dependency environment and Hugging Face cache, or set
`model_path` / `tokenizer_path` to local weight folders. A source ZIP alone is not an offline bundle.

## Web, chat and gateway

The chat `analyze` recipe and the direct `kronos` tool share one session switch. Successful fresh
forecasts made through web chat are written to gateway analytics. The model loads lazily on first
use. Authenticated `GET /api/tsfm` and `GET /api/state` report its status; the top bar shows TSFM
state after a page refresh. Chat itself still needs the separately configured LLM API key, and the LLM
decides whether to call a forecast tool.

Independent periodic forecasts must be requested explicitly:

```powershell
python -m finvex.web --open --tsfm-symbols SBER,GAZP --tsfm-timeframe 1h --tsfm-interval 900
```

This uses market APIs and TSFM only (no paid news LLM calls). Plain web startup does not scan a
universe. Disabling the analytics module blocks chat forecast tools and pauses background writes.
`--demo` disables web TSFM calls. Standalone `market_agent` is a separate CLI and ignores web module state.
Background shutdown waits two seconds; an already running PyTorch inference cannot be forcibly cancelled.
The existing LLM Python sandbox is not a security boundary.

## Safety and semantics

- Trading strategies skip a missing/mock adapter instead of turning synthetic output into a signal.
- Analysis inference is serialized per process/config and cached by input data, symbol, timeframe and samples.
- Only completed regular-interval bars are used; duplicate/nonfinite input and invalid forecasts are rejected.
- Forecast validity follows the input bar, so old reports are not inserted as fresh.
- `confidence` is channel tightness, not a calibrated probability.
- Future timestamps use regular steps, not an official MOEX session/holiday calendar.
- Execution settings were not enabled and no orders were used for validation.

## Checks and safe update

```powershell
python -m compileall -q finvex
python -m pytest tests/test_tsfm_integration.py tests/test_analytics_feeder.py tests/test_architecture.py -q
```

These contract tests use explicit inference test doubles. They do not prove real-model accuracy or
full Windows compatibility. See `TSFM_CHANGELOG.md` for the executed regression results.

Extract the delivered archive into a temporary folder and review it before copying public files over
your checkout. Do not delete `.git`, `.venv`, private configs or local data. Root `Kronos/` can stay
locally ignored; it is no longer needed or committed.

```powershell
git status --short
git add -A -- .gitignore pyproject.toml README.md configs/kronos.yaml configs/providers.yaml finvex scripts docs tests tools .github
git diff --cached --name-only
git diff --cached --check
```

Inspect the staged list and make sure no secret/private config is included. If `configs/kronos.lock.json`
is created, stage it separately.
