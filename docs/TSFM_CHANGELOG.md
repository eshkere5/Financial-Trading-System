# TSFM / Kronos change log (0.2.1)

Changed: bundled namespaced Kronos inference source; shared serialized `KronosService`; web chat and
direct tool use one switch; fresh web forecasts feed gateway analytics; optional explicit background
feeder; mock refusal in both trading strategies; explicit `finvex.tsfm prepare|doctor|smoke-test`;
Windows setup script; pack tool excludes private state/weights; `configs/providers.yaml` key typo
`param` -> `params` fixed; documentation and offline contract tests added.

## Executed validation (Linux sandbox)

- Contract tests: `tests/test_tsfm_integration.py`, `test_analytics_feeder.py`, `test_architecture.py`: 36 passed.
- Offline regression (excluding real-network tests and the pybit-dependent file): 194 passed, 1 skipped, 2 pre-existing failures.
- Two pre-existing red-team expectation failures (`revise` instead of `deny`) reproduce on the unmodified archive; policy was not changed.
- Not executed here: PyTorch, real weights, Windows installation, real market APIs, LLM API, brokers.
