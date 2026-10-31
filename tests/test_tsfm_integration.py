"""Offline contracts; inference is stubbed explicitly, never claimed to be real."""
import ast
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finvex.analysis.kronos.adapter import KronosAdapter, KronosState
from finvex.analysis.kronos.service import KronosService, load_config
from finvex.core.models import MSK
from finvex.feeds.analytics_feeder import AnalyticsFeeder, start_background
from finvex.gateway.analytics import AnalyticsStore


def frame(n=40):
    end = pd.Timestamp(datetime.now(MSK).replace(tzinfo=None)).floor("h") - pd.Timedelta(hours=1)
    x = np.arange(n, dtype=float) / 100 + 100
    return pd.DataFrame({"timestamps": pd.date_range(end=end, periods=n, freq="h"),
                         "open": x, "high": x + 1, "low": x - 1, "close": x, "volume": np.ones(n)})


class FakeAdapter:
    is_mock = False

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.calls = []

    def encode_prices(self, df, symbol, *, prediction_length, sample_count, future_timestamps=None):
        self.calls.append((prediction_length, sample_count, len(df)))
        time.sleep(0.01)
        close = np.linspace(100, 101, prediction_length)
        return KronosState(symbol, close, close + 1, close - 1, close)


def test_single_load_and_cached_forecast_under_parallel_calls():
    created = []

    def factory(cfg):
        ad = FakeAdapter(cfg)
        created.append(ad)
        return ad

    svc = KronosService(adapter_factory=factory)
    data = frame()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: svc.forecast(data, "SBER", "1h", 1), range(12)))
    assert all(r["status"] == "ok" for r in results)
    assert len(created) == 1 and len(created[0].calls) == 1
    results[0]["forecast_close"][0] = 0
    assert svc.forecast(data, "SBER", "1h", 1)["forecast_close"][0] > 0


def test_per_call_options_do_not_mutate_shared_config():
    svc = KronosService(adapter_factory=FakeAdapter)
    assert svc.forecast(frame(), "A", "1h", 1)["horizon"] == 24
    assert svc.forecast(frame(), "A", "1d", 2)["horizon"] == 10
    assert [c[:2] for c in svc._adapter.calls] == [(24, 1), (10, 2)]


def test_open_candles_are_not_passed_to_model():
    svc = KronosService(adapter_factory=FakeAdapter)
    data = frame()
    open_bar = data.iloc[-1:].copy()
    open_bar["timestamps"] = datetime.now(MSK).replace(tzinfo=None)
    data = pd.concat([data, open_bar], ignore_index=True)
    assert svc.forecast(data, "SBER", "1h", 1)["status"] == "ok"
    assert svc._adapter.calls[0][2] == 40


@pytest.mark.parametrize("case", ["nan", "duplicate", "few", "timeframe", "samples"])
def test_invalid_input_is_not_inferred(case):
    svc = KronosService(adapter_factory=FakeAdapter)
    data, tf, samples = frame(), "1h", 1
    if case == "nan":
        data.loc[2, "close"] = np.nan
    if case == "duplicate":
        data.loc[2, "timestamps"] = data.loc[1, "timestamps"]
    if case == "few":
        data = data.tail(1)
    if case == "timeframe":
        tf = "bad"
    if case == "samples":
        samples = 0
    assert svc.forecast(data, "SBER", tf, samples)["status"] == "error"
    assert svc._adapter is None


def test_disabled_and_unavailable_never_become_fake_forecasts():
    def fail(cfg):
        raise RuntimeError("weights missing")

    off = KronosService({"enabled": False}, adapter_factory=fail)
    assert off.forecast(frame(), "S", "1h")["status"] == "disabled"
    bad = KronosService(adapter_factory=fail)
    assert bad.forecast(frame(), "S", "1h")["status"] == "unavailable"
    assert bad.status()["state"] == "unavailable"


def test_adapter_refuses_mock_by_default_without_torch():
    ad = KronosAdapter.__new__(KronosAdapter)
    ad.cfg, ad.predictor, ad.load_error = {}, None, "missing"
    with pytest.raises(RuntimeError, match="Real Kronos unavailable"):
        ad.encode_prices(frame(), "S")


def test_trading_strategies_skip_mock_before_encoding():
    from finvex.decision.ai_tactic import AITacticStrategy
    from finvex.decision.kronos_default import KronosDefaultStrategy
    for cls in (AITacticStrategy, KronosDefaultStrategy):
        obj = cls.__new__(cls)
        obj._kronos = SimpleNamespace(is_mock=True)
        assert obj._get_kronos_signal("SBER", SimpleNamespace()) is None


def test_feeder_does_not_redate_historical_forecasts():
    store = AnalyticsStore(None)
    fd = AnalyticsFeeder(store)
    past = (datetime.now(MSK) - timedelta(days=1)).isoformat()
    assert fd.push_kronos("SBER", {"status": "ok", "change": .02, "volatility_norm": .01, "valid_until": past}) is None
    assert store.all() == {}
    future = (datetime.now(MSK) + timedelta(hours=1)).isoformat()
    fd.push_kronos("SBER", {"status": "ok", "change": .02, "volatility_norm": .01, "valid_until": future})
    assert "kronos_score" in store.snapshot("SBER")


def test_session_direct_tool_respects_switch(tmp_path):
    from finvex.market_agent import Session
    calls = []
    sess = Session(tmp_path, use_kronos=False, kronos_call=lambda *a: calls.append(a) or {"status": "ok"})
    assert sess.forecast(frame(), "SBER")["status"] == "disabled"
    assert not calls
    sess.use_kronos = True
    assert sess.forecast(frame(), "SBER")["status"] == "ok"
    assert len(calls) == 1


def test_background_pauses_and_can_stop():
    active, calls = [False], []
    stop = start_background(AnalyticsStore(None), ["SBER"], interval_s=.02,
                            kronos_source=lambda *a: calls.append(a) or {"status": "error"},
                            news_source=lambda _: [], enabled=lambda: active[0], say=lambda _: None)
    time.sleep(.06)
    assert not calls
    active[0] = True
    time.sleep(.08)
    stop.set()
    stop.thread.join(timeout=1)
    assert calls and not stop.thread.is_alive()


def test_lock_revisions_and_changed_model(tmp_path):
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "kronos.yaml"
    path.write_text("model_name: NeoQuasar/Kronos-base\n", encoding="utf-8")
    path.with_name("kronos.lock.json").write_text(json.dumps({"model": {"repo_id": "NeoQuasar/Kronos-base", "revision": "a" * 40}}))
    assert load_config(path)["model_revision"] == "a" * 40
    path.write_text("model_name: other/model\n", encoding="utf-8")
    assert not load_config(path).get("model_revision")


def test_bundled_code_is_namespaced_and_licensed():
    base = Path(__file__).resolve().parents[1] / "finvex/analysis/kronos/_vendor"
    assert (base / "LICENSE").is_file()
    metadata = json.loads((base / "UPSTREAM.json").read_text())
    assert metadata["upstream_commit"] is None
    assert len(metadata["original_sha256"]) == 3
    for path in (base / "model").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            assert not (isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith("model"))


def test_web_chat_publishes_forecast_and_switch_blocks_direct_tool(tmp_path, monkeypatch):
    import http.client
    from finvex import market_agent
    from finvex.analysis.kronos import service
    from finvex.gateway.factory import build
    from finvex.web.server import make_server
    from finvex.web.settings import LlmSettings

    svc = KronosService(adapter_factory=FakeAdapter)
    monkeypatch.setattr(service, "get_service", lambda root=None: svc)
    monkeypatch.setattr(LlmSettings, "has_key", lambda self: True)
    monkeypatch.setattr(market_agent, "Sandbox", lambda sess: None)

    class Agent:
        def __init__(self, sess, verbose=False):
            self.sess = sess

        def ask(self, text):
            return json.dumps(self.sess.forecast(frame(), "SBER", "1h", 1))

    monkeypatch.setattr(market_agent, "Agent", Agent)
    gw = build(tmp_path, demo=True)
    gw.demo = False
    gw.clock = lambda: datetime.now(MSK)
    gw.analytics = AnalyticsStore(None)
    srv = make_server(gw, "127.0.0.1", 0, tmp_path / "out", tmp_path)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    port = srv.server_address[1]

    def req(path, method="GET", data=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        try:
            c.request(method, path, json.dumps(data).encode() if data else None, headers or {})
            r = c.getresponse()
            return r.status, r.read().decode()
        finally:
            c.close()

    try:
        page = req("/")[1]
        token = {"X-Token": page.split('name="finvex-token" content="')[1].split('"')[0]}
        assert req("/api/tsfm")[0] == 401
        status, raw = req("/api/chat", "POST", {"sid": "test", "text": "forecast"}, token)
        assert status == 200
        assert json.loads(json.loads(raw)["answer"])["status"] == "ok"
        assert "kronos_score" in gw.analytics.snapshot("SBER")
        state = json.loads(req("/api/state", headers=token)[1])
        assert state["tsfm"]["state"] == "ready" and state["tsfm"]["active"]
        gw.modules.set_enabled("analytics", False)
        assert req("/api/tsfm", headers=token)[0] == 409
        raw = req("/api/chat", "POST", {"sid": "test", "text": "forecast"}, token)[1]
        assert json.loads(json.loads(raw)["answer"])["status"] == "disabled"
        assert len(svc._adapter.calls) == 1
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(2)


def test_adapter_uses_namespaced_source_and_loader_flags(monkeypatch):
    import sys
    import types
    fake_model = types.ModuleType("finvex.analysis.kronos._vendor.model")
    recorded = []

    class Weights:
        @classmethod
        def from_pretrained(cls, name, **kw):
            recorded.append((name, kw))
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    class Predictor:
        def __init__(self, **kw):
            pass

    fake_model.Kronos = fake_model.KronosTokenizer = Weights
    fake_model.KronosPredictor = Predictor
    monkeypatch.setitem(sys.modules, "finvex.analysis.kronos._vendor.model", fake_model)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    ad = KronosAdapter({"model_revision": "a" * 40, "tokenizer_revision": "b" * 40})
    assert not ad.is_mock
    assert recorded[0][1] == {"local_files_only": True, "revision": "b" * 40}
    assert recorded[1][1] == {"local_files_only": True, "revision": "a" * 40}


def test_prepare_records_resolved_revisions_without_network(tmp_path, monkeypatch):
    import huggingface_hub
    from finvex.tsfm import prepare
    calls = []

    def download(**kw):
        calls.append(kw)
        return str(tmp_path / "snapshots" / ("a" * 40))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "kronos.yaml"
    result = prepare(path, load_config(path))
    assert result["status"] == "prepared"
    lock = json.loads(path.with_name("kronos.lock.json").read_text())
    assert lock["model"]["revision"] == "a" * 40
    assert len(calls) == 2 and all(not c["local_files_only"] for c in calls)
    assert load_config(path)["model_revision"] == "a" * 40


def test_adapter_predict_options_and_real_shape():
    class Predictor:
        def __init__(self):
            self.options = None

        def predict(self, **kwargs):
            self.options = kwargs
            n = kwargs["pred_len"]
            return pd.DataFrame({"open": np.ones(n) * 100, "high": np.ones(n) * 101,
                                 "low": np.ones(n) * 99, "close": np.ones(n) * 100})

    ad = KronosAdapter.__new__(KronosAdapter)
    ad.cfg, ad.predictor, ad.is_mock = {"sample_count": 8}, Predictor(), False
    ad._predict_lock = threading.RLock()
    state = ad.encode_prices(frame(), "SBER", prediction_length=3, sample_count=1)
    assert len(state.forecast_close) == 3
    assert ad.predictor.options["sample_count"] == 1
    assert ad.cfg == {"sample_count": 8}
    assert ad.predictor.options["x_timestamp"].iloc[-1] == frame()["timestamps"].iloc[-1]


def test_pack_excludes_private_state_and_keeps_bundled_source(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("finvex_source_pack", Path(__file__).resolve().parents[1] / "tools/pack.py")
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    pack.ROOT = tmp_path
    for rel in ("configs/a.user.yaml", "data/gateway/agents.json", "archive/old.py", ".env.dev", "Kronos/model/kronos.py",
                "finvex/analysis/kronos/_vendor/model/kronos.py", "finvex/data/collector.py"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
        assert pack.include(path) == rel.startswith("finvex/")
