import json
import subprocess
import sys
import threading
import types
import urllib.request
from datetime import datetime
from pathlib import Path

import pytest

from finvex.core.models import MSK
from finvex.gateway.context import PaperPortfolio
from finvex.gateway.factory import DemoClock, build
from finvex.gateway.gateway import Gateway, GatewayError
from finvex.gateway.ledger import Ledger
from finvex.gateway.prices import StaticPriceSource

T = datetime(2026, 10, 21, 12, 0, tzinfo=MSK)


def mk(tmp=None, executor=None, enabled=False, pf=None):
    clock = DemoClock(T)
    gw = Gateway(tmp, StaticPriceSource(), pf or PaperPortfolio(1_000_000, {"SBER": 300.0}), clock=clock,
                 executor=executor, execution_enabled=enabled)
    return gw, clock


def intent(**kw):
    base = dict(symbol="SBER", side="buy", quantity=10, rationale="Пробой SMA50 на росте объёма, стоп под минимумом", confidence=0.7)
    base.update(kw)
    return base


def test_allow_and_rationale_revise():
    gw, _ = mk()
    ag, _ = gw.agents.create("a")
    assert gw.propose(ag, intent())["decision"] == "allow"
    v = gw.propose(ag, intent(symbol="GAZP", rationale=""))
    assert v["decision"] == "allow" and "Замечания" in v["agent_message"]   # обоснование не блокирует, если остальное верно


def test_revise_gives_safe_quantity():
    gw, _ = mk()
    ag, _ = gw.agents.create("a")
    v = gw.propose(ag, intent(quantity=5000))
    assert v["decision"] in ("revise", "deny")
    if v["decision"] == "revise":
        assert 0 < v["suggested_quantity"] < 5000


def test_short_sell_and_session_and_typo():
    gw, clock = mk()
    ag, _ = gw.agents.create("a")
    assert gw.propose(ag, intent(side="sell", quantity=500))["decision"] == "revise"
    v = gw.propose(ag, intent(symbol="GAZP", order_type="limit", limit_price=13.0))
    assert v["decision"] == "revise" and any("опечатк" in r for r in v["reasons"])
    clock.t = clock.t.replace(hour=22)
    v2 = gw.propose(ag, intent(symbol="MTSS"))
    assert v2["decision"] == "revise" and v2["retry_after"] > 0 and "повторите через" in v2["agent_message"]   # ошибки нет, нужно подождать
    assert gw.propose(ag, intent(symbol="BTCUSDT", quantity=0.05))["decision"] == "allow"


def test_unknown_symbol_fails_closed():
    gw, _ = mk()
    ag, _ = gw.agents.create("a")
    v = gw.propose(ag, intent(symbol="ZZZZ"))
    assert v["decision"] == "deny" and "Нет проверенных данных" in v["reasons"][0]


def test_idempotent_client_id():
    gw, _ = mk()
    ag, _ = gw.agents.create("a")
    a = gw.propose(ag, intent(client_id="x1"))
    b = gw.propose(ag, intent(client_id="x1"))
    assert a["id"] == b["id"] and b["duplicate"] and len(gw.list_verdicts()) == 1


def test_bad_input_and_auth():
    gw, _ = mk()
    ag, key = gw.agents.create("a")
    for bad in (intent(quantity=-1), intent(side="hold"), intent(symbol="a b"), intent(order_type="limit")):
        with pytest.raises(GatewayError):
            gw.propose(ag, bad)
    assert gw.authenticate(key).id == ag.id
    with pytest.raises(GatewayError):
        gw.authenticate("bad")
    gw.agents.update(ag.id, revoked=True)
    with pytest.raises(GatewayError):
        gw.authenticate(key)


def test_cooldown_only_after_allowed():
    gw, _ = mk()
    ag, _ = gw.agents.create("a")
    assert gw.propose(ag, intent(client_id="1"))["decision"] == "allow"
    v = gw.propose(ag, intent(client_id="2"))
    assert v["decision"] == "revise" and 0 < v["retry_after"] <= 300 and "cooldown" in v["codes"]


def test_ledger_persist_and_tamper(tmp_path):
    gw, _ = mk(tmp_path)
    ag, _ = gw.agents.create("a")
    gw.propose(ag, intent(client_id="1"))
    gw.propose(ag, intent(symbol="MTSS", client_id="2"))
    assert gw.ledger.verify()["ok"]
    gw2, _ = mk(tmp_path)
    assert len(gw2.list_verdicts()) == 2 and gw2.agents.get(ag.id)
    p = tmp_path / "ledger.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace('"allow"', '"deny"', 1), encoding="utf-8")
    assert not Ledger(p).verify()["ok"]


def test_rules_roundtrip_and_validation(tmp_path):
    gw, _ = mk(tmp_path)
    r = gw.rules
    r["max_order_value"]["value"] = 1000
    gw.set_rules(r)
    assert mk(tmp_path)[0].rules["max_order_value"]["value"] == 1000
    with pytest.raises(ValueError):
        gw.set_rules({"max_order_value": {"enabled": True, "value": "abc"}})
    with pytest.raises(ValueError):
        gw.set_rules({"nope": {"enabled": True}})


class FakeExec:
    def __init__(self):
        self.sent = []

    def submit(self, order):
        self.sent.append(order)
        return types.SimpleNamespace(status="accepted")


def test_execution_only_allowed_enforce_enabled():
    ex = FakeExec()
    gw, clock = mk(executor=ex, enabled=True)
    shadow, _ = gw.agents.create("s")
    enf, _ = gw.agents.create("e", mode="enforce")
    assert gw.propose(shadow, intent(client_id="1"))["executed"] is False
    assert gw.propose(enf, intent(symbol="MTSS", client_id="2", quantity=10))["executed"] is True
    assert gw.propose(enf, intent(symbol="GAZP", client_id="3", side="sell", quantity=1000))["executed"] is False
    assert len(ex.sent) == 1
    gw2, _ = mk(executor=ex, enabled=False)
    e2, _ = gw2.agents.create("e", mode="enforce")
    v = gw2.propose(e2, intent(client_id="4"))
    assert v["decision"] == "allow" and not v["executed"] and "не подключено" in v["execution_note"]


def test_context_profiles():
    gw, _ = mk()
    mn, _ = gw.agents.create("m", profile="minimal")
    fu, _ = gw.agents.create("f", profile="full")
    assert "rsi14" not in gw.get_context(mn, "SBER") and "positions" in gw.get_context(fu, "SBER")


def test_extras_plugin():
    gw, _ = mk()
    ag, _ = gw.agents.create("a")
    gw.context.register_extra("news", lambda s: {"risk": 0.2})
    assert gw.get_context(ag, "SBER")["extras"]["news"]["risk"] == 0.2


def test_demo_scenario():
    gw = build(".", demo=True)
    st = gw.stats()
    assert st["total"] >= 20 and st["allow"] and st["deny"] and st["revise"] and st["ledger"]["ok"]


@pytest.fixture()
def server(tmp_path):
    from finvex.web.server import make_server
    gw = build(".", demo=True)
    srv = make_server(gw, "127.0.0.1", 0, tmp_path)
    port = srv.server_address[1]
    srv.RequestHandlerClass  # noqa
    import finvex.web.server as S
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield gw, port
    srv.shutdown()
    srv.server_close()


def _req(port, path, method="GET", body=None, headers=None):
    h = {"Host": f"127.0.0.1:{port}", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    if data:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_http_gateway_and_ui(server):
    gw, port = server
    ag, key = gw.agents.create("http-agent")
    auth = {"Authorization": f"Bearer {key}"}
    assert _req(port, "/gateway/v1/limits")[0] == 401
    code, body = _req(port, "/gateway/v1/propose", "POST", intent(client_id="h1", symbol="MTSS"), auth)
    assert code == 200 and json.loads(body)["decision"] == "allow"
    assert _req(port, "/gateway/v1/context?symbol=SBER", headers=auth)[0] == 200
    assert _req(port, "/api/state")[0] == 401
    page = _req(port, "/")[1].decode()
    token = page.split('name="finvex-token" content="')[1].split('"')[0]
    code, body = _req(port, "/api/state", headers={"X-Token": token})
    assert code == 200 and json.loads(body)["stats"]["total"] >= 20
    assert _req(port, "/api/state", headers={"X-Token": token, "Host": "evil.com"})[0] == 403
    assert _req(port, "/api/rules", "PUT", {"rules": {}}, {"X-Token": token, "Origin": "http://evil.com"})[0] == 403
    code, body = _req(port, "/api/asset?symbol=SBER&range=1w", headers={"X-Token": token})
    assert code == 200 and len(json.loads(body)["points"]) > 10
    assert _req(port, "/api/asset?symbol=ZZZ&range=1w", headers={"X-Token": token})[0] == 404
    code, body = _req(port, "/api/intent", "POST", intent(client_id="u1", symbol="GMKN"), {"X-Token": token})
    assert code == 200 and json.loads(body)["agent_name"] == "Ручная проверка"
    assert _req(port, f"/api/export.csv?t={token}")[0] == 200


def test_mcp_end_to_end(server):
    gw, port = server
    ag, key = gw.agents.create("mcp-agent")
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "propose_order", "arguments": intent(symbol="MTSS", client_id="m1")}}]
    p = subprocess.run([sys.executable, "-m", "finvex.gateway.mcp", "--url", f"http://127.0.0.1:{port}", "--key", key],
                       input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, encoding="utf-8", timeout=30)
    out = [json.loads(x) for x in p.stdout.splitlines()]
    assert [o["id"] for o in out] == [1, 2, 3] and len(out[1]["result"]["tools"]) == 4
    assert json.loads(out[2]["result"]["content"][0]["text"])["decision"] == "allow"


def test_chat_bridge_with_fake_market_agent(tmp_path, monkeypatch):
    ma = types.ModuleType("market_agent")

    class Session:
        def __init__(self, out_dir, use_kronos=True):
            self.out_dir, self.turn_images = out_dir, []

    class Sandbox:
        def __init__(self, sess): pass

    class Agent:
        def __init__(self, sess, verbose=False): self.sess = sess

        def ask(self, text):
            (self.sess.out_dir / "c1.png").write_bytes(b"\x89PNG")
            self.sess.turn_images = [self.sess.out_dir / "c1.png"]
            return "ответ: " + text

    ma.Session, ma.Sandbox, ma.Agent = Session, Sandbox, Agent
    import finvex
    monkeypatch.setattr(finvex, "market_agent", ma, raising=False)
    monkeypatch.setitem(sys.modules, "finvex.market_agent", ma)
    from finvex.web.chat import Chat
    chat = Chat(tmp_path)
    assert chat.available
    res = chat.ask("s1", "привет")
    assert res["answer"] == "ответ: привет" and res["images"] == ["c1.png"]
    assert chat.image_path("s1", "c1.png") and chat.image_path("s1", "../x.png") is None


def test_real_market_agent_imports_and_builds_session(tmp_path):
    """Реальный чат-агент импортируется (раньше падал без market_stats) и создаёт сессию без сети."""
    from finvex.web.chat import Chat
    chat = Chat(tmp_path)
    assert chat.available, chat.note
    assert chat._session("s1")[1] is not None
