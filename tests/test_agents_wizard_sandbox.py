"""Подключение агентов (нейросеть, агент по адресу, внешний), удаление, песочница как отдельный слой."""
import http.client
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from finvex.core import urlguard
from finvex.gateway.agents import normalize_config, parse_symbols
from finvex.gateway.decisions import DecisionError, analyst_decision, call_http, decision_to_intent, extract_json
from finvex.gateway.gateway import GatewayError
from finvex.gateway.runner import AgentRunner
from finvex.web.sandbox import Sandbox, SandboxError
from tests.test_gateway import intent, mk

STATIC = Path(__file__).resolve().parents[1] / "finvex" / "web" / "static"
PROMPT = "Ты осторожный трейдер."
BUY = {"action": "buy", "quantity": 5, "order_type": "market", "limit_price": None, "confidence": 0.7, "rationale": "Тренд вверх, риск ограничен стопом"}


class Peer:
    """Локальный HTTP-сервер вместо n8n или Dify."""

    def __init__(self, routes):
        peer = self
        self.routes, self.seen = routes, []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                peer.seen.append({"path": self.path, "body": json.loads(body or b"{}"), "auth": self.headers.get("Authorization")})
                status, headers, payload, delay = peer.routes.get(self.path, (404, {}, b"", 0))
                time.sleep(delay)
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except OSError:
                    pass  # ?????? ?????? ?????????? ??????: ???????? ????? ???????
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def peer():
    made = []
    def make(routes):
        p = Peer(routes)
        made.append(p)
        return p
    yield make
    for p in made:
        p.close()


def j(obj, status=200, headers=None, delay=0):
    return (status, {"Content-Type": "application/json", **(headers or {})}, json.dumps(obj).encode(), delay)


# --- удаление и журнал ----------------------------------------------------------------------------
def test_delete_agent_purges_history_keeps_chain_valid_and_survives_restart(tmp_path):
    gw, _ = mk(tmp_path / "gw")
    a, _ = gw.agents.create("первый")
    b, _ = gw.agents.create("второй")
    gw.propose(a, intent(client_id="a1"))
    gw.propose(b, intent(client_id="b1"))
    gw.propose(a, intent(client_id="a2", symbol="GAZP"))
    head = gw.ledger.verify()["head"]
    assert gw.delete_agent(a.id) == {"deleted": a.id, "removed_verdicts": 2}
    assert gw.agents.get(a.id) is None and [v["agent_id"] for v in gw.list_verdicts()] == [b.id]
    ver = gw.ledger.verify()
    assert ver["ok"] and ver["checked"] == 2
    last = gw.ledger.rows()[-1]["data"]
    assert last["kind"] == "purge" and last["removed"] == 2 and last["previous_head"] == head and "agent_id" not in last
    text = (tmp_path / "gw" / "ledger.jsonl").read_text(encoding="utf-8")
    assert a.id not in text and '"a1"' not in text and '"a2"' not in text and "первый" not in text
    gw2, _ = mk(tmp_path / "gw")
    assert gw2.stats()["all_time"] == 1 and gw2.ledger.verify()["ok"] and gw2.agents.get(a.id) is None
    with pytest.raises(GatewayError):
        gw.delete_agent(a.id)


def test_delete_removes_secrets_and_empty_history_adds_no_record(tmp_path):
    gw, _ = mk(tmp_path / "gw")
    ag, _ = gw.agents.create("н", kind="llm", config={"system_prompt": PROMPT}, api_key="sk-own-12345678")
    assert gw.agents.secret(ag.id, "api_key") == "sk-own-12345678"
    assert gw.delete_agent(ag.id)["removed_verdicts"] == 0 and len(gw.ledger) == 0
    assert "sk-own" not in (tmp_path / "gw" / "agent_secrets.json").read_text(encoding="utf-8")


# --- виды агентов ----------------------------------------------------------------------------------
def test_agent_kinds_validation_and_authentication():
    gw, _ = mk()
    ext, key = gw.agents.create("внешний")
    assert gw.authenticate(key).id == ext.id
    llm, k2 = gw.agents.create("нейро", kind="llm", config={"system_prompt": PROMPT, "symbols": "sber, GAZP"})
    web, k3 = gw.agents.create("адрес", kind="http", config={"url": "http://localhost:5678/webhook/x", "symbols": "SBER"})
    assert gw.agents.authenticate(k2) is None and gw.agents.authenticate(k3) is None      # их вызывает сам Finvex, ключ не принимается
    assert llm.config["symbols"] == ["SBER", "GAZP"] and "key_hash" not in web.public()
    for kind, bad in (("llm", {"system_prompt": ""}), ("llm", {"system_prompt": PROMPT, "base_url": "http://example.com", "model": "m"}),
                      ("llm", {"system_prompt": PROMPT, "base_url": "https://example.com"}), ("llm", {"system_prompt": PROMPT, "symbols": "SB ER!"}),
                      ("http", {}), ("http", {"url": "http://example.com/x"}), ("http", {"url": "https://u:p@example.com/x"}),
                      ("http", {"url": "http://192.168.1.5/x"}), ("llm", {"system_prompt": PROMPT, "symbols": ",".join(f"A{i}" for i in range(11))})):
        with pytest.raises(ValueError):
            normalize_config(kind, bad)
    with pytest.raises(ValueError):
        gw.agents.create("x", kind="mcp")
    assert parse_symbols("sber, GAZP;sber") == ["SBER", "GAZP"]


def test_last_seen_is_recorded_on_agent_calls_and_survives_restart(tmp_path):
    gw, clock = mk(tmp_path / "gw")
    ag, key = gw.agents.create("внешний")
    assert gw.agents.get(ag.id).calls == 0 and not gw.agents.get(ag.id).last_seen
    gw.authenticate(key)
    gw.authenticate(key)
    got = gw.agents.get(ag.id)
    assert got.calls == 2 and got.last_seen.startswith("2026-10-21T12:00")
    gw2, _ = mk(tmp_path / "gw")                                  # первое обращение сохранено сразу
    assert gw2.agents.get(ag.id).calls >= 1
    with pytest.raises(GatewayError):
        gw.authenticate("bad-key")


# --- разбор ответа и аналитик ----------------------------------------------------------------------
def test_extract_json_variants_and_intent():
    assert extract_json("```json\n" + json.dumps(BUY) + "\n```")["action"] == "buy"
    for bad in ("нет json", "[1,2]", "{oops"):
        with pytest.raises(DecisionError):
            extract_json(bad)
    i = decision_to_intent(BUY, "SBER", "t")
    assert i["side"] == "buy" and i["client_id"].startswith("t-") and i["symbol"] == "SBER"


def ctx(**kw):
    base = {"symbol": "SBER", "price": 300.0, "equity": 1_000_000.0, "regime": "BULL_TREND", "adx": 30.0, "rsi14": 55.0, "position_qty": 0.0}
    base.update(kw)
    return base


def test_analyst_is_transparent_and_conservative():
    buy = analyst_decision(ctx())
    assert buy["action"] == "buy" and buy["quantity"] == 16 and "ADX" in buy["rationale"]        # 0,5% капитала
    assert analyst_decision(ctx(symbol="BTCUSDT", price=95000.0))["quantity"] == pytest.approx(0.052632, abs=1e-6)
    assert analyst_decision(ctx(adx=10))["action"] == "hold" and analyst_decision(ctx(rsi14=80))["action"] == "hold"
    assert analyst_decision(ctx(regime="RANGE"))["action"] == "hold" and analyst_decision(ctx(price=0))["action"] == "hold"
    sell = analyst_decision(ctx(position_qty=40, regime="BEAR_TREND"))
    assert sell["action"] == "sell" and sell["quantity"] == 40
    assert analyst_decision(ctx(position_qty=40, rsi14=80))["action"] == "sell"
    assert analyst_decision(ctx(position_qty=40))["action"] == "hold"


# --- вызов агента по адресу ------------------------------------------------------------------------
def test_call_http_accepts_common_response_shapes_and_sends_payload(peer):
    p = peer({"/direct": j(BUY), "/wrapped": j({"output": json.dumps(BUY)}), "/n8n": j([{"json": BUY}]),
              "/text": (200, {"Content-Type": "text/plain"}, ("Вот решение: " + json.dumps(BUY)).encode(), 0),
              "/nested": j({"data": {"result": {"text": json.dumps(BUY)}}})})
    for path in ("/direct", "/wrapped", "/n8n", "/text", "/nested"):
        assert call_http(p.base + path, "tok", {"symbol": "SBER", "context": {"price": 1}})["action"] == "buy", path
    assert p.seen[0]["auth"] == "Bearer tok" and p.seen[0]["body"]["symbol"] == "SBER"
    assert call_http(p.base + "/direct", "", {})["quantity"] == 5 and p.seen[-1]["auth"] is None


def test_call_http_failures_are_explained_and_bounded(peer):
    p = peer({"/500": j({}, 500), "/redirect": (302, {"Location": "http://localhost/x"}, b"", 0), "/none": j({"hello": 1}), "/slow": j(BUY, delay=1.5),
              "/huge": (200, {"Content-Type": "application/json"}, b"x" * 300_000, 0), "/list": j([])})
    for path, text in (("/500", "HTTP 500"), ("/redirect", "перенаправлением"), ("/none", "нет решения"), ("/huge", "слишком большой"), ("/list", "нет решения")):
        with pytest.raises(DecisionError, match=text):
            call_http(p.base + path, "", {})
    with pytest.raises(DecisionError, match="не ответил"):
        call_http(p.base + "/slow", "", {}, timeout=0.3)
    with pytest.raises(DecisionError, match="адрес агента"):
        call_http("http://192.168.0.7/x", "", {})
    with pytest.raises(DecisionError, match="недоступен|не ответил"):
        call_http("http://127.0.0.1:1/x", "", {}, timeout=1)


# --- запуск подключённых агентов -------------------------------------------------------------------
def test_runner_llm_agent_hold_verdict_garbage_and_no_context():
    gw, _ = mk()
    ag, _ = gw.agents.create("н", kind="llm", config={"system_prompt": PROMPT, "symbols": "SBER"})
    answers = iter(['{"action":"hold"}', "```json\n" + json.dumps(BUY) + "\n```", "не знаю", '{"action":"jump"}'])
    r = AgentRunner(gw, lambda s, u: next(answers))
    res = [r.run(ag.id)["results"][0] for _ in range(4)]
    assert [x["status"] for x in res] == ["hold", "verdict", "error", "error"]
    assert res[1]["verdict"]["agent_id"] == ag.id and not res[1]["verdict"]["executed"] and len(gw.list_verdicts()) == 1
    assert r.run(ag.id, ["ZZZZ"])["results"][0]["status"] == "no_context"


def test_runner_http_agent_uses_context_and_secret(peer):
    p = peer({"/hook": j({"output": json.dumps(BUY)}), "/down": j({}, 503)})
    gw, _ = mk()
    ok, _ = gw.agents.create("n8n", kind="http", config={"url": p.base + "/hook", "symbols": "SBER", "system_prompt": "осторожно"}, api_key="k-12345678")
    bad, _ = gw.agents.create("упавший", kind="http", config={"url": p.base + "/down", "symbols": "SBER"})
    r = AgentRunner(gw)
    out = r.run(ok.id)["results"][0]
    assert out["status"] == "verdict" and out["verdict"]["decision"] == "allow"
    sent = p.seen[0]
    assert sent["auth"] == "Bearer k-12345678" and sent["body"]["symbol"] == "SBER" and sent["body"]["context"]["price"] > 0 and sent["body"]["instructions"] == "осторожно"
    res = r.run(bad.id)["results"][0]
    assert res["status"] == "error" and "HTTP 503" in res["note"] and len(gw.list_verdicts()) == 1


def test_runner_guards_kind_revoked_symbols_and_concurrency():
    gw, _ = mk()
    ext, _ = gw.agents.create("внешний")
    ag, _ = gw.agents.create("н", kind="llm", config={"system_prompt": PROMPT})
    runner = AgentRunner(gw, lambda s, u: '{"action":"hold"}')
    with pytest.raises(GatewayError, match="подключён снаружи"):
        runner.run(ext.id, ["SBER"])
    with pytest.raises(GatewayError, match="не заданы тикеры"):
        runner.run(ag.id)
    release, started = threading.Event(), threading.Event()
    def slow(s, u):
        started.set()
        release.wait(2)
        return '{"action":"hold"}'
    slow_runner = AgentRunner(gw, slow)
    t = threading.Thread(target=lambda: slow_runner.run(ag.id, ["SBER"]))
    t.start()
    started.wait(2)
    with pytest.raises(GatewayError, match="уже выполняется"):
        slow_runner.run(ag.id, ["SBER"])
    release.set()
    t.join(3)
    gw.agents.update(ag.id, revoked=True)
    with pytest.raises(GatewayError, match="отозван"):
        runner.run(ag.id, ["SBER"])


def test_shared_key_never_goes_to_a_custom_host():
    gw, _ = mk()
    shared = lambda: {"base_url": "https://shared.example/v1", "model": "sm", "api_key": "SHARED"}
    other = gw.agents.create("a", kind="llm", config={"system_prompt": PROMPT, "base_url": "https://other.example/v1", "model": "m"})[0]
    plain = gw.agents.create("b", kind="llm", config={"system_prompt": PROMPT})[0]
    r = AgentRunner(gw, default_llm=shared)
    assert r._endpoint(other)["api_key"] == "" and r._endpoint(plain)["api_key"] == "SHARED"


def test_chat_with_llm_agent_uses_its_role_and_verdicts():
    gw, _ = mk()
    ag, _ = gw.agents.create("н", kind="llm", config={"system_prompt": PROMPT})
    gw.propose(ag, intent(client_id="c1"))
    seen = {}
    def fake(system, user):
        seen.update(system=system, user=user)
        return "  Я купил SBER из-за тренда.  "
    assert AgentRunner(gw, fake).chat(ag.id, "почему?") == "Я купил SBER из-за тренда." and PROMPT in seen["system"] and "buy SBER" in seen["system"]
    http_agent, _ = gw.agents.create("адрес", kind="http", config={"url": "http://localhost:1/x"})
    with pytest.raises(GatewayError, match="подключён снаружи"):
        AgentRunner(gw, fake).chat(http_agent.id, "привет")


# --- песочница как отдельный слой ------------------------------------------------------------------
class FakeSettings:
    def has_key(self): return True
    def public(self): return {"base_url": "https://x.example/v1", "model": "m"}
    def api_key(self): return "k"


def box(reply=BUY, runner=False):
    gw, _ = mk()
    calls = []
    def llm(system, user):
        calls.append((system, user))
        return (reply if isinstance(reply, str) else json.dumps(reply)), {"prompt": 10, "completion": 5}
    return gw, Sandbox(gw, FakeSettings(), llm, AgentRunner(gw, lambda s, u: json.dumps(BUY)) if runner else None), calls


def test_sandbox_is_isolated_from_the_production_gateway():
    gw, sb, _ = box()
    out = sb.run("trend", "SBER")
    assert out["verdict"]["decision"] == "allow" and out["source"] == "model"
    assert not gw.list_verdicts() and len(gw.ledger) == 0 and not [a for a in gw.agents.list() if a.name.startswith("Песочница")]
    assert len(sb.lab.list_verdicts()) == 1 and sb.lab.executor is None and not sb.lab.execution_enabled
    gw.set_rules({**gw.rules, "kill_switch": {**gw.rules["kill_switch"], "enabled": True}}) if "kill_switch" in gw.rules else None
    assert sb._sync().rules == gw.rules                                # правила рабочие


def test_sandbox_analyst_needs_no_key_and_no_model():
    gw, _ = mk()
    class NoKey:
        def has_key(self): return False
    sb = Sandbox(gw, NoKey())
    out = sb.run("trend", "SBER", source="analyst")
    assert out["source"] == "analyst" and out["model"]["action"] in ("buy", "sell", "hold") and out["tokens"] == {}
    assert out["verdict"] is None or out["verdict"]["mode"] == "shadow"
    with pytest.raises(SandboxError, match="нужен ключ"):
        sb.run("trend", "SBER")                                         # источнику «модель» ключ нужен


def test_sandbox_runs_connected_agents_and_urls(peer):
    p = peer({"/hook": j({"output": json.dumps(BUY)})})
    gw, sb, _ = box(runner=True)
    llm_agent, _ = gw.agents.create("н", kind="llm", config={"system_prompt": PROMPT})
    ext, _ = gw.agents.create("внешний")
    out = sb.run("trend", "SBER", source="agent", options={"agent": llm_agent.id})
    assert out["verdict"]["decision"] == "allow" and out["preset"] == "н"
    for bad in (ext.id, "ag_nope", ""):
        with pytest.raises(SandboxError, match="выберите подключённого агента"):
            sb.run("trend", "SBER", source="agent", options={"agent": bad})
    out = sb.run("trend", "SBER", source="http", options={"url": p.base + "/hook", "api_key": "k"})
    assert out["verdict"]["decision"] == "allow" and p.seen[0]["auth"] == "Bearer k" and p.seen[0]["body"]["context"]["symbol"] == "SBER"
    out = sb.run("trend", "SBER", source="http", options={"url": p.base + "/missing"})
    assert out["verdict"] is None and "HTTP 404" in out["note"]
    with pytest.raises(SandboxError, match="адрес агента"):
        sb.run("trend", "SBER", source="http", options={"url": "http://10.0.0.5/x"})
    with pytest.raises(SandboxError, match="адрес API"):
        sb.run("trend", "SBER", source="custom_api", options={"base_url": "http://example.com/v1", "model": "m"})
    with pytest.raises(SandboxError, match="название модели"):
        sb.run("trend", "SBER", source="custom_api", options={"base_url": "http://localhost:4000/v1", "model": " "})
    with pytest.raises(SandboxError, match="неизвестный источник"):
        sb.run("trend", "SBER", source="magic")
    assert not gw.list_verdicts()


def test_sandbox_rules_check_runs_fixed_scenarios_without_touching_production():
    gw, sb, _ = box()
    res = sb.rules_check("SBER")
    titles = [i["title"] for i in res["items"]]
    assert len(titles) == 6 and titles[0].startswith("Небольшая покупка") and all(i["decision"] in ("allow", "revise", "deny") for i in res["items"])
    assert res["items"][0]["decision"] == "allow" and res["items"][1]["decision"] != "allow"        # втрое больше капитала не проходит
    assert not gw.list_verdicts() and len(gw.ledger) == 0
    assert sb.rules_check("ZZZZ")["items"] == []
    with pytest.raises(SandboxError):
        sb.rules_check(" ")


def test_sandbox_info_lists_sources_and_only_connected_callable_agents():
    gw, sb, _ = box()
    gw.agents.create("внешний")
    n, _ = gw.agents.create("нейро", kind="llm", config={"system_prompt": PROMPT})
    a, _ = gw.agents.create("адрес", kind="http", config={"url": "http://localhost:1/x"})
    gw.agents.update(a.id, revoked=True)
    info = sb.info()
    assert [s["id"] for s in info["sources"]] == ["model", "custom_api", "agent", "http", "analyst"] and [x["id"] for x in info["agents"]] == [n.id]


# --- весь путь через HTTP --------------------------------------------------------------------------
@pytest.fixture
def web(tmp_path):
    from finvex.web.server import make_server
    gw, _ = mk(tmp_path / "gw")
    (tmp_path / "p" / "configs").mkdir(parents=True)
    (tmp_path / "p" / "configs" / "providers.yaml").write_text("market_data: {}\nexecution: {type: disabled, enabled: false}\n", encoding="utf-8")
    answers = []
    srv = make_server(gw, "127.0.0.1", 0, tmp_path / "out", tmp_path / "p", agent_llm=lambda s, u: answers.pop(0) if answers else '{"action":"hold"}')
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    def req(path, method="GET", data=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            c.request(method, path, json.dumps(data).encode() if data is not None else None, headers or {})
            r = c.getresponse()
            body = r.read().decode()
            return r.status, (json.loads(body) if body[:1] in "{[" else body)
        finally:
            c.close()
    page = req("/")[1]
    tok = {"X-Token": page.split('name="finvex-token" content="')[1].split('"')[0]}
    yield SimpleNamespace(gw=gw, req=req, tok=tok, answers=answers)
    srv.shutdown()
    srv.server_close()


def test_http_agent_lifecycle_wizard_routes_chat_delete(web, peer):
    api = lambda *a, **k: web.req(*a, headers=web.tok, **k)
    hook = peer({"/hook": j(BUY)})
    s, ext = api("/api/agents", "POST", {"name": "n8n-бот", "kind": "external"})
    assert s == 200 and ext["key"].startswith("fvx_")
    bearer = {"Authorization": "Bearer " + ext["key"], "Content-Type": "application/json"}
    assert web.req("/gateway/v1/limits", headers=bearer)[0] == 200
    s, items = api("/api/agents")
    row = next(a for a in items["items"] if a["id"] == ext["agent"]["id"])
    assert row["calls"] == 1 and row["last_seen"] and row["kind"] == "external"             # мастер видит первое обращение
    s, llm = api("/api/agents", "POST", {"name": "ИИ", "kind": "llm", "api_key": "sk-1-12345678", "config": {"system_prompt": PROMPT, "symbols": "SBER"}})
    assert s == 200 and llm["key"] is None and "sk-1-12345678" not in json.dumps(llm)
    s, adr = api("/api/agents", "POST", {"name": "n8n по адресу", "kind": "http", "config": {"url": hook.base + "/hook", "symbols": "SBER"}})
    assert s == 200 and adr["key"] is None
    s, run = api("/api/agents/run", "POST", {"id": adr["agent"]["id"]})
    assert s == 200 and run["results"][0]["status"] == "verdict"
    web.answers.append("Купил из-за тренда.")
    s, chat = api("/api/chat", "POST", {"sid": "t1", "text": "почему?", "agent": llm["agent"]["id"]})
    assert s == 200 and chat["via"] == "agent" and chat["answer"] == "Купил из-за тренда."
    assert api("/api/agents", "POST", {"name": "плохой", "kind": "http", "config": {"url": "http://example.com/x"}})[0] == 422
    assert api("/api/agents", "POST", {"name": "плохой", "kind": "mcp"})[0] == 422
    assert api("/api/agents?id=" + adr["agent"]["id"], "DELETE")[1]["removed_verdicts"] == 1
    assert adr["agent"]["id"] not in [a["id"] for a in api("/api/agents")[1]["items"]] and api("/api/agents?id=" + adr["agent"]["id"], "DELETE")[0] == 404
    assert api("/api/state")[1]["stats"]["ledger"]["ok"]
    assert web.req("/api/agents?id=x", "DELETE")[0] == 401


def test_http_sandbox_routes(web):
    api = lambda *a, **k: web.req(*a, headers=web.tok, **k)
    s, info = api("/api/sandbox/presets")
    assert s == 200 and [x["id"] for x in info["sources"]][-1] == "analyst" and info["agents"] == []
    s, out = api("/api/sandbox/run", "POST", {"source": "analyst", "symbol": "SBER", "preset": "trend"})
    assert s == 200 and out["source"] == "analyst" and out["model"]["action"] in ("buy", "sell", "hold")
    s, chk = api("/api/sandbox/rules", "POST", {"symbol": "SBER"})
    assert s == 200 and len(chk["items"]) == 6
    assert api("/api/verdicts")[1]["items"] == []                       # песочница не пишет в рабочий журнал
    assert api("/api/sandbox/run", "POST", {"source": "http", "symbol": "SBER", "preset": "trend", "url": "http://10.0.0.5/x"})[0] == 422
    assert api("/api/sandbox/rules", "POST", {"symbol": ""})[0] == 422
    assert web.req("/api/sandbox/run", "POST", {"source": "analyst", "symbol": "SBER"})[0] == 401


def test_chat_with_external_agent_is_labelled_honestly(web, monkeypatch):
    from finvex.web.chat import Chat
    from finvex.web.settings import LlmSettings
    monkeypatch.setattr(LlmSettings, "has_key", lambda self: True)
    monkeypatch.setattr(Chat, "available", property(lambda self: True))
    monkeypatch.setattr(Chat, "ask", lambda self, sid, text: {"answer": "ответ аналитика", "images": []})
    ext, _ = web.gw.agents.create("внешний бот")
    s, out = web.req("/api/chat", "POST", {"sid": "z", "text": "что делал бот?", "agent": ext.id}, web.tok)
    assert s == 200 and out["via"] == "analyst" and "не может вести с ним диалог" in out["note"]


# --- интерфейс -------------------------------------------------------------------------------------
def test_ui_wizard_sandbox_and_removed_adapters_menu_are_consistent():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="adapters"' not in html and "loadAdapters" not in js and "/api/adapters" not in js
    missing = sorted({i for i in re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', js) if f'id="{i}"' not in html})
    assert not missing, missing
    for needle in ('id="ag-route"', 'value="n8n"', 'value="mcp"', 'value="code"', 'value="http"', 'value="llm"', 'id="sbx-source"', 'id="sbx-rules"', 'id="llm-provider"'):
        assert needle in html, needle
    for needle in ("method: \"DELETE\"", "/api/sandbox/rules", "showCreated", "waitForAgent", "last_seen"):
        assert needle in js, needle
