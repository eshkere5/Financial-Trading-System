import json
import re
import threading
from pathlib import Path

import pytest

from finvex.gateway.gateway import GatewayError
from finvex.modules import ModuleError, ModuleRegistry
from tests.test_gateway import FakeExec, intent, mk, _req

STATIC = Path(__file__).resolve().parent.parent / "finvex" / "web" / "static"


def test_defaults_dependencies_and_persistence(tmp_path):
    path = tmp_path / "modules.json"
    reg = ModuleRegistry(path)
    assert all(m["enabled"] for m in reg.list())
    with pytest.raises(ModuleError):
        reg.set_enabled("data", False)
    reg.set_enabled("hub", False)
    reg.set_enabled("guard", False)
    with pytest.raises(ModuleError):
        reg.set_enabled("hub", True)
    assert ModuleRegistry(path).is_enabled("guard") is False
    with pytest.raises(ModuleError):
        reg.set_enabled("nope", True)


def test_corrupt_state_falls_back_and_repairs(tmp_path):
    path = tmp_path / "modules.json"
    path.write_text("{broken", encoding="utf-8")
    assert all(m["enabled"] for m in ModuleRegistry(path).list())
    path.write_text(json.dumps({"data": False}), encoding="utf-8")
    assert [m["id"] for m in ModuleRegistry(path).list() if m["enabled"]] == ["analytics"]


def test_guard_off_is_fail_closed_and_does_not_execute():
    ex = FakeExec()
    gw, _ = mk(executor=ex, enabled=True)
    ag, _ = gw.agents.create("a", mode="enforce")
    gw.modules.set_enabled("hub", False)
    gw.modules.set_enabled("guard", False)
    with pytest.raises(GatewayError) as e:
        gw.propose(ag, intent())
    assert e.value.http == 503 and e.value.code == "module_off"
    assert not ex.sent and gw.list_verdicts() == []
    gw.modules.set_enabled("guard", True)
    assert gw.propose(ag, intent())["decision"] == "allow" and ex.sent


def test_guard_works_without_analytics_module():
    gw, _ = mk()
    ag, _ = gw.agents.create("a")
    gw.analytics.add_news("SBER", {"id": "n", "score": -0.9, "category": "company", "ttl_hours": 10})
    assert gw.propose(ag, intent(client_id="c1"))["context"]["analytics_used"].get("news_ids") == ["n"]
    gw.modules.set_enabled("hub", False)
    gw.modules.set_enabled("analytics", False)
    assert gw.propose(ag, intent(client_id="c2"))["context"]["analytics_used"] == {}


@pytest.fixture
def srv(tmp_path):
    from finvex.web.server import make_server
    gw, _ = mk(tmp_path / "gw")
    root = tmp_path / "p"
    (root / "configs").mkdir(parents=True)
    (root / "configs/providers.yaml").write_text("market_data: {}\nexecution: {type: disabled, enabled: false}\n", encoding="utf-8")
    s = make_server(gw, "127.0.0.1", 0, tmp_path / "out", root)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    port = s.server_address[1]
    page = _req(port, "/")[1].decode()
    tok = {"X-Token": page.split('name="finvex-token" content="')[1].split('"')[0]}
    yield gw, port, tok, page
    s.shutdown(); s.server_close()


def test_modules_api_toggle_and_server_side_gates(srv):
    gw, port, tok, page = srv
    assert _req(port, "/api/modules")[0] == 401
    items = json.loads(_req(port, "/api/modules", headers=tok)[1])["items"]
    assert [m["id"] for m in items] == ["data", "analytics", "guard", "hub"] and all(m["ui"] for m in items)
    assert _req(port, "/api/modules", "PATCH", {"id": "data", "enabled": False}, tok)[0] == 409
    assert _req(port, "/api/modules", "PATCH", {"id": "analytics", "enabled": False}, tok)[0] == 200
    assert _req(port, "/api/analytics", headers=tok)[0] == 409
    assert _req(port, "/api/modules", "PATCH", {"id": "hub", "enabled": False}, tok)[0] == 200
    assert _req(port, "/api/chat", "POST", {"text": "x"}, tok)[0] == 409
    assert _req(port, "/api/sandbox/run", "POST", {"preset": "trend", "symbol": "SBER"}, tok)[0] == 409
    assert _req(port, "/api/modules", "PATCH", {"id": "guard", "enabled": False}, tok)[0] == 200
    for route in ("/api/verdicts", "/api/rules", "/api/agents", "/api/report", "/api/context?symbol=SBER", "/api/export.csv"):
        assert _req(port, route, headers=tok)[0] == 409, route
    assert _req(port, "/api/intent", "POST", intent(), tok)[0] == 409
    assert _req(port, "/api/state", headers=tok)[0] == 200          # состояние и модули доступны всегда
    assert _req(port, "/api/modules", headers=tok)[0] == 200


def test_agent_api_lists_capabilities_and_guard_off_returns_503(srv):
    gw, port, tok, _ = srv
    ag, key = gw.agents.create("api-agent")
    auth = {"Authorization": f"Bearer {key}"}
    assert {m["id"] for m in json.loads(_req(port, "/gateway/v1/modules", headers=auth)[1])["modules"]} == {"data", "analytics", "guard", "hub"}
    gw.modules.set_enabled("hub", False); gw.modules.set_enabled("guard", False)
    assert _req(port, "/gateway/v1/propose", "POST", intent(), auth)[0] == 503
    assert {m["id"] for m in json.loads(_req(port, "/gateway/v1/modules", headers=auth)[1])["modules"]} == {"data", "analytics"}


def test_ui_markup_declares_module_dependencies():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'data-view="modules"' in html and 'id="v-locked"' in html
    assert len(re.findall(r'data-requires="(guard|data)"', html)) == 2
    for view, mod in {"verdicts": "guard", "rules": "guard", "agents": "guard", "chat": "hub", "sandbox": "hub"}.items():
        assert f'{view}: "{mod}"' in js
