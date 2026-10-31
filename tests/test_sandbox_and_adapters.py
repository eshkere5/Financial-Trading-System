import json
from pathlib import Path

import pytest
import yaml

from finvex.web.adapters_admin import AdaptersAdmin, AdminError
from finvex.web.sandbox import Sandbox, SandboxError
from tests.test_gateway import mk


class FakeSettings:
    def has_key(self): return True


def box(reply):
    gw, _ = mk()
    calls = []

    def llm(system, user):
        calls.append((system, user))
        return (reply if isinstance(reply, str) else json.dumps(reply)), {"prompt": 10, "completion": 5}

    sb = Sandbox(gw, FakeSettings(), llm)
    return gw, sb, calls


GOOD = {"action": "buy", "quantity": 5, "order_type": "market", "limit_price": None, "confidence": 0.7, "rationale": "Тренд вверх, риск ограничен стопом"}


def test_full_cycle_model_to_verdict_in_shadow():
    gw, sb, calls = box(GOOD)
    out = sb.run("trend", "SBER", "осторожно")
    assert out["verdict"]["decision"] == "allow" and out["verdict"]["executed"] is False and out["verdict"]["mode"] == "shadow"
    assert "SBER" in calls[0][1] and "осторожно" in calls[0][1] and out["tokens"]["prompt"] == 10


def test_sandbox_agent_is_forced_back_to_shadow():
    gw, sb, _ = box(GOOD)
    sb.run("trend", "SBER")
    ag = next(a for a in sb.lab.agents.list() if a.name.startswith("Песочница"))
    sb.lab.agents.update(ag.id, mode="enforce")
    assert sb.run("trend", "SBER")["verdict"]["mode"] == "shadow"


def test_hold_sends_nothing_to_gateway():
    gw, sb, _ = box({"action": "hold", "rationale": "нет основания"})
    out = sb.run("cautious", "SBER")
    assert out["verdict"] is None and "hold" in out["note"] and not gw.list_verdicts()


def test_hallucinated_ticker_is_stopped_before_the_model_is_called():
    gw, sb, calls = box(GOOD)
    out = sb.run("trend", "NOSUCHTICKER")
    assert out["verdict"] is None and "недоступен" in out["note"] and not calls


@pytest.mark.parametrize("reply", ["не json вообще", "[1,2]", "{oops"])
def test_garbage_from_model_is_a_result_not_a_crash(reply):
    _, sb, _ = box(reply)
    out = sb.run("trend", "SBER")
    assert out["verdict"] is None and "JSON" in out["note"]


def test_model_garbage_fields_are_reported_as_model_error():
    _, sb, _ = box({**GOOD, "quantity": -3})
    out = sb.run("trend", "SBER")
    assert out["verdict"] is None and "формат" in out["note"]


def test_aggressive_agent_is_limited_by_gateway():
    _, sb, _ = box({**GOOD, "quantity": 5000})
    assert sb.run("aggressive", "SBER")["verdict"]["decision"] == "revise"


def test_custom_needs_prompt_and_unknown_preset_rejected():
    _, sb, _ = box(GOOD)
    with pytest.raises(SandboxError):
        sb.run("custom", "SBER", custom_prompt="  ")
    with pytest.raises(SandboxError):
        sb.run("nope", "SBER")
    assert sb.run("custom", "SBER", custom_prompt="Покупай на откате")["verdict"]


def test_no_key_blocks_real_calls(tmp_path):
    gw, _ = mk()

    class NoKey:
        def has_key(self): return False

    with pytest.raises(SandboxError):
        Sandbox(gw, NoKey()).run("trend", "SBER")


def test_code_fences_around_json_are_tolerated():
    _, sb, _ = box("```json\n" + json.dumps(GOOD) + "\n```")
    assert sb.run("trend", "SBER")["verdict"]["decision"] == "allow"


# ---- адаптеры
@pytest.fixture
def root(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "providers.yaml").write_text(yaml.safe_dump({
        "default_market_data": "moex", "market_data": {"moex": {"type": "moex_iss", "params": {}}},
        "execution": {"type": "disabled", "enabled": False}}), encoding="utf-8")
    return tmp_path


def test_add_and_merge_adapter(root):
    from finvex.core.providers_config import merged_config
    adm = AdaptersAdmin(root)
    d = adm.add("bb_linear", "bybit_public", {"category": "linear"})
    assert any(r["name"] == "bb_linear" and r["removable"] for r in d["market_data"])
    assert set(merged_config(root)["market_data"]) == {"moex", "bb_linear"}
    assert "providers.local.yaml" not in (root / "configs" / "providers.yaml").read_text()
    adm.remove("bb_linear")
    assert set(merged_config(root)["market_data"]) == {"moex"}


@pytest.mark.parametrize("name,typ,params", [("Bad Name", "bybit_public", {}), ("x1", "os:system", {}), ("x1", "nope", {}),
                                              ("moex", "bybit_public", {}), ("x1", "bybit_public", {"base_url": "http://evil"}), ("x1", "bybit_public", [1])])
def test_unsafe_adapter_input_rejected(root, name, typ, params):
    with pytest.raises(AdminError):
        AdaptersAdmin(root).add(name, typ, params)


def test_cannot_remove_main_config_entries_or_enable_execution(root):
    adm = AdaptersAdmin(root)
    with pytest.raises(AdminError):
        adm.remove("moex")
    assert adm.describe()["execution"]["enabled"] is False
    assert not hasattr(adm, "enable_execution")
