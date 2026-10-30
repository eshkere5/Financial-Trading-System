import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_fake(tmp_path, *extra):
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "live_testrun.py"), "--fake", "--rounds", "3", "--out", str(out), *extra],
                       capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stderr[-800:]
    ev = [json.loads(x) for x in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return out, ev, json.loads((out / "summary.json").read_text(encoding="utf-8"))


def test_dry_run_sends_nothing(tmp_path):
    out, ev, summ = run_fake(tmp_path)
    assert summ["fake"] and not summ["send"] and summ["orders_sent"] == 0 and not any(e.get("sent") for e in ev)
    assert {"allow", "hold", "revise"} <= set(summ["outcomes"]) and "СИНТЕТИЧЕСКИЙ" in (out / "report.md").read_text(encoding="utf-8")


def test_send_mode_respects_caps_and_only_sends_allowed(tmp_path):
    _, ev, summ = run_fake(tmp_path, "--send")
    sent = [e for e in ev if e.get("sent")]
    assert sent and all(e["outcome"] == "allow" and e["order_id"] for e in sent) and summ["orders_sent"] == len(sent)
    assert any("лимита тестового прогона" in (e.get("order_message") or "") for e in ev)   # крупные заявки не ушли


def test_max_orders_stops_the_run(tmp_path):
    _, _, summ = run_fake(tmp_path, "--send", "--max-orders", "1")
    assert summ["orders_sent"] == 1


def test_presentation_builds_from_a_run(tmp_path):
    out, _, _ = run_fake(tmp_path, "--send")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "make_presentation.py"), str(out)], capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stderr[-800:]
    from pptx import Presentation
    assert len(Presentation(str(out / "presentation.pptx")).slides) == 9
