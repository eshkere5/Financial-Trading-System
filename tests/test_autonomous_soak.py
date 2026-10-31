import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_autonomous_contour_survives_shock_kill_switch_and_recovery(tmp_path):
    """Короткий автономный прогон (около 15 с): шок аналитики, аварийная остановка, восстановление. Полный: tools/autonomous_soak.py --minutes 5."""
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "autonomous_soak.py"), "--minutes", "0.25", "--out", str(tmp_path)], capture_output=True, text=True, cwd=ROOT, timeout=120)
    s = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert r.returncode == 0, s["checks"]
    assert s["decisions"] >= 15 and s["flawed_through"] == 0 and s["allowed_buys_in_shock"] == 0 and not s["errors"]
    assert s["per_phase"]["C"].get("allow", 0) == 0 and (tmp_path / "chart_timeline.png").exists()
