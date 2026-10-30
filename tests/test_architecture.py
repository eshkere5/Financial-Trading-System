"""Границы пакетов из docs/ARCHITECTURE.md проверяются кодом. Новое нарушение роняет тест."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "finvex"
ALLOWED = {
    "core": {"utils"}, "utils": set(), "venues": {"core", "utils"}, "analysis": {"core", "venues", "utils"},
    "decision": {"core", "analysis", "utils"}, "execution": {"core", "venues", "engine", "utils"},
    "gateway": {"core", "analysis", "modules", "utils"}, "modules": {"utils"}, "data": {"core", "venues", "utils"},
    "engine": {"core", "venues", "analysis", "decision", "execution", "data", "utils"},
}
KNOWN = {("venues/bybit/client.py", "engine")}   # убрать при переносе конфигов в core: список только сокращается
BROKER_SDK = ("pybit", "tinkoff", "t_tech", "finvex.execution", "finvex.venues")


def _imports(path: Path):
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.ImportFrom) and n.module and not n.level:
            yield n.module
        elif isinstance(n, ast.Import):
            for a in n.names:
                yield a.name


def _layer(mod: str):
    p = mod.split(".")
    return p[1] if p[0] == "finvex" and len(p) > 1 else None


def test_layers_respect_dependency_rules():
    bad = []
    for f in ROOT.rglob("*.py"):
        rel = f.relative_to(ROOT).as_posix()
        own = rel.split("/")[0]
        for m in _imports(f) if own in ALLOWED else ():
            tgt = _layer(m)
            if tgt and tgt != own and tgt not in ALLOWED[own] and (rel, tgt) not in KNOWN:
                bad.append(f"{rel} -> {m}")
    assert not bad, "нарушены границы пакетов:\n" + "\n".join(sorted(bad))


def test_gateway_and_web_have_no_path_to_broker():
    for pkg in ("gateway", "web"):
        bad = [f"{f.name}: {m}" for f in (ROOT / pkg).rglob("*.py") for m in _imports(f)
               if any(m == b or m.startswith(b + ".") for b in BROKER_SDK)]
        assert not bad, bad
