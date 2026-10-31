import datetime, fnmatch, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXC_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
            "node_modules", ".idea", ".vscode", "logs"}
EXC_FILES = ["*.pyc", "*.log", "*.pt", "*.pth", "*.safetensors", "*.bin", "*.parquet",
             "*.db", ".env", "*_key.yaml", "llm_shared.yaml", "*.zip"]
MAX_MB = 5

out = ROOT.parent / f"finvex_{datetime.datetime.now():%Y%m%d_%H%M%S}.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if p.is_dir() or EXC_DIRS & set(rel.parts):
            continue
        if any(fnmatch.fnmatch(p.name, x) for x in EXC_FILES):
            continue
        if p.stat().st_size > MAX_MB * 1024 * 1024:
            continue
        z.write(p, rel.as_posix())
print(out, round(out.stat().st_size / 1e6, 1), "MB")