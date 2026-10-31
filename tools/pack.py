"""Package public project source only; exclude private runtime state and weights."""
import datetime
import fnmatch
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXC_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
            "node_modules", ".idea", ".vscode", "logs", "archive", "models", "weights", "Kronos"}
EXC_FILES = ["*.pyc", "*.log", "*.pt", "*.pth", "*.safetensors", "*.bin", "*.parquet", "*.db", "*.sqlite*",
             ".env", ".env.*", "*key*.yaml", "*secret*.yaml", "*token*.yaml", "llm_shared.yaml", "deepseek_news.yaml",
             "*.user.yaml", "providers.local.yaml", "*.zip", "*.bak", "*.tmp"]


def include(path):
    rel = path.relative_to(ROOT)
    return (path.is_file() and not path.is_symlink() and not (EXC_DIRS & set(rel.parts))
            and rel.parts[0] not in {"data", "reports", "third_party", "output"}
            and not any(fnmatch.fnmatch(path.name, pat) for pat in EXC_FILES))


def main():
    out = ROOT.parent / f"finvex_{datetime.datetime.now():%Y%m%d_%H%M%S}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            if include(p):
                z.write(p, p.relative_to(ROOT).as_posix())
    print(out, round(out.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
