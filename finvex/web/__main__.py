from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from finvex.gateway.factory import build

from .server import make_server


def main() -> None:
    ap = argparse.ArgumentParser(prog="finvex.web")
    ap.add_argument("--demo", action="store_true", help="синтетические данные, без сети")
    ap.add_argument("--root", default=".")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", default=None, help="папка для графиков чата")
    ap.add_argument("--open", action="store_true", help="открыть браузер")
    a = ap.parse_args()
    gw = build(a.root, demo=a.demo)
    srv = make_server(gw, "127.0.0.1", a.port, Path(a.out or Path(a.root) / "data" / "web_out"), Path(a.root))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"Finvex: {url}  ({'демо, синтетические данные' if a.demo else 'живые данные'}). Остановка: Ctrl+C")
    if a.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
