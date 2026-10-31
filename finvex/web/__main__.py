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
    ap.add_argument("--tsfm-symbols", default="", help="фоновые прогнозы: SBER,GAZP (без LLM и торговли)")
    ap.add_argument("--tsfm-interval", type=float, default=900.0)
    ap.add_argument("--tsfm-timeframe", choices=["15m", "1h", "4h", "1d"], default="1h")
    ap.add_argument("--open", action="store_true", help="открыть браузер")
    a = ap.parse_args()
    if a.tsfm_interval <= 0:
        ap.error("--tsfm-interval must be positive")
    gw = build(a.root, demo=a.demo)
    srv = make_server(gw, "127.0.0.1", a.port, Path(a.out or Path(a.root) / "data" / "web_out"), Path(a.root))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"Finvex: {url}  ({'демо, синтетические данные' if a.demo else 'живые данные'}). Остановка: Ctrl+C")
    feeder_stop = None
    symbols = list(dict.fromkeys(s.strip().upper() for s in a.tsfm_symbols.split(",") if s.strip()))
    if symbols and not a.demo:
        from finvex.analysis.kronos.service import get_service
        from finvex.feeds.analytics_feeder import start_background, real_kronos_source
        service = get_service(a.root)

        def allowed():
            return gw.modules.is_enabled("analytics") and service.enabled()

        def source(symbol, tf):
            result = real_kronos_source(service)(symbol, tf)
            return result if allowed() else {"status": "disabled"}

        feeder_stop = start_background(gw.analytics, symbols, interval_s=a.tsfm_interval, tf=a.tsfm_timeframe,
                                       kronos_source=source, news_source=lambda s: [], enabled=allowed)
    if a.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if feeder_stop is not None:
            feeder_stop.set()
            feeder_stop.thread.join(timeout=2.0)
        srv.server_close()


if __name__ == "__main__":
    main()
