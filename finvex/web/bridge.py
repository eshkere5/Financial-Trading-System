"""
Мост между веб-интерфейсом и market_agent: события шагов для живой ленты, команда track() для панели актива,
тёмная тема графиков. market_agent.py при этом не меняется.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

from cycler import cycler

from finvex import market_agent as mg

DARK_RC = {
    "figure.facecolor": "#000000", "axes.facecolor": "#000000", "savefig.facecolor": "#000000", "text.color": "#ffffff",
    "axes.labelcolor": "#b4b4b4", "axes.edgecolor": "#2a2a2e", "xtick.color": "#949494", "ytick.color": "#949494",
    "grid.color": "#1c1c20", "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.facecolor": "#0d0d0f", "legend.edgecolor": "#2a2a2e",
    "axes.prop_cycle": cycler(color=["#2f80ff", "#f2f2f3", "#8a8c92", "#e5566f", "#44a282", "#d9a441"]),
}

WEB_NOTE = (
    "\nИнтерфейс веб: справа панель графика отслеживаемого актива. Чтобы показать актив там, вызови track(symbol) внутри python. "
    "Графики matplotlib оформлены под чёрный фон: основной ряд синий, не задавай свои светлые фоны. "
    "Отвечай в Markdown (заголовки ###, списки, таблицы), коротко."
)


class WebSandbox(mg.Sandbox):
    def __init__(self, sess: mg.Session) -> None:
        super().__init__(sess)
        self.ns["track"] = lambda symbol, market=None: (sess.emit or (lambda e: None))(
            {"type": "track", "symbol": mg.get_instrument(sess, symbol, market).symbol})

    def run(self, code: str, timeout: float = 150.0) -> dict[str, Any]:
        with mg.ma._plt().rc_context(DARK_RC):
            return super().run(code, timeout)


class WebAgent(mg.Agent):
    def __init__(self, sess: mg.Session, verbose: bool = False) -> None:
        super().__init__(sess, verbose)
        sess.sandbox = WebSandbox(sess)
        sess.emit = self._emit
        self.on_event: Optional[Callable[[dict], None]] = None

    def _emit(self, ev: dict) -> None:
        if self.on_event:
            self.on_event(ev)

    def _system(self) -> dict:
        m = super()._system()
        m["content"] += WEB_NOTE
        return m

    def _run_tool(self, name: str, raw_args: str) -> str:
        try:
            args = json.loads(raw_args or "{}")
            first = (str(args.get("code") or json.dumps(args, ensure_ascii=False)).strip().splitlines() or [""])[0][:140]
        except Exception:  # noqa: BLE001
            first = ""
        self._emit({"type": "step", "name": name, "detail": first})
        out = super()._run_tool(name, raw_args)
        if '"error"' in out[:12]:
            self._emit({"type": "step", "name": "retry", "detail": "ошибка в расчёте, модель исправляет"})
        return out
