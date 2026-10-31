"""Локальный HTTP-сервер: интерфейс, API интерфейса и API шлюза для агентов. Только 127.0.0.1."""
from __future__ import annotations

import hmac
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from finvex.gateway.gateway import Gateway, GatewayError
from finvex.gateway.analytics import AnalyticsError
from finvex.gateway.ledger import Ledger
from finvex.modules import ModuleDisabled, ModuleError

from .chat import Chat
from .adapters_admin import AdaptersAdmin, AdminError
from .sandbox import Sandbox, SandboxError
from .llm_providers import PRESETS
from .settings import LlmSettings, SettingsError

STATIC = Path(__file__).parent / "static"
FILES = {"/": ("index.html", "text/html; charset=utf-8"), "/app.css": ("app.css", "text/css; charset=utf-8"),
         "/app.js": ("app.js", "application/javascript; charset=utf-8")}
CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; connect-src 'self'; "
       "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
RANGES = {"1d": ("1h", 12, "1 ч"), "1w": ("1h", 63, "1 ч"), "1m": ("1d", 22, "1 д"), "3m": ("1d", 66, "1 д"), "1y": ("1d", 250, "1 д")}
MAX_BODY = 64 * 1024
# Серверная проверка модуля - настоящая защита; скрытие элементов в интерфейсе только удобство.
API_GATES = {"chat": "hub", "sandbox/presets": "hub", "sandbox/run": "hub", "analytics": "analytics", "tsfm": "analytics", "adapters": "data",
             "adapters/check": "data", "verdicts": "guard", "rules": "guard", "agents": "guard", "intent": "guard", "context": "guard",
             "report": "guard", "export.csv": "guard", "agents/run": "guard", "sandbox/rules": "hub"}


def make_server(gw: Gateway, host: str, port: int, out_dir: Path, root: "Path | None" = None, llm_call=None, agent_llm=None) -> ThreadingHTTPServer:
    token = secrets.token_urlsafe(24)
    from finvex.analysis.kronos import service as tsfm_service
    from finvex.feeds.analytics_feeder import AnalyticsFeeder

    def tsfm():
        return tsfm_service.get_service(root)

    def tsfm_enabled():
        return (not getattr(gw, "demo", False) and gw.modules.is_enabled("analytics") and tsfm().enabled())

    def tsfm_call(df, symbol, tf="1d", samples=3):
        if not tsfm_enabled():
            return {"status": "disabled", "error": "TSFM or analytics module disabled (or demo mode)"}
        result = tsfm().forecast(df, symbol, tf, samples)
        if result.get("status") == "ok" and tsfm_enabled():
            AnalyticsFeeder(gw.analytics, tf=tf).push_kronos(symbol, result)
        return result

    chat = Chat(out_dir, kronos_enabled=tsfm_enabled, kronos_call=tsfm_call)
    llm = LlmSettings()
    admin = AdaptersAdmin(root or Path("."))
    from finvex.gateway.runner import AgentRunner
    from finvex.gateway.agents import parse_symbols
    runner = AgentRunner(gw, agent_llm, lambda: {"base_url": llm.public()["base_url"], "model": llm.public()["model"], "api_key": llm.api_key()})
    sandbox = Sandbox(gw, llm, llm_call, runner)
    ui_agent_lock = threading.Lock()
    bound = [port]

    def ui_agent():
        with ui_agent_lock:
            for a in gw.agents.list():
                if a.name == "Ручная проверка" and not a.revoked:
                    return a
            return gw.agents.create("Ручная проверка", "shadow", "full", daily_quota=100000)[0]

    class H(BaseHTTPRequestHandler):
        server_version = "Finvex"
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _err(self, e: GatewayError) -> None:
            self._json(e.http, {"error": {"code": e.code, "message": str(e)}})

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise GatewayError("too_large", "слишком большой запрос", 413)
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                raise GatewayError("bad_json", "некорректный JSON", 400)

        def _ui_ok(self, q: dict) -> bool:
            t = self.headers.get("X-Token") or (q.get("t") or [""])[0]
            return hmac.compare_digest(t, token)

        def _bearer(self) -> str:
            h = self.headers.get("Authorization", "")
            return h[7:].strip() if h.lower().startswith("bearer ") else ""

        def _dispatch(self, method: str) -> None:
            if self.headers.get("Host", "") not in (f"127.0.0.1:{bound[0]}", f"localhost:{bound[0]}"):
                return self._send(403, b"forbidden", "text/plain")
            u = urlparse(self.path)
            path, q = u.path, parse_qs(u.query)
            try:
                if method == "GET" and path == "/favicon.ico":
                    return self._send(204, b"", "image/x-icon")
                if method == "GET" and path in FILES:
                    name, ctype = FILES[path]
                    body = (STATIC / name).read_bytes()
                    if name == "index.html":
                        body = body.replace(b"__TOKEN__", token.encode())
                    return self._send(200, body, ctype)
                if method == "GET" and path.startswith("/fonts/") and path.endswith(".woff2") and "/" not in path[7:]:
                    f = STATIC / "fonts" / path[7:]
                    if f.is_file():
                        return self._send(200, f.read_bytes(), "font/woff2", {"Cache-Control": "public, max-age=86400"})
                if path.startswith("/gateway/v1/"):
                    return self._gateway(method, path[len("/gateway/v1/"):], q)
                if path.startswith("/api/"):
                    if method != "GET" and self.headers.get("Origin") not in (None, f"http://127.0.0.1:{bound[0]}", f"http://localhost:{bound[0]}"):
                        raise GatewayError("forbidden", "чужой источник запроса", 403)
                    if not self._ui_ok(q):
                        raise GatewayError("unauthorized", "нет токена интерфейса: обновите страницу", 401)
                    return self._api(method, path[5:], q)
                self._send(404, b"not found", "text/plain")
            except GatewayError as e:
                self._err(e)
            except Exception as e:
                self._json(500, {"error": {"code": "internal", "message": str(e)[:200]}})

        def _gateway(self, method: str, route: str, q: dict) -> None:
            ag = gw.authenticate(self._bearer())
            if method == "GET" and route == "modules":
                return self._json(200, gw.modules.capabilities())
            if method == "GET" and route == "limits":
                return self._json(200, gw.limits(ag))
            if method == "GET" and route == "context":
                return self._json(200, gw.get_context(ag, (q.get("symbol") or [""])[0]))
            if method == "POST" and route == "propose":
                return self._json(200, gw.propose(ag, self._body()))
            if method == "GET" and route.startswith("verdicts/"):
                return self._json(200, gw.get_verdict(ag, unquote(route.split("/", 1)[1])))
            raise GatewayError("not_found", "неизвестный маршрут", 404)

        def _api(self, method: str, route: str, q: dict) -> None:
            g = lambda k, d="": (q.get(k) or [d])[0]
            gate = API_GATES.get(route)
            if gate:
                try:
                    gw.modules.require(gate)
                except ModuleDisabled as exc:
                    raise GatewayError("module_off", str(exc), 409)
            if route == "modules":
                if method == "GET":
                    return self._json(200, {"items": gw.modules.list()})
                if method == "PATCH":
                    b = self._body()
                    try:
                        return self._json(200, {"items": gw.modules.set_enabled(str(b.get("id", "")), bool(b.get("enabled")))})
                    except ModuleError as exc:
                        raise GatewayError("bad_module", str(exc), 409)
            if method == "GET" and route == "state":
                return self._json(200, {"demo": getattr(gw, "demo", False), "stats": gw.stats(), "chat": chat.available, "llm": llm.public(),
                                        "chat_note": chat.note, "tsfm": {**tsfm().status(), "active": tsfm_enabled()}, "execution": {"enabled": gw.execution_enabled and gw.executor is not None}})
            if method == "GET" and route == "verdicts":
                items = gw.list_verdicts(int(g("limit", "100") or 100), g("decision"), g("agent"))
                return self._json(200, {"items": items})
            if method == "GET" and route == "report":
                return self._json(200, gw.report())
            if route == "settings":
                if method == "GET":
                    return self._json(200, {**llm.public(), "providers": PRESETS})
                if method == "PUT":
                    b = self._body()
                    try:
                        res = llm.save(b.get("base_url", ""), b.get("model", ""), b.get("api_key", ""), bool(b.get("clear")), str(b.get("provider") or ""))
                    except SettingsError as e:
                        raise GatewayError("bad_settings", str(e), 422)
                    chat.reset()
                    return self._json(200, res)
            if method == "POST" and route == "settings/check":
                return self._json(200, llm.check())
            if route == "adapters":
                if method == "GET":
                    return self._json(200, admin.describe())
                b = self._body()
                try:
                    if method == "PUT":
                        return self._json(200, admin.remove(str(b["remove"])) if b.get("remove") else admin.add(b.get("name", ""), b.get("type", ""), b.get("params") or {}))
                except AdminError as e:
                    raise GatewayError("bad_adapter", str(e), 422)
            if method == "POST" and route == "adapters/check":
                try:
                    return self._json(200, admin.check(str(self._body().get("name", ""))))
                except AdminError as e:
                    raise GatewayError("bad_adapter", str(e), 422)
            if method == "GET" and route == "sandbox/presets":
                return self._json(200, sandbox.info())
            if method == "POST" and route == "sandbox/run":
                b = self._body()
                try:
                    return self._json(200, sandbox.run(str(b.get("preset", "")), str(b.get("symbol", "")), str(b.get("note", "")), str(b.get("prompt", "")),
                                                       str(b.get("source") or "model"), {k: b.get(k) for k in ("base_url", "model", "api_key", "agent", "url")}))
                except SandboxError as e:
                    raise GatewayError("sandbox", str(e), 409 if "ключ" in str(e) else 422)
            if method == "GET" and route == "tsfm":
                return self._json(200, {**tsfm().status(), "active": tsfm_enabled()})
            if method == "POST" and route == "sandbox/rules":
                try:
                    return self._json(200, sandbox.rules_check(str(self._body().get("symbol", ""))))
                except SandboxError as e:
                    raise GatewayError("sandbox", str(e), 422)
            if route == "analytics":
                if method == "GET":
                    return self._json(200, {"items": gw.analytics.all()})
                if method == "PUT":
                    b = self._body()
                    try:
                        if isinstance(b.get("news"), dict):
                            gw.analytics.add_news(b.get("symbol", ""), b["news"])
                        if isinstance(b.get("kronos"), dict):
                            k = b["kronos"]
                            gw.analytics.set_kronos(b.get("symbol", ""), k.get("score"), k.get("valid_until"), k.get("tightness"))
                        if not isinstance(b.get("news"), dict) and not isinstance(b.get("kronos"), dict):
                            raise AnalyticsError("нужен объект news или kronos")
                    except AnalyticsError as e:
                        raise GatewayError("bad_analytics", str(e), 422)
                    return self._json(200, {"ok": True})
            if method == "GET" and route == "export.csv":
                body = ("\ufeff" + Ledger.to_csv(reversed(gw.list_verdicts(1000)))).encode("utf-8")
                return self._send(200, body, "text/csv; charset=utf-8", {"Content-Disposition": 'attachment; filename="finvex_verdicts.csv"'})
            if method == "GET" and route == "asset":
                return self._asset(g("symbol").upper(), g("range", "1w"))
            if method == "GET" and route == "context":
                return self._json(200, gw.get_context(ui_agent(), g("symbol").upper()))
            if method == "POST" and route == "intent":
                return self._json(200, gw.propose(ui_agent(), self._body()))
            if route == "rules":
                if method == "GET":
                    from finvex.gateway.rules import RULES_META
                    return self._json(200, {"meta": RULES_META, "rules": gw.rules})
                if method == "PUT":
                    try:
                        return self._json(200, {"rules": gw.set_rules(self._body().get("rules") or {})})
                    except ValueError as e:
                        raise GatewayError("bad_rules", str(e), 422)
            if method == "POST" and route == "agents/run":
                b = self._body()
                try:
                    syms = parse_symbols(b["symbols"]) if b.get("symbols") else None
                except ValueError as e:
                    raise GatewayError("bad_symbols", str(e), 422)
                return self._json(200, runner.run(str(b.get("id", "")), syms))
            if route == "agents":
                if method == "GET":
                    cnt = gw.agent_counts()
                    return self._json(200, {"items": [{**a.public(), "verdicts": cnt.get(a.id, 0), "has_key": bool(gw.agents.secret(a.id, "api_key"))}
                                                      for a in gw.agents.list() if a.name != "Ручная проверка" or cnt.get(a.id)]})
                if method == "DELETE":
                    return self._json(200, gw.delete_agent(g("id")))
                b = self._body()
                try:
                    if method == "POST":
                        kind = str(b.get("kind") or "external")
                        ag, key = gw.agents.create(b.get("name", ""), b.get("mode", "shadow"), b.get("context_profile", "standard"),
                                                   created=gw.clock().isoformat(timespec="seconds"), kind=kind,
                                                   config=b.get("config") or {}, api_key=str(b.get("api_key") or ""))
                        return self._json(200, {"agent": ag.public(), "key": key if kind == "external" else None})
                    if method == "PATCH":
                        kw = {k: b[k] for k in ("mode", "context_profile", "revoked", "config", "api_key", "clear_api_key") if k in b}
                        return self._json(200, {"agent": gw.agents.update(b.get("id", ""), **kw).public()})
                except (ValueError, KeyError) as e:
                    raise GatewayError("bad_agent", str(e), 422)
            if method == "POST" and route == "chat":
                b = self._body()
                text, sid = str(b.get("text", ""))[:2000], str(b.get("sid", ""))
                ag = next((a for a in gw.agents.list() if a.id == b.get("agent")), None)
                if ag is not None and ag.kind == "llm":
                    # диалог с самим агентом: вопрос уходит его модели с его ролью, а не аналитику Finvex
                    return self._json(200, {"answer": runner.chat(ag.id, text), "images": [], "via": "agent"})
                if not chat.available:
                    raise GatewayError("chat_off", chat.note, 503)
                if not llm.has_key():
                    raise GatewayError("no_key", "Нужен ключ LLM: добавьте его во вкладке «Настройки»", 409)
                if ag is not None:
                    sid = "agent-" + ag.id
                    last = "; ".join(f"{v['intent']['side']} {v['intent']['symbol']} {v['intent']['quantity']:g} -> {v['decision']}"
                                     for v in gw.list_verdicts(5, "", ag.id))
                    text = (f"[Контекст: чат агента «{ag.name}», режим {ag.mode}. Последние решения шлюза: {last or 'нет'}. "
                            f"Отвечай про этого агента и его бумаги.]\n{text}")
                res = chat.ask(sid, text)
                if ag is not None:
                    res = {**res, "via": "analyst", "note": f"Агент «{ag.name}» работает снаружи или по адресу, и Finvex не может вести с ним диалог. Отвечает аналитик Finvex по решениям этого агента в шлюзе."}
                return self._json(200, res)
            if method == "GET" and route == "chart":
                p = chat.image_path(g("sid"), g("name"))
                if not p:
                    raise GatewayError("not_found", "файл не найден", 404)
                return self._send(200, p.read_bytes(), "image/png")
            raise GatewayError("not_found", "неизвестный маршрут", 404)

        def _asset(self, symbol: str, rng: str) -> None:
            if rng not in RANGES:
                raise GatewayError("bad_range", "диапазон: 1d, 1w, 1m, 3m, 1y", 422)
            tf, bars, label = RANGES[rng]
            try:
                df = gw.candles(symbol, tf, bars)
            except Exception as e:
                msg = str(e)
                raise GatewayError("no_data", msg if msg.startswith("нет данных") else f"нет данных по {symbol or '?'}: {msg[:100]}", 404)
            pts = [[ts.isoformat(timespec="minutes"), round(float(c), 4)] for ts, c in zip(df.index, df["close"])]
            self._json(200, {"symbol": symbol, "timeframe": label, "points": pts, "first": pts[0][1], "last": pts[-1][1],
                             "high": round(float(df["high"].max()), 4), "low": round(float(df["low"].min()), 4),
                             "volume": round(float(df["volume"].sum()), 0), "synthetic": bool(getattr(gw, "demo", False))})

        def do_GET(self): self._dispatch("GET")
        def do_POST(self): self._dispatch("POST")
        def do_PUT(self): self._dispatch("PUT")
        def do_PATCH(self): self._dispatch("PATCH")
        def do_DELETE(self): self._dispatch("DELETE")

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    bound[0] = srv.server_address[1]
    return srv
