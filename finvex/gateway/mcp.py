"""MCP-сервер (stdio, JSON-RPC) для ИИ-агентов. Говорит только с HTTP-шлюзом и знает только ключ агента."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

TOOLS = [
    {"name": "get_context", "description": "Проверенный контекст по тикеру: цена, свежесть данных, режим рынка, индикаторы, позиция.",
     "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "get_limits", "description": "Действующие правила и остаток заявок на сегодня.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "propose_order", "description": "Отправить намерение на проверку. Возвращает вердикт: allow, revise или deny. Заявка сама не исполняется.",
     "inputSchema": {"type": "object", "properties": {
         "symbol": {"type": "string"}, "side": {"type": "string", "enum": ["buy", "sell"]}, "quantity": {"type": "number"},
         "order_type": {"type": "string", "enum": ["market", "limit"]}, "limit_price": {"type": "number"},
         "rationale": {"type": "string"}, "confidence": {"type": "number"}, "client_id": {"type": "string"}},
         "required": ["symbol", "side", "quantity", "rationale"]}},
    {"name": "get_verdict", "description": "Получить вердикт по идентификатору.",
     "inputSchema": {"type": "object", "properties": {"verdict_id": {"type": "string"}}, "required": ["verdict_id"]}},
]


class Client:
    def __init__(self, url: str, key: str) -> None:
        self.url, self.key = url.rstrip("/"), key

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(self.url + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read())
            except Exception:
                return {"error": {"message": f"HTTP {e.code}"}}

    def tool(self, name: str, args: dict) -> dict:
        if name == "get_context":
            return self.call("GET", "/gateway/v1/context?symbol=" + urllib.parse.quote(str(args.get("symbol", ""))))
        if name == "get_limits":
            return self.call("GET", "/gateway/v1/limits")
        if name == "propose_order":
            return self.call("POST", "/gateway/v1/propose", args)
        if name == "get_verdict":
            return self.call("GET", "/gateway/v1/verdicts/" + urllib.parse.quote(str(args.get("verdict_id", ""))))
        return {"error": {"message": f"неизвестный инструмент {name}"}}


def handle(client: Client, msg: dict) -> dict | None:
    mid, method = msg.get("id"), msg.get("method")
    if mid is None:
        return None
    if method == "initialize":
        res = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
               "capabilities": {"tools": {}}, "serverInfo": {"name": "finvex-gateway", "version": "1.0"}}
    elif method == "tools/list":
        res = {"tools": TOOLS}
    elif method == "tools/call":
        p = msg.get("params", {})
        out = client.tool(p.get("name", ""), p.get("arguments") or {})
        res = {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}], "isError": "error" in out}
    elif method == "ping":
        res = {}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "метод не поддерживается"}}
    return {"jsonrpc": "2.0", "id": mid, "result": res}


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--key", required=True)
    a = ap.parse_args()
    client = Client(a.url, a.key)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            out = handle(client, json.loads(line))
        except Exception as exc:
            out = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(exc)[:200]}}
        if out is not None:
            sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
