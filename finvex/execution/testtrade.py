"""Отправка разрешённого шлюзом вердикта в тестовый контур. Боевое исполнение шлюза не затрагивается."""
from __future__ import annotations

from finvex.core.models import InstrumentId, OrderRequest, OrderType, Side

from .testnet import TestVenueError, TestVenues


def send_verdict(venues: TestVenues, verdict: dict) -> dict:
    """Возвращает {sent, venue, status, order_id, message}. Отправляет только вердикт allow."""
    if verdict.get("decision") != "allow":
        return {"sent": False, "message": "вердикт не разрешает заявку: в тестовый контур не отправлено"}
    i = verdict["intent"]
    venue = venues.venue_for(i["symbol"])
    try:
        ex = venues.get(venue)
    except TestVenueError as exc:
        return {"sent": False, "venue": venue, "message": str(exc)}
    try:
        req = OrderRequest(plan_id=verdict["id"], client_order_id=verdict["id"], instrument=InstrumentId(i["symbol"], venue),
                           side=Side(i["side"]), quantity=float(i["quantity"]), order_type=OrderType(i["order_type"]),
                           limit_price=i.get("limit_price"))
        ack = ex.submit(req)
    except Exception as exc:
        return {"sent": False, "venue": venue, "message": f"контур вернул ошибку: {type(exc).__name__}: {str(exc)[:200]}"}
    ok = ack.status == "accepted"
    return {"sent": ok, "venue": venue, "status": ack.status, "order_id": ack.venue_order_id,
            "message": "заявка принята тестовым контуром" if ok else ack.status}
