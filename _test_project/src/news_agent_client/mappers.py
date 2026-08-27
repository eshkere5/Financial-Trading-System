"""Преобразование raw HTTP-ответа news-агента в NewsDoc."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

from src.news_agent_client.client import (
    NewsDoc,
    _compute_sanction_score,
    _hybrid_sanction_score,
)

logger = logging.getLogger(__name__)


def _as_list(raw: Any) -> List[str]:
    """Поле может прийти списком или CSV-строкой — приводим к списку."""
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def _as_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


### FIXED 2026-08-01 (К-1) ###
# Было: NewsDoc(...) собирался из полей `ticker` и `relevance`, которых в
# dataclass NewsDoc нет, и без обязательных item_id/url — любой вызов
# гарантированно падал с TypeError. Функция была нерабочей с момента
# написания (HTTP-транспорт её не звал, у него была своя копия маппинга).
# Стало: маппинг на реальные поля NewsDoc; это единственная реализация —
# client._http_search теперь делегирует сюда, дубликат удалён.
# Дополнительно: `data` принимается и как dict {"items": [...]}, и как
# готовый список, чтобы не ломать обоих возможных вызывающих.
def map_http_response(data: Any) -> List[NewsDoc]:
    if isinstance(data, dict):
        items: Iterable[Dict[str, Any]] = data.get("items", []) or []
    elif isinstance(data, list):
        items = data
    else:
        logger.warning("map_http_response: неожиданный тип ответа %s", type(data).__name__)
        return []

    docs: List[NewsDoc] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        item_id = str(item.get("item_id") or item.get("id") or "")
        if not item_id:
            logger.warning("map_http_response: пропущен элемент без item_id: %.120s", item)
            continue

        title = str(item.get("title", "") or "")
        body = str(item.get("text_clean") or item.get("text") or item.get("body") or "")

        docs.append(NewsDoc(
            item_id=item_id,
            title=title,
            body=body,
            source=str(item.get("source", "") or ""),
            published_at=str(item.get("date") or item.get("published_at") or ""),
            url=str(item.get("url", "") or ""),
            topics=_as_list(item.get("topics")),
            tickers=[t.upper() for t in _as_list(item.get("tickers"))],
            ner_companies=_as_list(item.get("ner_companies")),
            ner_orgs=_as_list(item.get("ner_orgs")),
            sentiment_label=str(item.get("sentiment_label", "neutral") or "neutral"),
            sentiment=_as_float(item.get("sentiment_score", item.get("sentiment", 0.0))),
            sanction_score=_hybrid_sanction_score(
                _compute_sanction_score(title, body),
                _as_float(item.get("sanction_risk")),
            ),
            summary=str(item.get("summary", "") or ""),
        ))
    return docs
