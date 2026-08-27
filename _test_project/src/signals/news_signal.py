"""
NewsSignalExtractor — мост между NewsAgentClient и торговыми сигналами.

Отвечает за:
  1. Получение новостей через NewsAgentClient (local SQLite или HTTP)
  2. Агрегацию сентимента / sanction_risk в NewsSnapshot (адаптер из client.NewsSnapshot)
  3. Создание NewsSnapshot и прокидывание в MarketState.news[ticker]
  4. Опциональный вес новостного сигнала при комбинировании (combine_with_kronos)

Не зависит от стратегии — чистый data layer.

### ARCHITECTURE_REVIEW FIX (2026-07-09) ###
Исходная версия _build_snapshot() вызывала client.search_news(ticker) и обращалась
к `d.relevance` на каждом NewsDoc. Но news_agent_client/client.py (актуальная схема,
отражающая реальную таблицу SQLite новостного блока) не имеет поля `relevance` в
NewsDoc вовсе (такого понятия нет в таблице news, см. news_db.py) — это вызывало
AttributeError на каждой новости и ломало 7 тестов в tests/test_news_signal.py.
Исправлено: _build_snapshot() теперь вызывает client.get_snapshot(ticker, hours_back),
который возвращает готовый агрегат (avg_sentiment, sanction_risk, positive_ratio,
negative_ratio, doc_count, signal_weight), и мапит его на market_state.NewsSnapshot
(sentiment=avg_sentiment, relevance — вычисляется как доля нейтральных новостей от общего
числа, headline_count=doc_count). market_state.py, strategy.py, signal_types.py и тесты
оставлены без изменений (они работают с market_state.NewsSnapshot, не с client.NewsSnapshot).
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Optional

from src.engine.market_state import MarketState, NewsSnapshot
from src.news_agent_client.client import NewsAgentClient, NewsDoc

logger = logging.getLogger(__name__)


class NewsSignalExtractor:
    """
    Обновляет MarketState.news[ticker] для заданного списка тикеров.

    Пример использования:
        extractor = NewsSignalExtractor(news_client, hours_back=6)
        extractor.refresh(state, tickers=["GAZP", "LKOH"])
        snapshot = state.get_news("GAZP")
        weight = snapshot.signal_weight  # [-1, +1]
    """

    def __init__(
        self,
        client: NewsAgentClient,
        hours_back: int = 6,
        min_relevance: float = 0.3,
        llm_summarize: bool = False,
    ) -> None:
        self._client = client
        self.hours_back = hours_back
        self.min_relevance = min_relevance
        self.llm_summarize = llm_summarize

    # ── публичный API ─────────────────────────────────────────────────────

    def refresh(self, state: MarketState, tickers: List[str]) -> None:
        """
        Обновляет state.news для каждого тикера.
        Вызывается из LiveEngine или перед on_bar в бэктесте.
        """
        for ticker in tickers:
            try:
                snapshot = self._build_snapshot(ticker)
                state.news[ticker] = snapshot
                logger.debug(
                    "News[%s]: sentiment=%.2f relevance=%.2f risk=%.2f count=%d",
                    ticker, snapshot.sentiment, snapshot.relevance,
                    snapshot.sanction_risk, snapshot.headline_count,
                )
            except Exception as exc:
                logger.warning("NewsSignalExtractor.refresh failed for %s: %s", ticker, exc)

    def get_snapshot(self, ticker: str) -> Optional[NewsSnapshot]:
        """Получить snapshot без записи в state (для разового запроса)."""
        try:
            return self._build_snapshot(ticker)
        except Exception as exc:
            logger.warning("get_snapshot failed for %s: %s", ticker, exc)
            return None

    # ── внутренняя логика ─────────────────────────────────────────────────

    @staticmethod
    def _empty(ticker: str) -> NewsSnapshot:
        return NewsSnapshot(
            ticker=ticker,
            sentiment=0.0,
            relevance=0.0,
            sanction_risk=0.0,
            headline_count=0,
        )

    def _build_snapshot(self, ticker: str) -> NewsSnapshot:
        """
        Строит market_state.NewsSnapshot из агрегата NewsAgentClient.get_snapshot().

        Маппинг полей client.NewsSnapshot -> market_state.NewsSnapshot:
          avg_sentiment          -> sentiment
          1 - |positive-negative|-> relevance (доля новостей, не нейтральных к общему шуму
                                    заменена суррогатом: чем выше расхождение positive/negative
                                    ratio, тем выше "релевантность" сигнала для тикера)
          sanction_risk          -> sanction_risk (без изменений)
          doc_count              -> headline_count
        """
        client_snap = self._client.get_snapshot(ticker, hours_back=self.hours_back)

        if client_snap.doc_count == 0:
            return self._empty(ticker)

        # relevance — суррогатная метрика на основе доли новостей с явной
        # (не нейтральной) сентимент-меткой; чем выше доля positive+negative,
        # тем выше уверенность, что новости релевантны и несут сигнал.
        relevance = round(min(1.0, client_snap.positive_ratio + client_snap.negative_ratio), 4)

        ### FIXED 2026-08-01 (С-12) ###
        # Было: self.min_relevance принимался в __init__, сохранялся и нигде
        # не использовался. Порог из конфига (по умолчанию 0.3) молча не
        # работал: снапшот из одних нейтральных новостей (relevance == 0.0)
        # всё равно уезжал в MarketState и влиял на сайзинг.
        # Стало: ниже порога отдаём пустой снапшот — стратегия видит
        # "новостей нет", а не шум. Санкционный риск при этом не теряется:
        # если sanction_risk высокий, снапшот пропускается несмотря на
        # низкую relevance (глушить сигнал о санкциях нельзя).
        if relevance < self.min_relevance and client_snap.sanction_risk < 0.5:
            logger.debug(
                "News[%s]: relevance=%.2f < min_relevance=%.2f — снапшот отброшен (%d новостей)",
                ticker, relevance, self.min_relevance, client_snap.doc_count,
            )
            return self._empty(ticker)

        # LLM-саммари опционально
        summary = ""
        if self.llm_summarize and hasattr(self._client, "summarize_events"):
            try:
                summary = self._client.summarize_events(ticker, hours_back=self.hours_back)
            except Exception as exc:
                logger.warning("summarize_events failed for %s: %s", ticker, exc)
        elif client_snap.docs:
            summary = " | ".join(d.title for d in client_snap.docs[:5])

        return NewsSnapshot(
            ticker=ticker,
            sentiment=round(client_snap.avg_sentiment, 4),
            relevance=relevance,
            sanction_risk=round(client_snap.sanction_risk, 4),
            headline_count=client_snap.doc_count,
            updated_at=datetime.utcnow(),
            summary=summary,
        )

    # ── статический хелпер для комбинирования сигналов ───────────────────

    @staticmethod
    def combine_weights(
        kronos_signal: float,
        news_weight: float,
        news_alpha: float = 0.3,
    ) -> float:
        """
        Линейная комбинация Kronos-сигнала и новостного веса.

        Args:
            kronos_signal: направленный сигнал от Kronos [-1, +1]
            news_weight:   NewsSnapshot.signal_weight [-1, +1]
            news_alpha:    доля новостного сигнала (0 = только Kronos)

        Returns:
            Итоговый комбинированный сигнал [-1, +1]
        """
        combined = (1 - news_alpha) * kronos_signal + news_alpha * news_weight
        return float(max(-1.0, min(1.0, combined)))
