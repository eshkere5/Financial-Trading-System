"""HTTP-транспорт. Отделён от провайдеров, чтобы их можно было тестировать без сети."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Mapping, Optional, Protocol

from finvex.core.errors import DataFormatError, ProviderError

logger = logging.getLogger(__name__)


class JsonTransport(Protocol):
    def get_json(self, url: str, params: Optional[Mapping[str, Any]] = None) -> Any: ...
    def close(self) -> None: ...


class HttpxTransport:
    """httpx-клиент с таймаутом, повторами на 429/5xx и минимальным интервалом между запросами."""

    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        timeout: float = 20.0,
        retries: int = 3,
        backoff: float = 0.7,
        min_interval: float = 0.05,
        user_agent: str = "finvex-market-adapters/0.1",
    ) -> None:
        import httpx  # ленивый импорт: модуль нужен только для сети
        self._httpx = httpx
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": user_agent})
        self._retries, self._backoff, self._min_interval = retries, backoff, min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def get_json(self, url: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(self._retries + 1):
            self._throttle()
            try:
                resp = self._client.get(url, params=params)
            except self._httpx.HTTPError as exc:
                last_err = exc
            else:
                if resp.status_code in self.RETRY_STATUS:
                    last_err = ProviderError(f"HTTP {resp.status_code} от {url}")
                elif resp.status_code >= 400:
                    raise ProviderError(f"HTTP {resp.status_code} от {url}")
                else:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise DataFormatError(f"Ответ {url} не является JSON") from exc
            if attempt < self._retries:
                time.sleep(self._backoff * (2 ** attempt))
        raise ProviderError(f"Не удалось получить {url}: {last_err}") from last_err

    def close(self) -> None:
        self._client.close()
