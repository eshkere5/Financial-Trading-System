"""Bybit, публичные рыночные данные (REST v5). Ключи и pybit не нужны, заявки не принимаются.

Только чтение: свечи и список инструментов. Отдельный адаптер от `client.py`, который нужен движку для торговли.
Для тестовой сети укажите `base_url: https://api-testnet.bybit.com` (цены там отличаются от реального рынка).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from finvex.core.base import Capability, MarketDataProvider, ProviderInfo
from finvex.core.errors import DataFormatError, ProviderError
from finvex.core.models import AssetClass, Candle, CandleSeries, Instrument, InstrumentId, Timeframe
from finvex.core.quality import check_candles
from finvex.core.registry import register
from finvex.core.transport import HttpxTransport, JsonTransport

VENUE = "BYBIT"
BASE_URL = "https://api.bybit.com"
_INTERVAL = {Timeframe.M1: "1", Timeframe.M5: "5", Timeframe.M15: "15", Timeframe.M30: "30", Timeframe.H1: "60",
             Timeframe.H4: "240", Timeframe.D1: "D", Timeframe.W1: "W"}
_PAGE = 1000   # максимум свечей в одном ответе Bybit
CATEGORIES = ("spot", "linear", "inverse")


@register("bybit_public", kind="market_data")
class BybitPublicProvider(MarketDataProvider):
    def __init__(self, category: str = "spot", base_url: str = BASE_URL, transport: Optional[JsonTransport] = None,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc), **transport_opts: Any) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"category: одно из {CATEGORIES}")
        self.category, self._base, self._now = category, base_url.rstrip("/"), now
        self._tr = transport or HttpxTransport(**transport_opts)
        self._info = ProviderInfo(
            name=f"bybit_public:{category}", venue=VENUE,
            capabilities=Capability.INSTRUMENTS | Capability.CANDLES, timeframes=frozenset(_INTERVAL),
            license_note="Публичные данные Bybit, ключ не нужен. Лимиты запросов действуют на IP; из РФ нужен VPN.")

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def close(self) -> None:
        self._tr.close()

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        data = self._tr.get_json(f"{self._base}{path}", params)
        if not isinstance(data, dict) or data.get("retCode") != 0:
            raise ProviderError(f"Bybit: {data.get('retMsg') if isinstance(data, dict) else 'неожиданный ответ'}")
        return data.get("result") or {}

    def get_instruments(self) -> Sequence[Instrument]:
        rows = self._get("/v5/market/instruments-info", {"category": self.category, "limit": 1000}).get("list") or []
        return [Instrument(id=InstrumentId(r["symbol"], VENUE), asset_class=AssetClass.CRYPTO, name=r.get("baseCoin", ""),
                           currency=r.get("quoteCoin")) for r in rows if r.get("symbol")]

    def get_candles(self, instrument: "InstrumentId | str", timeframe: "Timeframe | str", start: datetime,
                    end: Optional[datetime] = None, *, only_closed: bool = True, max_pages: int = 20) -> CandleSeries:
        tf = Timeframe.parse(timeframe)
        self.require_timeframe(tf)
        iid = InstrumentId.parse(instrument, default_venue=VENUE)
        if iid.venue != VENUE:
            raise ProviderError(f"{self.info.name} обслуживает {VENUE}, получено {iid}")
        now = self._now()
        end = end or now
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start/end должны быть tz-aware")
        if end <= start:
            raise ValueError("end должен быть позже start")
        step = timedelta(seconds=tf.seconds)
        cursor_ms, start_ms, rows = int(end.timestamp() * 1000), int(start.timestamp() * 1000), {}
        for _ in range(max_pages):   # Bybit отдаёт страницы от новых к старым
            page = self._get("/v5/market/kline", {"category": self.category, "symbol": iid.symbol, "interval": _INTERVAL[tf],
                                                  "start": start_ms, "end": cursor_ms, "limit": _PAGE}).get("list") or []
            if not page:
                break
            for r in page:
                rows[int(r[0])] = r
            oldest = min(int(r[0]) for r in page)
            if len(page) < _PAGE or oldest <= start_ms:
                break
            cursor_ms = oldest - 1
        else:
            raise ProviderError(f"{iid.symbol}: слишком много страниц свечей")
        candles: List[Candle] = []
        for ms in sorted(rows):
            r = rows[ms]
            try:
                t0 = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
                candles.append(Candle(ts_open=t0, ts_close=t0 + step, open=float(r[1]), high=float(r[2]), low=float(r[3]),
                                      close=float(r[4]), volume=float(r[5]), turnover=float(r[6]) if len(r) > 6 else None))
            except (ValueError, IndexError) as exc:
                raise DataFormatError(f"Свеча {iid.symbol} в неожиданном формате: {r}") from exc
        if only_closed:
            candles = [c for c in candles if c.ts_close <= now]
        candles = [c for c in candles if start <= c.ts_open <= end]
        return CandleSeries(instrument=iid, timeframe=tf, candles=tuple(candles), source=self.info.name, fetched_at=now,
                            quality=check_candles(candles, tf, now=now))
