"""
Московская биржа, ISS (https://iss.moex.com/iss) — рыночные данные и справочники.

Это только данные: ISS не принимает заявки. Структура по схеме OpenBB
«Transform → Extract → Transform»: параметры приводятся к виду ISS, ответ
забирается через JsonTransport, строки блоков переводятся в модели Finvex.

Особенности ISS, которые учтены:
- время в ответах — московское, без указания пояса;
- свечи отдаются страницами (по ~500 строк), листаем параметром `start`;
- параметр `limit` у сделок принимает только 1/10/100/1000/5000;
- родные интервалы свечей: 1, 10, 60, 24 (день), 7, 31, 4 — остальные
  таймфреймы провайдер не имитирует и честно отклоняет;
- последняя свеча может быть незавершённой — по умолчанию отбрасывается.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from finvex.core.base import Capability, MarketDataProvider, ProviderInfo
from finvex.core.errors import DataFormatError, ProviderError
from finvex.core.models import (
    AssetClass, Candle, CandleSeries, Instrument, InstrumentId, MSK, Quote, Side,
    Timeframe, Trade,
)
from finvex.core.quality import check_candles
from finvex.core.registry import register
from finvex.core.transport import HttpxTransport, JsonTransport

BASE_URL = "https://iss.moex.com/iss"
VENUE = "MOEX"

_INTERVAL = {
    Timeframe.M1: 1, Timeframe.M10: 10, Timeframe.H1: 60,
    Timeframe.D1: 24, Timeframe.W1: 7, Timeframe.MN1: 31,
}
_TRADE_LIMITS = (1, 10, 100, 1000, 5000)
_ASSET = {"shares": AssetClass.EQUITY, "bonds": AssetClass.BOND, "index": AssetClass.INDEX,
          "selt": AssetClass.CURRENCY, "forts": AssetClass.FUTURE}


def _rows(payload: Mapping[str, Any], block: str) -> List[Dict[str, Any]]:
    """Блок ISS {columns:[...], data:[[...]]} → список словарей с ключами в нижнем регистре."""
    try:
        blk = payload[block]
        cols = [c.lower() for c in blk["columns"]]
        return [dict(zip(cols, row)) for row in blk["data"]]
    except (KeyError, TypeError) as exc:
        raise DataFormatError(f"В ответе ISS нет блока {block!r}") from exc


def _msk(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=MSK).astimezone(timezone.utc)
    except ValueError as exc:
        raise DataFormatError(f"Неверный формат времени ISS: {text!r}") from exc


def _last_trade_ts(row: Mapping[str, Any]) -> Optional[datetime]:
    """В marketdata TIME — только ЧЧ:ММ:СС по Москве; дату берём из SYSTIME."""
    t, sys_ = row.get("time"), row.get("systime")
    if not t or not sys_:
        return None
    try:
        day = datetime.strptime(sys_, "%Y-%m-%d %H:%M:%S").date()
        clock = datetime.strptime(t, "%H:%M:%S").time()
    except ValueError:
        return None
    ts = datetime.combine(day, clock, tzinfo=MSK).astimezone(timezone.utc)
    snap = _msk(sys_)
    if snap is not None and ts > snap:  # «сделка из будущего» → это вчерашняя
        ts -= timedelta(days=1)
    return ts


def _f(v: Any) -> Optional[float]:
    return None if v is None else float(v)


@register("moex_iss", kind="market_data")
class MoexISSProvider(MarketDataProvider):
    def __init__(
        self,
        engine: str = "stock",
        market: str = "shares",
        board: str = "TQBR",
        base_url: str = BASE_URL,
        transport: Optional[JsonTransport] = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        **transport_opts: Any,
    ) -> None:
        self.engine, self.market, self.board = engine, market, board
        self._base = base_url.rstrip("/")
        self._tr = transport or HttpxTransport(**transport_opts)
        self._now = now
        self._info = ProviderInfo(
            name=f"moex_iss:{market}:{board}",
            venue=VENUE,
            capabilities=Capability.INSTRUMENTS | Capability.CANDLES | Capability.QUOTES | Capability.TRADES,
            timeframes=frozenset(_INTERVAL),
            delayed=True,
            license_note=(
                "Бесплатные данные ISS публикуются с задержкой и предназначены для ознакомления; "
                "для коммерческого использования и данных реального времени нужен договор с биржей."
            ),
        )

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def close(self) -> None:
        self._tr.close()

    # ── URL ──────────────────────────────────────────────────────────────
    def _board_url(self, tail: str) -> str:
        return f"{self._base}/engines/{self.engine}/markets/{self.market}/boards/{self.board}/{tail}"

    def _sym(self, instrument: "InstrumentId | str") -> str:
        iid = InstrumentId.parse(instrument, default_venue=VENUE)
        if iid.venue != VENUE:
            raise ProviderError(f"{self.info.name} обслуживает {VENUE}, получено {iid}")
        return iid.symbol

    # ── инструменты ──────────────────────────────────────────────────────
    def get_instruments(self) -> Sequence[Instrument]:
        payload = self._tr.get_json(
            self._board_url("securities.json"),
            {"iss.meta": "off", "iss.only": "securities"},
        )
        asset = _ASSET.get(self.market, AssetClass.OTHER)
        out = []
        for r in _rows(payload, "securities"):
            if not r.get("secid"):
                continue
            out.append(Instrument(
                id=InstrumentId(r["secid"], VENUE),
                asset_class=asset,
                name=r.get("shortname") or r.get("name") or r.get("secname") or "",
                isin=r.get("isin"),
                currency=r.get("currencyid") or r.get("faceunit"),
                lot_size=int(r.get("lotsize") or 1),
                tick_size=_f(r.get("minstep")),
                price_precision=None if r.get("decimals") is None else int(r["decimals"]),
                board=r.get("boardid") or self.board,
            ))
        return out

    # ── свечи ────────────────────────────────────────────────────────────
    def get_candles(
        self,
        instrument: "InstrumentId | str",
        timeframe: "Timeframe | str",
        start: datetime,
        end: Optional[datetime] = None,
        *,
        only_closed: bool = True,
        max_pages: int = 200,
    ) -> CandleSeries:
        tf = Timeframe.parse(timeframe)
        self.require_timeframe(tf)
        sym = self._sym(instrument)
        now = self._now()
        end = end or now
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start/end должны быть tz-aware (UTC или MSK)")
        if end <= start:
            raise ValueError("end должен быть позже start")

        fmt = "%Y-%m-%d %H:%M:%S"
        params: Dict[str, Any] = {
            "iss.meta": "off",
            "from": start.astimezone(MSK).strftime(fmt),
            "till": end.astimezone(MSK).strftime(fmt),
            "interval": _INTERVAL[tf],
        }
        url = self._board_url(f"securities/{sym}/candles.json")
        candles: List[Candle] = []
        offset = 0
        for _ in range(max_pages):
            page = _rows(self._tr.get_json(url, {**params, "start": offset}), "candles")
            if not page:
                break
            for r in page:
                t_open, t_close = _msk(r["begin"]), _msk(r["end"])
                if t_open is None or t_close is None:
                    raise DataFormatError(f"Свеча {sym} без begin/end: {r}")
                candles.append(Candle(
                    ts_open=t_open, ts_close=t_close,
                    open=float(r["open"]), high=float(r["high"]),
                    low=float(r["low"]), close=float(r["close"]),
                    volume=float(r["volume"] or 0.0), turnover=_f(r.get("value")),
                ))
            offset += len(page)
        else:
            raise ProviderError(f"{sym}: превышено число страниц ({max_pages}) при загрузке свечей")

        if only_closed:  # не отдаём формирующийся бар — иначе возможен lookahead
            candles = [c for c in candles if c.ts_close <= now]
        candles = [c for c in candles if start <= c.ts_open <= end]
        iid = InstrumentId(sym, VENUE)
        return CandleSeries(
            instrument=iid, timeframe=tf, candles=tuple(candles),
            source=self.info.name, fetched_at=now,
            quality=check_candles(candles, tf, now=now),
        )

    # ── котировки ────────────────────────────────────────────────────────
    def get_quotes(self, instruments: Optional[Sequence["InstrumentId | str"]] = None) -> Sequence[Quote]:
        params: Dict[str, Any] = {"iss.meta": "off", "iss.only": "marketdata"}
        if instruments:
            params["securities"] = ",".join(self._sym(i) for i in instruments)
        payload = self._tr.get_json(self._board_url("securities.json"), params)
        received = self._now()
        return [
            Quote(
                instrument=InstrumentId(r["secid"], VENUE),
                bid=_f(r.get("bid")), ask=_f(r.get("offer")), last=_f(r.get("last")),
                volume_today=_f(r.get("voltoday")),
                trading_status=r.get("tradingstatus"),
                ts_snapshot=_msk(r.get("systime")), ts_received=received,
                ts_last_trade=_last_trade_ts(r),
            )
            for r in _rows(payload, "marketdata") if r.get("secid")
        ]

    # ── сделки ───────────────────────────────────────────────────────────
    def get_trades(self, instrument: "InstrumentId | str", limit: int = 100) -> Sequence[Trade]:
        if limit < 1:
            raise ValueError("limit должен быть >= 1")
        snap = next((x for x in _TRADE_LIMITS if x >= limit), _TRADE_LIMITS[-1])
        sym = self._sym(instrument)
        payload = self._tr.get_json(
            self._board_url(f"securities/{sym}/trades.json"),
            {"iss.meta": "off", "iss.only": "trades", "limit": snap, "reversed": 1},
        )
        rows = _rows(payload, "trades")[:limit]
        iid = InstrumentId(sym, VENUE)
        side = {"B": Side.BUY, "S": Side.SELL}
        trades = [
            Trade(
                instrument=iid, trade_id=str(int(r["tradeno"])),
                price=float(r["price"]), quantity=float(r["quantity"]),
                side=side.get(r.get("buysell")), ts_event=_msk(r["systime"]),  # type: ignore[arg-type]
            )
            for r in rows
        ]
        return sorted(trades, key=lambda t: t.ts_event)
