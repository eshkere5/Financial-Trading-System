"""
LiveEngine — async event loop для sandbox/prod торговли.

"""
from __future__ import annotations

import asyncio
import inspect
import logging
import signal
import time
from typing import Any, Dict, List, Optional

from src.engine.config_loader import EngineConfig
from src.engine.market_state import MarketState, Position
from src.engine.portfolio import Portfolio
from src.engine.risk_manager import RiskManager, RiskConfig
from src.engine.orders import Order, OrderStatus
from src.utils.time_utils import utcnow

logger = logging.getLogger(__name__)

PRICE_POLL_INTERVAL = 60
NEWS_POLL_INTERVAL = 300
RISK_CHECK_INTERVAL = 30
STRATEGIST_POLL_INTERVAL = 900
RECONCILE_INTERVAL_TICKS = 10  # каждые N тиков price_loop

# ### FIXED 2026-08-02 (Fix A) ###
# Сколько держим ордер в "ожидании исполнения", прежде чем перестать его
# отслеживать. Дальше судьбу позиции определяет сверка с брокером
# (_reconcile_state) — она остаётся страховочным механизмом.
PENDING_ORDER_TTL_SEC = 900

# Суффиксы/паттерны, по которым тикер распознаётся как крипто-инструмент.
# Пример: BTCUSDT, ETHUSDT — торгуются на Bybit, а не на Tinkoff.
CRYPTO_SUFFIXES = ("USDT", "USDC", "BTC", "ETH")


def _guess_broker(ticker: str) -> str:
    """
    Определяет, какому брокеру принадлежит тикер.
    Простая героистика: крипто-пары оканчиваются на USDT/USDC/BTC/ETH.
    Всё остальное считается акцией и уходит на Tinkoff.
    Если понадобится точный маппинг — заменить на explicit dict из engine.yaml.
    """
    upper = ticker.upper()
    if any(upper.endswith(suf) for suf in CRYPTO_SUFFIXES):
        return "bybit"
    return "tinkoff"


# ### FIXED 2026-08-02 (Fix A) ###
def _status_str(status: Any) -> str:
    """
    Нормализует статус ордера к нижнему регистру строки.

    OrderStatus (engine/orders.py) на данный момент НЕ содержит члена
    ACCEPTED — новый контракт роутеров вводит его как строковое значение
    "accepted". Сравнение через строку работает в обоих случаях и не требует
    менять orders.py (он вне периметра этой правки).
    """
    if status is None:
        return ""
    value = getattr(status, "value", status)
    return str(value).strip().lower()


class LiveEngine:
    """
    Пример запуска (один брокер, обратная совместимость):
        engine = LiveEngine(cfg, token=token)
        engine.register_strategy(strategy)
        engine.run()

    Пример запуска (мультибиржевой режим):
        engine = LiveEngine(cfg, token=tinkoff_token, enable_bybit=True)
        engine.register_strategy(ai_tactic_strategy)
        engine.run()
    """

    def __init__(
        self,
        cfg: EngineConfig,
        mode: str = "sandbox",
        token: str = "",
        news_pipeline=None,
        strategist=None,
        price_poll_interval: int = PRICE_POLL_INTERVAL,
        news_poll_interval: int = NEWS_POLL_INTERVAL,
        strategist_poll_interval: int = STRATEGIST_POLL_INTERVAL,
        enable_bybit: bool = False,
        bybit_category: str = "spot",
        universe_selector=None,
    ) -> None:
        self.cfg = cfg
        self.mode = mode
        self._token = token
        self._news_pipeline = news_pipeline
        if self._news_pipeline is not None:
            self._news_pipeline.on_assessment = self._on_news_assessment
        self._strategist = strategist
        self._price_poll_interval = price_poll_interval
        self._news_poll_interval = news_poll_interval
        self._strategist_poll_interval = strategist_poll_interval
        self._decision = None

        self._enable_bybit = enable_bybit
        self._bybit_category = bybit_category
        self._universe_selector = universe_selector

        self._baseline_synced = False
        self._state = MarketState(cash=cfg.initial_capital, timestamp=utcnow())
        self._portfolio = Portfolio(self._state, cfg.commission)

        risk_cfg: RiskConfig = getattr(cfg, "risk", RiskConfig())
        self._risk = RiskManager(risk_cfg)
        self._risk.reset_peak(cfg.initial_capital)

        self._strategies: list = []
        self._running = False

        # Мультибиржевой роутинг: {"tinkoff": router, "bybit": router}
        self._routers: Dict[str, object] = {}

        # ### FIXED 2026-08-02 (Fix A) ###
        # Ордера, принятые брокером, но с неизвестным исполнением.
        # {broker_order_id: {"order": Order, "since": monotonic_ts}}
        self._pending_orders: Dict[str, Dict[str, Any]] = {}

    def register_strategy(self, strategy) -> None:
        self._strategies.append(strategy)

    def set_universe_selector(self, selector) -> None:
        self._universe_selector = selector

    # ── точка входа (sync wrapper) ────────────────────────────────────────

    def run(self) -> None:
        """Синхронная точка входа — запускает async loop."""
        asyncio.run(self._run_async())

    # ── async loop ────────────────────────────────────────────────────────

    async def _run_async(self) -> None:
        await self._init_routers()
        self._running = True

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown)
            except NotImplementedError:
                signal.signal(
                    sig,
                    lambda signum, frame: loop.call_soon_threadsafe(self._shutdown),
                )

        logger.info(
            "LiveEngine started | mode=%s | brokers=%s | price_interval=%ds | news_interval=%ds",
            self.mode, list(self._routers.keys()), self._price_poll_interval, self._news_poll_interval,
        )

        # Crash recovery: восстановить реальное состояние со всех брокеров
        await self._reconcile_state()

        if self._news_pipeline is not None:
            # ── сидируем watch_terms ОДНОКРАТНО (при старте, не по таймеру) ──
            # Это позволяет парсеру новостей работать сразу, без ожидания
            # первого strategist-цикла (LLM может отсутствовать в дефолтном режиме).
            from src.engine.kronos_default_strategy import KronosDefaultStrategy
            for strategy in self._strategies:
                if isinstance(strategy, KronosDefaultStrategy):
                    seeded = strategy.seed_watch_terms()
                    if seeded:
                        self._news_pipeline.update_watch_terms(seeded, "seed@startup")
                        logger.info("watch_terms засеяны из KronosDefaultStrategy для %d тикеров", len(seeded))

            # ── холодный старт "не в вакууме" ──
            # Перед первым price-тиком собираем RSS-бэклог за COLD_START_BACKLOG_DAYS
            # дней, малая e5-модель отбирает top-5 самых релевантных новостей ПО
            # ТИКЕРУ (не весь бэклог — дорого), и только они уходят в DeepSeek.
            # Идея: график Kronos уже отражает прошлое поведение рынка, поэтому
            # эта корректировка весов заведомо небольшая — это "мягкая поправка"
            # перед стартом, а не замена анализа Kronos.
            await self._cold_start_news_backfill()

            self._news_pipeline.start()
            logger.info("NewsRiskPipeline запущен в фоновом потоке (событийная обработка RSS)")

            # восстановленный из news.db риск-фон — сразу в state, не ждём свежих новостей
            tickers = list({t for s in self._strategies for t in getattr(s, "tickers", [])})
            if tickers:
                self._news_pipeline.publish_restored(tickers)

        tasks = [
            asyncio.create_task(self._price_loop(), name="price_loop"),
            asyncio.create_task(self._strategist_loop(), name="strategist_loop"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            if self._pending_orders:
                logger.warning(
                    "[pending] На момент остановки не подтверждено %d ордеров: %s",
                    len(self._pending_orders), list(self._pending_orders.keys()),
                )
            logger.info("LiveEngine stopped | final PnL=%.2f", self._state.total_pnl)
            for name, router in self._routers.items():
                try:
                    close = router.close
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.warning("Router %s close failed: %s", name, exc)

    async def _init_routers(self) -> None:
        if self._token:
            from src.executors.execution_router import ExecutionRouter
            self._routers["tinkoff"] = ExecutionRouter(mode=self.mode, token=self._token)
            logger.info("Tinkoff router initialized | mode=%s", self.mode)
        else:
            logger.info("Tinkoff router skipped (no token) — testing Bybit only")

        if self._enable_bybit:
            try:
                from src.executors.bybit_execution_router import BybitExecutionRouter
                testnet = self.mode != "live"
                self._routers["bybit"] = await BybitExecutionRouter.create(
                    category=self._bybit_category, testnet=testnet,
                )
                logger.info("Bybit router initialized | testnet=%s category=%s", testnet, self._bybit_category)
            except Exception as exc:
                logger.error("Bybit router init failed — crypto trading disabled: %s", exc)

    def _router_for(self, ticker: str):
        """Выбирает нужного брокера по тикеру. Fallback — tinkoff."""
        broker_name = _guess_broker(ticker)
        router = self._routers.get(broker_name)
        if router is None:
            logger.warning("No router for broker=%s (ticker=%s) — falling back to tinkoff", broker_name, ticker)
            router = self._routers.get("tinkoff")
        return router

    def _shutdown(self) -> None:
        if not self._running:
            return
        logger.info("Shutdown signal received")
        self._running = False
        try:
            for task in asyncio.all_tasks():
                task.cancel()
        except RuntimeError:
            pass

    # ── crash recovery (reconciliation) ────────────────────────────────────

    async def _reconcile_state(self) -> None:
        """
        Синхронизирует self._state.positions/cash с реальным портфелем
        каждого подключённого брокера. Приоритет у данных брокера
        (источник правды). Суммирует cash по всем брокерам.

        ### FIXED 2026-08-02 (Fix B) ###
        Сверка теперь различает "позиции точно нет у брокера" и "мы не смогли
        это проверить". Удаление локальных позиций выполняется ТОЛЬКО когда
        сверка прошла полностью по всем роутерам. Неполной она считается, если:
          - у роутера нет метода get_portfolio (текущий ExecutionRouter такой!);
          - вызов get_portfolio упал;
          - брокер вернул позицию, которую нельзя сопоставить с тикером
            (в ответе только figi/instrument_uid).

        До этой правки любой из трёх случаев приводил к `continue`, пустому
        all_broker_tickers и, как следствие, к удалению ВСЕХ локальных позиций
        из состояния — то есть движок "забывал" реально открытые позиции и
        переставал считать по ним риск.
        """
        loop = asyncio.get_running_loop()
        total_cash = 0.0
        all_broker_tickers: set = set()
        reconcile_complete = True   # ### FIXED 2026-08-02 (Fix B) ###
        queried_brokers = 0

        for name, router in self._routers.items():
            get_portfolio = getattr(router, "get_portfolio", None)
            if get_portfolio is None:
                # ### FIXED 2026-08-02 (Fix B) ###
                logger.warning(
                    "[RECONCILE:%s] Роутер не реализует get_portfolio — сверка по нему невозможна",
                    name,
                )
                reconcile_complete = False
                continue
            try:
                if asyncio.iscoroutinefunction(get_portfolio):
                    broker_portfolio = await asyncio.wait_for(get_portfolio(), timeout=20.0)
                else:
                    broker_portfolio = await asyncio.wait_for(
                        loop.run_in_executor(None, get_portfolio),
                        timeout=20.0,
                    )
            except Exception as exc:
                logger.error("[RECONCILE] %s get_portfolio failed: %s", name, exc)
                continue

            queried_brokers += 1
            raw_positions = broker_portfolio.get("positions", [])
            broker_cash = broker_portfolio.get(
                "cash", broker_portfolio.get("total_amount", 0.0)
            )
            total_cash += float(broker_cash or 0.0)

            for p in raw_positions:
                ticker = p.get("ticker") or p.get("symbol")
                if not ticker:
                    # ### FIXED 2026-08-02 (Fix B) ###
                    # Позиция есть, но идентифицировать её нечем — только figi.
                    # Прагматичное решение: НЕ удалять локальные позиции в этом
                    # цикле сверки, потому что мы не можем доказать, что данная
                    # безымянная позиция не является одной из них.
                    # TODO: маппинг figi→ticker должен жить в
                    # ExecutionRouter.get_portfolio (у TinkoffClient уже есть
                    # get_figi, нужен обратный резолв), а не в движке.
                    logger.warning(
                        "[RECONCILE:%s] Позиция без тикера (figi=%s uid=%s qty=%s) — "
                        "сверка помечена неполной, локальные позиции сохраняются",
                        name, p.get("figi"), p.get("instrument_uid") or p.get("instrumentUid"),
                        p.get("qty"),
                    )
                    reconcile_complete = False
                    continue

                all_broker_tickers.add(ticker)
                if ticker not in self._state.positions:
                    logger.warning(
                        "[RECONCILE:%s] Untracked position on broker: %s qty=%s — restoring",
                        name, ticker, p.get("qty"),
                    )
                self._state.positions[ticker] = Position(
                    ticker=ticker,
                    qty=float(p.get("qty", 0.0)),
                    avg_price=float(p.get("avg_price", 0.0)),
                    open_time=self._state.timestamp,
                )

        if queried_brokers == 0:
            # ### FIXED 2026-08-02 (Fix B) ###
            logger.warning(
                "[RECONCILE] Ни один роутер не отдал портфель — сверка пропущена, "
                "локальное состояние оставлено как есть (%d позиций)",
                len(self._state.positions),
            )
            return

        local_tickers = set(self._state.positions.keys())
        stale = local_tickers - all_broker_tickers
        if stale and not reconcile_complete:
            # ### FIXED 2026-08-02 (Fix B) ###
            logger.warning(
                "[RECONCILE] Сверка неполная — удаление %d локальных позиций пропущено (%s). "
                "Состояние может расходиться с брокером до следующей успешной сверки",
                len(stale), sorted(stale),
            )
        elif stale:
            for ticker in stale:
                logger.warning(
                    "[RECONCILE] Local position %s not found on any broker — dropping (закрыта извне?)",
                    ticker,
                )
                del self._state.positions[ticker]

        # ### FIXED 2026-08-02 (Fix B) ###
        # Cash обновляем только по полной сверке: сумма по части брокеров
        # занижает equity и через portfolio_value искажает drawdown-стоп.
        if reconcile_complete and total_cash > 0:
            self._state.cash = total_cash
        elif total_cash > 0:
            logger.warning(
                "[RECONCILE] Cash не обновлён (получено %.2f по %d брокерам из %d) — сверка неполная",
                total_cash, queried_brokers, len(self._routers),
            )

        logger.info(
            "[RECONCILE] Synced: %d positions, cash=%.2f (brokers=%s, complete=%s)",
            len(self._state.positions), self._state.cash,
            list(self._routers.keys()), reconcile_complete,
        )

        if not self._baseline_synced:
            self._baseline_synced = True
            self._risk.reset_peak(self._state.portfolio_value)
            self._state.pnl_history.clear()
            self._state.pnl_history.append(
                (self._state.timestamp, self._state.portfolio_value)
            )
            if hasattr(self._state, "initial_equity"):
                self._state.initial_equity = self._state.portfolio_value
            logger.info(
                "[RECONCILE] Baseline rebased to broker equity=%.2f",
                self._state.portfolio_value,
            )
    # ── pending orders (Fix A) ─────────────────────────────────────────────

    async def _poll_pending_orders(self) -> None:
        """
        ### FIXED 2026-08-02 (Fix A) ###
        Догоняет исполнение ордеров, которые брокер принял, но по которым
        submit() не смог подтвердить факт сделки.

        Раньше такой ордер просто считался неисполненным: submit() возвращал
        False → _tick() не вызывал portfolio.apply_order() → позиция реально
        открывалась у брокера, но отсутствовала в MarketState. Расхождение
        держалось до ближайшей сверки (раз в 10 тиков), а при отсутствии
        get_portfolio у роутера — навсегда.

        Роутеры, не реализующие get_order_state, пропускаются: для них
        страховкой остаются _expire_pending_orders() + _reconcile_state().
        """
        if not self._pending_orders:
            return

        loop = asyncio.get_running_loop()

        for key, entry in list(self._pending_orders.items()):
            order: Order = entry["order"]
            router = self._router_for(order.ticker)
            if router is None:
                continue

            get_order_state = getattr(router, "get_order_state", None)
            if get_order_state is None:
                continue

            try:
                if asyncio.iscoroutinefunction(get_order_state):
                    state = await get_order_state(order)
                else:
                    state = await loop.run_in_executor(None, get_order_state, order)
            except Exception as exc:
                # Ошибка опроса одного ордера не должна ронять весь цикл
                logger.warning("[pending] get_order_state failed for %s (%s): %s",
                               order.ticker, key, exc)
                continue

            if not state:
                continue

            status = str(state.get("status", "")).strip().lower()

            if status in ("filled", "partially_filled"):
                filled_qty = self._safe_float(state.get("filled_qty"))
                if filled_qty <= 0:
                    if status == "partially_filled":
                        # частичное исполнение нулевого объёма — ещё не сделка
                        continue
                    filled_qty = float(order.qty)

                filled_price = self._safe_float(state.get("filled_price"))
                if filled_price <= 0:
                    filled_price = self._safe_float(
                        order.filled_price if order.filled_price is not None
                        else self._state.prices.get(order.ticker)
                    )
                if filled_price <= 0:
                    logger.warning(
                        "[pending] %s (%s) исполнен, но цена неизвестна — ждём следующий опрос",
                        order.ticker, key,
                    )
                    continue

                order.qty = filled_qty
                order.filled_price = filled_price
                order.status = OrderStatus.FILLED
                order.filled_at = utcnow()

                try:
                    self._portfolio.apply_order(order)
                except Exception as exc:
                    logger.error("[pending] apply_order failed for %s (%s): %s",
                                 order.ticker, key, exc)
                else:
                    logger.info(
                        "[pending] %s подтверждён брокером: status=%s qty=%.6f price=%.4f",
                        order.ticker, status, filled_qty, filled_price,
                    )
                self._pending_orders.pop(key, None)

            elif status in ("cancelled", "canceled", "rejected"):
                order.status = (
                    OrderStatus.REJECTED if status == "rejected" else OrderStatus.CANCELLED
                )
                logger.warning("[pending] %s (%s) завершён без исполнения: status=%s",
                               order.ticker, key, status)
                self._pending_orders.pop(key, None)

            # status == "accepted" → оставляем в ожидании до TTL

    def _expire_pending_orders(self) -> None:
        """
        ### FIXED 2026-08-02 (Fix A) ###
        Снимает с отслеживания ордера, висящие в "accepted" дольше TTL.
        Позиция НЕ применяется к портфелю: если сделка всё же прошла, её
        подхватит _reconcile_state() (источник правды — брокер).
        """
        now = time.monotonic()
        for key, entry in list(self._pending_orders.items()):
            if now - entry["since"] < PENDING_ORDER_TTL_SEC:
                continue
            order: Order = entry["order"]
            logger.error(
                "[pending] TTL %ds истёк для %s (%s, %s x%s) — снимаем с отслеживания; "
                "фактическое состояние определит сверка с брокером",
                PENDING_ORDER_TTL_SEC, order.ticker, key, order.side, order.qty,
            )
            self._pending_orders.pop(key, None)

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        return result if result == result else 0.0   # NaN → 0.0

    # ── price loop ────────────────────────────────────────────────────────

    async def _price_loop(self) -> None:
        tick_count = 0
        while self._running:
            # ### FIXED 2026-08-02 (Fix A) ###
            # Опрос статусов идёт ДО проверки halt: уже отправленные ордера
            # нужно довести до конца даже при остановленной торговле, иначе
            # позиция у брокера останется неучтённой в MarketState.
            try:
                await self._poll_pending_orders()
                self._expire_pending_orders()
            except Exception as exc:
                logger.exception("Pending order poll error: %s", exc)

            if self._risk.is_halted:
                logger.error("RiskManager HALT — trading suspended")
                await asyncio.sleep(self._price_poll_interval)
                continue

            try:
                await self._tick()
                tick_count += 1
                if tick_count % RECONCILE_INTERVAL_TICKS == 0:
                    await self._reconcile_state()
            except Exception as exc:
                logger.exception("Price loop error: %s", exc)

            await asyncio.sleep(self._price_poll_interval)

    async def _tick(self) -> None:
        """Один торговый тик: получить цены со всех брокеров → on_bar → исполнить ордера."""
        tickers: List[str] = list(
            {t for s in self._strategies for t in getattr(s, "tickers", [])}
        )
        if not tickers:
            return

        loop = asyncio.get_running_loop()

        # Разбиваем тикеры по брокерам и фетчим цены параллельно у каждого
        by_broker: Dict[str, List[str]] = {}
        for t in tickers:
            by_broker.setdefault(_guess_broker(t), []).append(t)

        prices: Dict[str, float] = {}
        fetch_tasks = []
        fetch_broker_names = []
        for broker_name, broker_tickers in by_broker.items():
            router = self._routers.get(broker_name)
            if router is None:
                logger.warning("No router for broker=%s — skipping tickers %s", broker_name, broker_tickers)
                continue
            fetch = router.fetch_prices
            if asyncio.iscoroutinefunction(fetch):
                fetch_tasks.append(fetch(broker_tickers))
            else:
                fetch_tasks.append(loop.run_in_executor(None, fetch, broker_tickers))
            fetch_broker_names.append(broker_name)

        if not fetch_tasks:
            logger.warning("No routers available for any ticker — skipping tick")
            return

        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for broker_name, result in zip(fetch_broker_names, results):
            if isinstance(result, Exception):
                logger.warning("fetch_prices failed for broker=%s: %s", broker_name, result)
                continue
            prices.update(result or {})

        if not prices:
            logger.warning("Empty prices response from all brokers")
            return

        self._state.timestamp = utcnow()
        self._state.prices.update(prices)
        self._state.pnl_history.append((self._state.timestamp, self._state.portfolio_value))

        close_orders = self._risk.check_positions(self._state)
        for order in close_orders:
            # ### FIXED 2026-08-02 (Fix A / К1) ###
            # Было: `await self._submit(order)` без apply_order — закрытие по
            # стопу/тейку уходило брокеру, но позиция оставалась в MarketState
            # открытой, и на следующем тике стоп срабатывал повторно.
            submitted = await self._submit_and_track(order)
            if submitted:
                self._portfolio.apply_order(order)

        for strategy in self._strategies:
            orders: List[Order] = strategy.on_bar(self._state)
            orders = self._apply_strategist_filter(orders)
            for order in orders:
                # ### FIXED 2026-08-02 ###
                # prices.get() мог вернуть None (тикер есть у стратегии, но
                # брокер цену не отдал) и затирал цену, уже проставленную
                # стратегией, — дальше None уходил в apply_order.
                price = (
                    prices.get(order.ticker)
                    or self._state.prices.get(order.ticker)
                    or order.filled_price
                )
                if not price:
                    logger.warning("Нет цены для %s — ордер пропущен", order.ticker)
                    continue
                order.filled_price = price

                if self._risk.check_order(order, self._state):
                    submitted = await self._submit_and_track(order)
                    if submitted:
                        self._portfolio.apply_order(order)

        self._risk.check_drawdown(self._state)
        if hasattr(self._risk, "check_daily_loss"):
            self._risk.check_daily_loss(self._state)

        logger.info(
            "[tick] equity=%.2f cash=%.2f positions=%d pnl=%.2f pending=%d brokers=%s",
            self._state.portfolio_value, self._state.cash,
            len(self._state.positions), self._state.total_pnl,
            len(self._pending_orders), list(by_broker.keys()),
        )

    # ── news risk pipeline (событийный, не по таймеру) ─────────────────────

    def _on_news_assessment(self, ticker: str, assessment) -> None:
        """
        Callback из NewsRiskPipeline (фоновый поток) — вызывается СРАЗУ,
        как только релевантная новость прошла дешёвый фильтр и получила
        полный DeepSeek-анализ. Обновляет state.news[ticker] немедленно,
        без ожидания следующего price-тика.

        ### FIXED 2026-08-02 (Fix C) ###
        Fail-closed по отношению к предыдущему снимку: пустой или битый
        assessment больше не затирает уже накопленный новостной контекст и
        не роняет фоновый поток пайплайна необработанным исключением.
        Раньше assessment=None давал AttributeError внутри чужого потока —
        колбэк умирал молча, а последующие новости по тикеру терялись.
        """
        if assessment is None:
            logger.warning(
                "[news-event] %s: получен пустой assessment — снимок не обновляется "
                "(сохраняем предыдущий новостной контекст)",
                ticker,
            )
            return

        from src.engine.market_state import NewsSnapshot

        try:
            prev = self._state.news.get(ticker)
            headline_count = (prev.headline_count + 1) if prev else 1

            severity = self._safe_float(getattr(assessment, "severity", 0.0))
            materiality = self._safe_float(getattr(assessment, "materiality", 0.0))
            effective_risk = getattr(assessment, "effective_risk", None)
            if effective_risk is None:
                effective_risk = severity * materiality

            snapshot = NewsSnapshot(
                ticker=ticker,
                risk_category=str(getattr(assessment, "risk_category", "none") or "none"),
                severity=severity,
                materiality=materiality,
                sentiment=self._safe_float(getattr(assessment, "sentiment", 0.0)),
                horizon=str(getattr(assessment, "horizon", "short") or "short"),
                headline_count=headline_count,
                rationale=str(getattr(assessment, "rationale", "") or ""),
                source_title=str(getattr(assessment, "source_title", "") or ""),
                source_link=str(getattr(assessment, "source_link", "") or ""),
                effective_risk=assessment.effective_risk,
            )
        except Exception as exc:
            logger.warning(
                "[news-event] %s: не удалось построить NewsSnapshot (%s) — "
                "предыдущий снимок сохранён", ticker, exc,
            )
            return

        self._state.news[ticker] = snapshot
        logger.info(
            "[news-event] %s | category=%s effective_risk=%.2f horizon=%s action=%s | %s",
            ticker, snapshot.risk_category, self._safe_float(effective_risk),
            snapshot.horizon, getattr(assessment, "action", "?"),
            snapshot.source_title[:80],
        )

    async def _cold_start_news_backfill(self) -> None:
        """
        Холодный старт: собирает RSS-бэклог за несколько дней ДО первого
        price-тика, отбирает top-5 самых релевантных новостей на тикер через
        малую e5-модель и прогоняет их через DeepSeek — чтобы NewsSnapshot
        (и, следовательно, dampen conviction) был заполнен ДО первого решения
        Kronos, а не "с чистого листа". Long-горизонт новостей (санкции,
        governance) ищем за более длинный период (14 дн.), short — за 3 дня,
        в соответствии с их TTL в NewsMemoryStore.
        """
        if self._news_pipeline is None:
            return

        tickers = list({t for s in self._strategies for t in getattr(s, "tickers", [])})
        if not tickers:
            return

        try:
            from src.news_agent_client.keyword_filter import EmbeddingFilter

            loop = asyncio.get_running_loop()
            # Берём максимальный TTL (14 дней = long), чтобы не упустить долгоживущие
            # новости — top-5 по каждому тикеру всё равно отфильтрует шум.
            backlog = await loop.run_in_executor(
                None, self._news_pipeline.collector.fetch_backlog, 14
            )
            if not backlog:
                logger.info("[cold-start] RSS backlog пуст — пропускаем предторговый анализ новостей")
                return

            ranker = EmbeddingFilter(self._news_pipeline.watch_terms)
            top_items = []
            for ticker in tickers:
                ranked = await loop.run_in_executor(None, ranker.rank_top_k, ticker, backlog, 5)
                top_items.extend(ranked)

            if not top_items:
                logger.info("[cold-start] Ни одна новость из бэклога не прошла e5-ранжирование")
                return

            latest = await loop.run_in_executor(None, self._news_pipeline.backfill, top_items)
            for ticker, assessment in latest.items():
                self._on_news_assessment(ticker, assessment)

            logger.info(
                "[cold-start] Предторговый анализ новостей завершён: %d тикеров получили начальный NewsSnapshot",
                len(latest),
            )
        except Exception as exc:
            logger.warning("[cold-start] Ошибка предторгового анализа новостей (не критично): %s", exc)

    # ── strategist loop (медленный мозг, DeepSeek V4 Flash) ───────────────

    async def _strategist_loop(self) -> None:
        if self._strategist is None:
            logger.info("LLMStrategist not configured — strategist loop disabled")
            return

        from src.llm_strategist.strategist import build_context

        while self._running:
            try:
                context = build_context(
                    self._state, self._strategies, recent_pnl=list(self._state.pnl_history)
                )
                loop = asyncio.get_running_loop()

                price_regimes = {}
                for s in self._strategies:
                    for ticker, cached in getattr(s, "_regime_cache", {}).items():
                        price_regimes[ticker] = cached[1]
                self._decision = await loop.run_in_executor(
                    None, self._strategist.decide, context, price_regimes
                )

                d = self._decision

                # ── tactical_bias → AITacticStrategy ──
                if d.tactical_bias:
                    for strategy in self._strategies:
                        if hasattr(strategy, "set_tactical_bias"):
                            strategy.set_tactical_bias(d.tactical_bias)
                            logger.info(
                                "tactical_bias передан в %s (%d тикеров)",
                                type(strategy).__name__, len(d.tactical_bias),
                            )

                # ── динамический universe (watchlist + veto от LLM) ──
                # Контракт не менялся: UniverseSelector.get_universe принимает
                # strategist_watchlist / strategist_veto и возвращает снапшот
                # с .tickers и .source. vetoed_pairs приходят парами
                # ("SBER/GAZP") — их разбор на ноги делает сам селектор.
                if self._universe_selector is not None:
                    snap = self._universe_selector.get_universe(
                        strategist_watchlist=d.watchlist,
                        strategist_veto=d.vetoed_pairs,
                    )
                    for strategy in self._strategies:
                        if hasattr(strategy, "set_universe"):
                            strategy.set_universe(snap.tickers)

                # ── news_alpha override ──
                if d.news_alpha_override is not None:
                    for strategy in self._strategies:
                        if hasattr(strategy, "news_alpha"):
                            strategy.news_alpha = d.news_alpha_override

                # ── ВАЖНО: watch_terms теперь инициализируются ТОЛЬКО при старте ──
                # На старте LiveEngine сидирует ключевые слова через KronosDefaultStrategy
                # (seed_watch_terms) и передаёт их в NewsRiskPipeline. Стратегист больше не
                # трогает watch_terms, чтобы не перегенерировать одни и те же слова каждые
                # 15 минут и не зависеть от LLM для базовой работоспособности новостного
                # фильтра.
                # Если понадобится динамически обновлять критерии (при смене watchlist),
                # это будет сделано явным событием (отдельный вызов AliasGenerator /
                # KronosDefaultStrategy), а не внутри периодического цикла стратегиста.

            except Exception as exc:
                logger.warning("Strategist loop error: %s", exc)

            await asyncio.sleep(self._strategist_poll_interval)

    # ── strategist filter (halt/veto/size) ─────────────────────────────────

    def _apply_strategist_filter(self, orders: List[Order]) -> List[Order]:
        """
        halt → блокируются только открывающие ордера (закрытия проходят)
        veto → отбрасываются ордера по тикерам из vetoed_pairs
        size → qty умножается на size_multiplier (float! квантизация — в роутере)
        """
        d = self._decision
        if d is None or d.is_fallback:
            return orders

        veto = d.veto_set()
        out: List[Order] = []
        for order in orders:
            is_closing = getattr(order, "closes_position", False) or self._is_reducing(order)

            if d.halted and not is_closing:
                logger.info("Strategist HALT — блокирован открывающий ордер %s", order.ticker)
                continue
            if order.ticker in veto and not is_closing:
                logger.info("Strategist VETO — блокирован ордер %s", order.ticker)
                continue

            if not is_closing and d.size_multiplier != 1.0:
                order.qty = order.qty * d.size_multiplier
                if order.qty <= 0:
                    logger.info("size_multiplier=%.3f → qty≈0, ордер %s пропущен",
                                d.size_multiplier, order.ticker)
                    continue

            out.append(order)   # ← было потеряно

        return out

    def _is_reducing(self, order: Order) -> bool:
        pos = self._state.positions.get(order.ticker)
        if pos is None:
            return False
        pos_qty = getattr(pos, "qty", 0)
        side = getattr(order, "side", "")
        side_val = side.value if hasattr(side, "value") else str(side)
        return (pos_qty > 0 and side_val.upper() == "SELL") or (pos_qty < 0 and side_val.upper() == "BUY")

    # ── submit helper (мультибиржевой роутинг) ─────────────────────────────

    async def _submit_and_track(self, order: Order) -> bool:
        """
        ### FIXED 2026-08-02 (Fix A) ###
        Отправляет ордер и, если брокер его принял без подтверждения сделки,
        ставит на отслеживание вместо того, чтобы считать неисполненным.

        Возвращает True ТОЛЬКО при подтверждённом немедленном исполнении —
        именно в этом случае вызывающий код применяет ордер к портфелю.
        Для принятых, но неподтверждённых ордеров возвращается False, а
        apply_order выполнит _poll_pending_orders() после подтверждения.

        Ключ отслеживания берётся ПОСЛЕ submit: роутеры перезаписывают
        order.order_id идентификатором брокера.
        """
        filled = await self._submit(order)
        if filled:
            return True

        if _status_str(order.status) == "accepted":
            key = order.order_id or f"{order.ticker}:{id(order)}"
            self._pending_orders[key] = {"order": order, "since": time.monotonic()}
            logger.info(
                "[pending] %s принят брокером без подтверждения исполнения (id=%s) — "
                "поставлен на отслеживание",
                order.ticker, key,
            )

        return False

    async def _submit(self, order: Order) -> bool:
        router = self._router_for(order.ticker)
        if router is None:
            logger.error("No router available for ticker=%s — order rejected", order.ticker)
            return False

        loop = asyncio.get_running_loop()
        try:
            submit = router.submit
            if asyncio.iscoroutinefunction(submit):
                return await submit(order)
            return await loop.run_in_executor(None, submit, order)
        except Exception as exc:
            logger.error("Submit failed for %s: %s", order.ticker, exc)
            return False
