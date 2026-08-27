"""
RiskManager — проверка лимитов риска до/после исполнения ордеров.

"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from src.engine.market_state import MarketState
from src.engine.orders import Order, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Параметры риска. Читается из engine.yaml → risk:"""
    max_position_pct: float = 0.20
    max_drawdown: float = 0.20
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    max_open_positions: int = 10
    daily_loss_limit_pct: float = 0.05
    min_cash_pct: float = 0.05
    commission_pct: float = 0.0005  # используется для точного pnl_pct в стопах


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self._peak_equity: float = 0.0
        self._halted: bool = False
        self._halted_reason: str = ""  # "drawdown" | "daily_loss" | ""
        # ── дневной лимит убытка ──────────────────────────────────────
        self._day_start_equity: float = 0.0
        self._current_day: Optional[date] = None

    def reset_peak(self, initial_capital: float) -> None:
        self._peak_equity = initial_capital
        self._halted = False
        self._halted_reason = ""
        self._day_start_equity = initial_capital
        self._current_day = None

    @property
    def is_halted(self) -> bool:
        return self._halted

    # ── увеличивает ли ордер риск-экспозицию (фикс С1/С2) ──────────────

    @staticmethod
    def _increases_exposure(order: Order, state: MarketState) -> bool:
        """
        BUY при отсутствии позиции или long-позиции → наращивание long.
        SELL при отсутствии позиции или short-позиции → наращивание short.
        Обратная сторона (закрытие/уменьшение) — не увеличивает риск.
        """
        pos = state.positions.get(order.ticker)
        pos_qty = getattr(pos, "qty", 0.0) if pos else 0.0
        if order.side == OrderSide.BUY:
            return pos_qty >= 0
        return pos_qty <= 0

    # ── проверка входящего ордера ───────────────────────────────────────

    def check_order(self, order: Order, state: MarketState) -> bool:
        if self._halted:
            logger.warning("[RiskManager] Trading halted (%s) — order %s rejected", self._halted_reason, order.order_id)
            order.status = OrderStatus.REJECTED
            return False

        price = order.filled_price or state.prices.get(order.ticker)
        if price is None:
            logger.warning("[RiskManager] No price for %s", order.ticker)
            order.status = OrderStatus.REJECTED
            return False

        equity = state.portfolio_value
        cost = price * abs(order.qty)
        increases_exposure = self._increases_exposure(order, state)

        # Фикс С2: лимит размера позиции — по признаку наращивания экспозиции,
        # не только для BUY (иначе шорты через SELL были вне лимита).
        if increases_exposure:
            existing_pos = state.positions.get(order.ticker)
            existing_value = abs(existing_pos.qty) * price if existing_pos else 0.0
            if (existing_value + cost) / equity > self.cfg.max_position_pct:
                logger.warning(
                    "[RiskManager] Position limit exceeded for %s: %.1f%% > %.1f%%",
                    order.ticker, (existing_value + cost) / equity * 100,
                    self.cfg.max_position_pct * 100,
                )
                order.status = OrderStatus.REJECTED
                return False

        # Фикс С1: min_cash_pct применяется только к BUY — SELL увеличивает
        # кэш, блокировать его из-за низкого кэша нельзя (мешает закрытию).
        if order.side == OrderSide.BUY and state.cash - cost < equity * self.cfg.min_cash_pct:
            logger.warning("[RiskManager] Min cash reserve violated for %s", order.ticker)
            order.status = OrderStatus.REJECTED
            return False

        # Фикс С2: лимит числа открытых позиций — по признаку наращивания
        # экспозиции в НОВЫЙ инструмент, не только для BUY.
        if increases_exposure and order.ticker not in state.positions:
            if len(state.positions) >= self.cfg.max_open_positions:
                logger.warning("[RiskManager] Max open positions (%d) reached", self.cfg.max_open_positions)
                order.status = OrderStatus.REJECTED
                return False

        return True

    # ── стопы / тейки по текущим позициям ─────────────────────────────

    def check_positions(self, state: MarketState) -> list[Order]:
        """
        Возвращает список ордеров закрытия для позиций, пробивших
        stop_loss/take_profit. pnl_pct учитывает комиссию закрытия.
        """
        close_orders: list[Order] = []

        for ticker, pos in list(state.positions.items()):
            price = state.prices.get(ticker)
            if price is None or pos.avg_price == 0:
                continue

            pnl_pct = (price - pos.avg_price) / pos.avg_price
            if pos.qty < 0:
                pnl_pct = -pnl_pct
            pnl_pct -= self.cfg.commission_pct * 2

            triggered = False
            reason = ""

            if self.cfg.stop_loss_pct > 0 and pnl_pct <= -self.cfg.stop_loss_pct:
                triggered = True
                reason = f"stop_loss ({pnl_pct:.2%} incl. commission)"
            elif self.cfg.take_profit_pct > 0 and pnl_pct >= self.cfg.take_profit_pct:
                triggered = True
                reason = f"take_profit ({pnl_pct:.2%} incl. commission)"

            if triggered:
                side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
                logger.info("[RiskManager] Closing %s | reason=%s | qty=%s", ticker, reason, abs(pos.qty))
                order = Order(
                    order_id=str(uuid.uuid4()), ticker=ticker, side=side,
                    qty=abs(pos.qty), filled_price=price,
                )
                order.closes_position = True
                close_orders.append(order)

        return close_orders

    # ── глобальный drawdown-стоп ────────────────────────────────────────

    def check_drawdown(self, state: MarketState) -> None:
        equity = state.portfolio_value
        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self.cfg.max_drawdown:
                logger.error("[RiskManager] Max drawdown breached: %.2f%% | TRADING HALTED", drawdown * 100)
                self._halted = True
                self._halted_reason = "drawdown"

    # ── дневной лимит убытка ─────────────────────────────────────────────

    def check_daily_loss(self, state: MarketState) -> None:
        """
        Проверяет дневной лимит убытка (daily_loss_limit_pct).
        Сбрасывает day_start_equity при смене календарного дня (UTC).

        Фикс К9: halt, вызванный ЭТИМ методом (daily_loss), теперь сам
        сбрасывается на смене дня — раньше _halted оставался True навечно
        после первого срабатывания, даже когда новый день начинался чисто.
        Halt от check_drawdown() (глобальный) НЕ сбрасывается здесь намеренно —
        просадка максимума эквити это не дневная метрика, снимается только
        явным reset_peak() (например, при осознанном рестарте стратегии).
        """
        today = state.timestamp.date() if hasattr(state.timestamp, "date") else None
        if today is None:
            return

        if self._current_day != today:
            self._current_day = today
            self._day_start_equity = state.portfolio_value
            if self._halted_reason == "daily_loss":
                logger.info("[RiskManager] New trading day %s — daily_loss halt lifted", today)
                self._halted = False
                self._halted_reason = ""
            logger.debug("[RiskManager] New trading day %s, start_equity=%.2f", today, self._day_start_equity)
            return

        if self._day_start_equity <= 0:
            return

        daily_pnl_pct = (state.portfolio_value - self._day_start_equity) / self._day_start_equity
        if daily_pnl_pct <= -self.cfg.daily_loss_limit_pct:
            logger.error(
                "[RiskManager] Daily loss limit breached: %.2f%% <= -%.2f%% | TRADING HALTED for today",
                daily_pnl_pct * 100, self.cfg.daily_loss_limit_pct * 100,
            )
            self._halted = True
            self._halted_reason = "daily_loss"