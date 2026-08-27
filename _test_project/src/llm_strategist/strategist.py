"""
LLMStrategist — «медленный мозг» системы (DeepSeek V4 Flash).


"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.time_utils import utcnow

logger = logging.getLogger(__name__)

VALID_REGIMES = {"trend", "range", "stress", "unknown"}
VALID_POSTURES = {"normal", "reduced", "halt"}
VALID_STRATEGY_MODES = {"pairs", "ai_tactic", "hybrid"}

MAX_DECISION_REUSE = 3
API_MAX_ATTEMPTS = 3
API_RETRY_BASE_DELAY = 2.0

# ── режимный слой (2026-08-24) ───────────────────────────────────────
# Маппинг выхода RegimeDetector -> (risk_posture, size_multiplier).
# Двухслойность: это ГРУБЫЙ уровень (движок); точные множители по
# направлению (REGIME_MULTIPLIERS) применяются на уровне стратегии.
REGIME_TO_POSTURE: Dict[str, tuple] = {
    "BULL_TREND":    ("normal",  1.0),
    "RALLY_FRENZY":  ("reduced", 0.8),
    "BEAR_TREND":    ("normal",  0.9),
    "RECOVERY":      ("reduced", 0.8),
    "RANGE_SQUEEZE": ("reduced", 0.6),
    "RANGE_CHOP":    ("reduced", 0.5),
    "TRANSITION":    ("reduced", 0.8),
    "CRASH":         ("halt",    0.0),
}

# effective_risk выше этого порога + horizon=long -> принудительный CRASH,
# независимо от того, что говорят цены (новостной шок впереди цены).
NEWS_OVERRIDE_RISK = 0.7

# ранжирование posture для правила "LLM может только понижать"
_POSTURE_RANK = {"normal": 0, "reduced": 1, "halt": 2}

_SYSTEM_PROMPT = """\
Ты — риск-супервизор алгоритмической stat-arb системы (парный трейдинг, \
коинтеграция + Kronos foundation model прогнозы + новостной risk-анализ).
Твоя задача — НЕ генерировать сделки, а оценить рыночный контекст и выдать \
тактические ограничения для детерминированного торгового слоя.

ВАЖНО (с 2026-08-24): рыночный режим (trend/range/stress) определяет \
детерминированный ценовой детектор — он передаётся в контексте как \
"price_regimes". НЕ угадывай режим сам. Твоя задача по режиму — только \
ПОНИЗИТЬ его, если новости сообщают о шоке, которого цена ещё не отразила \
(санкции, регуляторный удар, каскад ликвидаций). Повышать режим против \
детектора запрещено.

Отвечай СТРОГО валидным JSON без markdown:

{
  "regime": "trend|range|stress|unknown",
  "risk_posture": "normal|reduced|halt",
  "size_multiplier": 0.0-1.5,
  "news_alpha_override": null | 0.0-1.0,
  "vetoed_pairs": ["TICKER1/TICKER2", ...],

  "strategy_mode": "pairs|ai_tactic|hybrid",
  "watchlist": ["TICKER1", "TICKER2", ...],
  "tactical_bias": {
    "TICKER": {"direction": 1, "conviction": 0.0-1.0, "horizon": "short|medium|long"}
  },

  "watch_terms": (не более 3 основных терминов на тикер){
    "TICKER": ["полное название компании", "сокращение", "CEO/ключевая персона",
               "бренд продукта", "англ. название", "любые синонимы для поиска новостей"]
  },

  "rationale": "СТРОГО одно предложение, не более 20 слов. Только суть решения."
}

Принципы:
- halt только при ЯВНЫХ признаках рыночного стресса: высокая realized volatility,
  глубокий drawdown портфеля, сильные санкционные/governance новости по нескольким
  ключевым тикерам одновременно. Не останавливай торговлю из-за одной спорной новости.
- risk_posture="reduced" выбирай при умеренном росте волатильности/новостного риска
  или локальных проблемах по части тикеров. normal — базовый режим.
- size_multiplier привязывай к уровню риска: чем выше волатильность, агрегированный
  новостной риск и глубже недавняя просадка, тем ниже size_multiplier.
- vetoed_pairs — если по конкретной паре противоречие: Kronos говорит одно, спред другое,
  новости третье. Формат строго "TICKER1/TICKER2" — разделитель только "/",
  дефис внутри тикера (например BTC-USDT) частью формата НЕ является.
- strategy_mode="pairs" — стандартный режим по умолчанию.
- strategy_mode="ai_tactic" — сильная одиночная возможность БЕЗ пары; заполни
  watchlist и tactical_bias.
- strategy_mode="hybrid" — pairs для устоявшихся связей + ai_tactic для остального.
- tactical_bias заполняй ТОЛЬКО для тикеров из watchlist, direction: 1=long, -1=short, 0=flat.
- conviction < 0.35 не имеет смысла — AITacticStrategy проигнорирует такой сигнал.
- Будь консервативен: сомневаешься -> reduced/pairs, а не halt/ai_tactic.
- Не выдумывай данные, работай только с переданным контекстом.
"""


@dataclass
class StrategistDecision:
    regime: str = "unknown"
    risk_posture: str = "normal"
    size_multiplier: float = 1.0
    news_alpha_override: Optional[float] = None
    vetoed_pairs: List[str] = field(default_factory=list)
    rationale: str = ""

    strategy_mode: str = "pairs"
    watchlist: List[str] = field(default_factory=list)
    tactical_bias: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    watch_terms: Dict[str, List[str]] = field(default_factory=dict)

    timestamp: str = ""
    model: str = ""
    latency_sec: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    is_fallback: bool = False

    @property
    def halted(self) -> bool:
        return self.risk_posture == "halt"

    def veto_set(self) -> set:
        """Множество тикеров под вето. Делим ТОЛЬКО по "/" (фикс С-13)."""
        out: set = set()
        for pair in self.vetoed_pairs:
            text = str(pair).strip()
            if not text:
                continue
            out.add(text.upper())
            for leg in text.split("/"):
                leg = leg.strip()
                if leg:
                    out.add(leg.upper())
        return out


def _neutral_decision(reason: str) -> StrategistDecision:
    return StrategistDecision(
        rationale=f"[fallback] {reason}",
        timestamp=utcnow().isoformat(),
        is_fallback=True,
    )


def _is_retryable(exc: Exception) -> bool:
    """Ретраим только транзиентные сбои (429/5xx/таймауты), не 4xx."""
    name = type(exc).__name__
    if name in {"RateLimitError", "APITimeoutError", "APIConnectionError",
                "InternalServerError", "APIStatusError"}:
        status = getattr(exc, "status_code", None)
        if status is None:
            return True
        return status == 429 or status >= 500
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return isinstance(exc, (TimeoutError, ConnectionError))


class LLMStrategist:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._model = cfg.get("model", "deepseek-v4-flash")
        self._base_url = cfg.get("base_url", "https://api.deepseek.com")
        self._temperature = float(cfg.get("temperature", 0.2))
        self._max_tokens = int(cfg.get("max_tokens", 2000))
        self._thinking = bool(cfg.get("thinking", False))
        self._timeout = int(cfg.get("timeout_seconds", 60))
        self._max_attempts = max(1, int(cfg.get("max_attempts", API_MAX_ATTEMPTS)))
        self._max_reuse = max(0, int(cfg.get("max_decision_reuse", MAX_DECISION_REUSE)))

        api_key = cfg.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY не задан — стратег будет выдавать fallback-решения")
        self._api_key = api_key

        self._log_path = Path(cfg.get("decision_log", "data/strategist_decisions.jsonl"))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._client = None
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key, base_url=self._base_url, timeout=self._timeout)
            except ImportError:
                logger.error("openai package не установлен: pip install openai>=1.0")

        self.total_tokens_in = 0
        self.total_tokens_out = 0

        self._last_good: Optional[StrategistDecision] = None
        self._reuse_count = 0

    # ── С-14: деградация вместо обнуления ограничений ────────────────

    def _degraded(self, reason: str) -> StrategistDecision:
        if self._last_good is not None and self._reuse_count < self._max_reuse:
            self._reuse_count += 1
            stale = replace(
                self._last_good,
                timestamp=utcnow().isoformat(),
                latency_sec=0.0,
                tokens_in=0,
                tokens_out=0,
                is_fallback=False,
                rationale=(
                    f"[stale {self._reuse_count}/{self._max_reuse}: {reason}] "
                    f"{self._last_good.rationale}"
                )[:1000],
            )
            logger.warning(
                "Strategist: LLM недоступен (%s) — переиспользуем предыдущее решение "
                "(posture=%s, size=%.2f), попытка %d/%d",
                reason, stale.risk_posture, stale.size_multiplier,
                self._reuse_count, self._max_reuse,
            )
            return stale

        if self._last_good is not None:
            logger.warning(
                "Strategist: предыдущее решение исчерпало срок переиспользования "
                "(%d циклов) — переходим на нейтральный fallback", self._max_reuse,
            )
            self._last_good = None
        return _neutral_decision(reason)

    # ── режимный слой: подмена/понижение после парсинга LLM ──────────

    def _apply_price_regime(
        self,
        decision: StrategistDecision,
        price_regimes: Optional[Dict[str, Any]],
    ) -> StrategistDecision:
        """
        Подменяет regime/posture/size выводом RegimeDetector.
        price_regimes: {ticker: RegimeInfo | dict с полем regime} из движка.
        Правила:
        - базовый режим = САМЫЙ ЖЁСТКИЙ из режимов активных тикеров;
        - LLM может только ПОНИЗИТЬ posture детектора (новостной шок);
        - новостной override: effective_risk >= NEWS_OVERRIDE_RISK и есть
          long-горизонт новость -> принудительный CRASH (halt на открытия).
        """
        if not price_regimes:
            return decision

        # самый жёсткий режим среди тикеров
        worst = None
        worst_rank = -1
        for ticker, info in price_regimes.items():
            name = getattr(info, "regime", None) or (info.get("regime") if isinstance(info, dict) else None)
            if not name or name not in REGIME_TO_POSTURE:
                continue
            posture, _ = REGIME_TO_POSTURE[name]
            rank = _POSTURE_RANK[posture]
            if rank > worst_rank:
                worst_rank, worst = rank, name
        if worst is None:
            return decision

        det_posture, det_size = REGIME_TO_POSTURE[worst]

        # LLM может только понижать
        llm_rank = _POSTURE_RANK.get(decision.risk_posture, 0)
        if llm_rank > worst_rank:
            final_posture = decision.risk_posture
            final_size = min(decision.size_multiplier, det_size if det_size > 0 else 0.5)
            overridden = f"detector={worst}, LLM понизил до {final_posture}"
        else:
            final_posture, final_size = det_posture, det_size
            overridden = f"detector={worst}"

        decision.regime = worst.lower()
        decision.risk_posture = final_posture
        decision.size_multiplier = final_size
        decision.rationale = (f"[regime:{overridden}] " + decision.rationale)[:1000]
        return decision

    def decide(
        self,
        context: Dict[str, Any],
        price_regimes: Optional[Dict[str, Any]] = None,
    ) -> StrategistDecision:
        if self._client is None:
            return self._log(_neutral_decision("LLM client not configured"))

        # режимы детектора кладём в контекст — LLM их видит, но не выдумывает
        if price_regimes:
            context = dict(context)
            context["price_regimes"] = {
                t: (getattr(i, "regime", None) or (i.get("regime") if isinstance(i, dict) else None))
                for t, i in price_regimes.items()
            }

        user_msg = json.dumps(context, ensure_ascii=False, default=str)
        t0 = time.monotonic()

        kwargs: Dict[str, Any] = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
        )
        if self._thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        resp = None
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_attempts or not _is_retryable(exc):
                    logger.error("Strategist API error (%s): %s", type(exc).__name__, exc)
                    break
                delay = API_RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.random()
                logger.warning(
                    "Strategist API attempt %d/%d failed (%s: %s), retry in %.1fs",
                    attempt, self._max_attempts, type(exc).__name__, exc, delay,
                )
                time.sleep(delay)

        if resp is None:
            return self._log(self._degraded(f"API error: {last_exc}"))

        latency = time.monotonic() - t0
        choice = resp.choices[0]
        raw = choice.message.content or "{}"
        finish_reason = getattr(choice, "finish_reason", None)

        if resp.usage:
            self.total_tokens_in += resp.usage.prompt_tokens
            self.total_tokens_out += resp.usage.completion_tokens

        if finish_reason == "length":
            logger.warning(
                "Strategist: ответ LLM обрезан по max_tokens=%d (finish_reason=length) — fallback",
                self._max_tokens,
            )
            return self._log(self._degraded(
                f"truncated response: finish_reason=length, max_tokens={self._max_tokens}"
            ))

        decision = self._parse(raw)

        if decision.is_fallback:
            return self._log(self._degraded(decision.rationale or "invalid LLM response"))

        decision.timestamp = utcnow().isoformat()
        decision.model = self._model
        decision.latency_sec = round(latency, 2)
        if resp.usage:
            decision.tokens_in = resp.usage.prompt_tokens
            decision.tokens_out = resp.usage.completion_tokens

        # ── режимный слой: подменяем ПОСЛЕ успешного парсинга ────────
        decision = self._apply_price_regime(decision, price_regimes)

        self._last_good = decision
        self._reuse_count = 0

        return self._log(decision)

    def _parse(self, raw: str) -> StrategistDecision:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Strategist вернул невалидный JSON: %.200s", raw)
            return _neutral_decision("invalid JSON from LLM")

        regime = str(data.get("regime", "unknown")).lower()
        if regime not in VALID_REGIMES:
            regime = "unknown"

        posture = str(data.get("risk_posture", "normal")).lower()
        if posture not in VALID_POSTURES:
            posture = "normal"

        try:
            size_mult = float(data.get("size_multiplier", 1.0))
        except (TypeError, ValueError):
            size_mult = 1.0
        size_mult = max(0.0, min(1.5, size_mult))

        alpha = data.get("news_alpha_override")
        if alpha is not None:
            try:
                alpha = max(0.0, min(1.0, float(alpha)))
            except (TypeError, ValueError):
                alpha = None

        vetoed = data.get("vetoed_pairs") or []
        if not isinstance(vetoed, list):
            vetoed = []
        vetoed = [str(p) for p in vetoed][:20]

        strategy_mode = str(data.get("strategy_mode", "pairs")).lower()
        if strategy_mode not in VALID_STRATEGY_MODES:
            strategy_mode = "pairs"

        watchlist = data.get("watchlist") or []
        if not isinstance(watchlist, list):
            watchlist = []
        watchlist = [str(t).upper() for t in watchlist][:40]

        raw_bias = data.get("tactical_bias") or {}
        tactical_bias: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_bias, dict):
            for ticker, bias in raw_bias.items():
                if not isinstance(bias, dict):
                    continue
                try:
                    direction = int(bias.get("direction", 0))
                    direction = max(-1, min(1, direction))
                    conviction = max(0.0, min(1.0, float(bias.get("conviction", 0.0))))
                    horizon = str(bias.get("horizon", "short"))
                    if horizon not in ("short", "medium", "long"):
                        horizon = "short"
                    tactical_bias[str(ticker).upper()] = {
                        "direction": direction,
                        "conviction": conviction,
                        "horizon": horizon,
                    }
                except (TypeError, ValueError):
                    continue

        raw_terms = data.get("watch_terms") or {}
        watch_terms: Dict[str, List[str]] = {}
        if isinstance(raw_terms, dict):
            for ticker, terms in raw_terms.items():
                if not isinstance(terms, list):
                    continue
                cleaned = [str(t).strip() for t in terms if str(t).strip()][:8]
                if cleaned:
                    watch_terms[str(ticker).upper()] = cleaned

        return StrategistDecision(
            regime=regime,
            risk_posture=posture,
            size_multiplier=size_mult,
            news_alpha_override=alpha,
            vetoed_pairs=vetoed,
            strategy_mode=strategy_mode,
            watchlist=watchlist,
            tactical_bias=tactical_bias,
            watch_terms=watch_terms,
            rationale=str(data.get("rationale", ""))[:1000],
        )

    def _log(self, decision: StrategistDecision) -> StrategistDecision:
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Не удалось записать decision log: %s", exc)

        logger.info(
            "Strategist | regime=%s posture=%s size=%.2f mode=%s vetoed=%d bias=%d terms=%d | %s",
            decision.regime, decision.risk_posture, decision.size_multiplier,
            decision.strategy_mode, len(decision.vetoed_pairs),
            len(decision.tactical_bias), len(decision.watch_terms), decision.rationale[:120],
        )
        return decision


def build_context(state, strategies: list, recent_pnl: Optional[list] = None) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "timestamp": utcnow().isoformat(),
        "portfolio": {
            "equity": round(getattr(state, "portfolio_value", 0.0), 2),
            "cash": round(getattr(state, "cash", 0.0), 2),
            "total_pnl": round(getattr(state, "total_pnl", 0.0), 2),
            "n_positions": len(getattr(state, "positions", {})),
        },
        "prices": {k: round(v, 4) for k, v in list(getattr(state, "prices", {}).items())[:40]},
    }

    kronos: Dict[str, Any] = {}
    pairs: List[str] = []
    for strat in strategies:
        for pair, pos in getattr(strat, "positions", {}).items():
            pairs.append(str(pair))
        ks = getattr(strat, "last_kronos_states", None)
        if ks:
            for ticker, k in ks.items():
                kronos[ticker] = {
                    "expected_return": round(getattr(k, "expected_return", 0.0), 5),
                    "volatility_norm": round(getattr(k, "volatility_norm", 0.0), 3),
                    "confidence": round(getattr(k, "confidence", 0.0), 3),
                    "direction": getattr(k, "direction", "flat"),
                }
    ctx["kronos"] = kronos
    ctx["open_pairs"] = pairs

    news = getattr(state, "news", None) or {}
    ctx["news"] = {
        ticker: {
            "risk_category": getattr(snap, "risk_category", "none"),
            "severity": round(getattr(snap, "severity", 0.0), 3),
            "materiality": round(getattr(snap, "materiality", 0.0), 3),
            "effective_risk": round(getattr(snap, "effective_risk", 0.0), 3),
            "sentiment": round(getattr(snap, "sentiment", 0.0), 3),
            "docs": getattr(snap, "headline_count", 0),
        }
        for ticker, snap in list(news.items())[:40]
    }

    if recent_pnl:
        ctx["recent_equity"] = [round(v, 2) for _, v in recent_pnl[-24:]]

    return ctx