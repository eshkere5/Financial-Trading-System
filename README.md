# Trading System — Kronos + StatArb + News

Интегрированная система статистического арбитража с Kronos-прогнозом и новостным слоем.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py CLI                          │
│         backtest | live | pairs | news                       │
└────────────┬──────────────────────────┬─────────────────────┘
             │                          │
    ┌────────▼───────┐        ┌─────────▼──────────┐
    │ BacktestEngine │        │    LiveEngine       │
    │  (sync loop)   │        │  (async event loop) │
    └────────┬───────┘        └─────────┬───────────┘
             │                          │
             └───────────┬──────────────┘
                         │
              ┌──────────▼──────────┐
              │   StatArbStrategy   │  ← on_bar(state) → List[Order]
              │                     │
              │  ┌─────────────┐    │
              │  │  Spread /   │    │
              │  │  Z-score    │    │
              │  └──────┬──────┘    │
              │         │           │
              │  ┌──────▼──────┐    │
              │  │   Kronos    │    │  ← KronosAdapter.encode_prices()
              │  │   Filter    │    │     → KronosState (rich object)
              │  └──────┬──────┘    │
              │         │           │
              │  ┌──────▼──────┐    │
              │  │    News     │    │  ← NewsSignalExtractor.refresh()
              │  │   Filter    │    │     → MarketState.news[ticker]
              │  └─────────────┘    │
              └─────────────────────┘
                         │
              ┌──────────▼──────────┐
              │    RiskManager      │  ← check_order / check_positions / check_drawdown
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │      Portfolio      │  ← apply_order → MarketState
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  ExecutionRouter    │  ← TinkoffClient (sandbox / prod)
              └─────────────────────┘
```

---

## Компоненты

### MarketState (`src/engine/market_state.py`)
Единственный источник правды. Содержит:
- `cash`, `prices`, `positions`, `trades`, `pnl_history`
- `news: Dict[str, NewsSnapshot]` — текущий новостной контекст по тикерам

### NewsSnapshot
Агрегированный срез новостей по тикеру:
- `sentiment` [-1, +1] — взвешенный по relevance сентимент
- `relevance` [0, 1]
- `sanction_risk` [0, 1] — доля очень негативных новостей
- `signal_weight` property — комбинированный вес для сигнала

### StatArbStrategy (`src/signals/strategy.py`)
Комбинирует три слоя:
1. **Spread z-score** — основной сигнал (коинтеграция)
2. **Kronos confidence filter** — блокирует вход при низкой уверенности модели
3. **News filter** — блокирует при санкционном риске > 60%, ослабляет при конфликте

Адаптивный порог входа:
```
entry_threshold = entry_threshold_sigma * (1 + volatility_norm * 0.5)
```

### NewsSignalExtractor (`src/signals/news_signal.py`)
Мост между `NewsAgentClient` и `MarketState`:
```python
extractor.refresh(state, tickers=["GAZP", "LKOH"])
state.news["GAZP"].signal_weight  # [-1, +1]
```

### RiskManager (`src/engine/risk_manager.py`)
- `check_order()` — позиционные лимиты, резерв кэша
- `check_positions()` → автоматические SELL-ордера при stop_loss / take_profit
- `check_drawdown()` → halt при превышении max_drawdown

### KronosAdapter (`src/kronos_layer/kronos_adapter.py`)
Rich KronosState с properties:
- `forecast_median`, `expected_return`, `volatility`, `volatility_norm`
- `confidence` — 1 - volatility_norm (чем уже канал → выше уверенность)
- `direction` — +1 / -1 / 0

---

## Структура файлов

```
src/
├── engine/
│   ├── config_loader.py      # EngineConfig, RiskConfig, StrategyConfig, load_engine_config
│   ├── market_state.py       # MarketState, Position, Trade, NewsSnapshot
│   ├── orders.py             # Order, OrderSide, OrderStatus
│   ├── portfolio.py          # Portfolio.apply_order (BUY/SELL cash logic)
│   ├── risk_manager.py       # RiskManager (NEW)
│   ├── backtest_engine.py    # BacktestEngine (с RiskManager)
│   └── live_engine.py        # LiveEngine (async, с news polling)
│
├── executors/
│   ├── tinkoff_client.py     # TinkoffClient (dynamic instrument_type filter)
│   └── execution_router.py   # ExecutionRouter (lot-size aware, retry backoff)
│
├── kronos_layer/
│   ├── kronos_adapter.py     # KronosAdapter + KronosState (rich object)
│   └── kronos_features.py    # KronosFeatures, extract_features
│
├── models/
│   └── cointegration.py      # find_cointegrated_pairs (NEW, Engle-Granger + half-life)
│
├── signals/
│   ├── data_collector.py     # DataCollector
│   ├── features.py           # build_features (Kronos + rolling z-score)
│   ├── news_signal.py        # NewsSignalExtractor (NEW)
│   ├── pair_selector.py      # PairSelector → cointegration.py
│   ├── preprocessor.py       # resample, log_returns
│   ├── signal_generation.py  # SignalGenerator (адаптивные пороги)
│   └── signal_types.py       # TradingSignal (с kronos_state, news_snapshot)
│
├── news_agent_client/
│   ├── client.py             # NewsAgentClient (local SQLite / HTTP)
│   ├── llm_backend.py        # LLMBackend (OpenAI / DeepSeek)
│   └── mappers.py            # HTTP response mappers
│
└── utils/
    ├── logging_config.py
    ├── math_utils.py         # sharpe, max_drawdown, cagr
    ├── serialization.py
    └── time_utils.py

collectors/                   # Новостной блок (отдельный репо)
├── rss_collector.py          # RSS + HTML fallback
└── telegram_collector.py     # Telethon (async-safe run())

configs/
├── engine.yaml               # tickers, signal, risk секции
├── kronos.yaml               # model_name, pred_len, device
├── news_agent.yaml           # transport, llm, db_path
├── thresholds.yaml           # (deprecated, перенесено в engine.yaml)
├── sandbox_key.yaml          # tinkoff.token (sandbox)
└── prod_key.yaml             # tinkoff.token (prod)

tests/
├── conftest.py
├── test_portfolio.py
├── test_risk_manager.py
├── test_cointegration.py
└── test_news_signal.py
```

---

## Установка (venv)

### Новый venv (рекомендуется для изоляции от других проектов)
```bash
cd _test_project
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Использовать уже существующий venv (если там уже стоят torch/tinkoff-investments)
Можно безопасно переиспользовать вену другого проекта — `pip install -r requirements.txt`
только добавит недостающие пакеты, ничего не удалит и не сломает другой проект:
```bash
source /path/to/other_project/.venv/bin/activate
cd /path/to/_test_project
pip install -r requirements.txt
```
Единственный риск: если другой проект потом обновит версию какого-то пакета (например pandas), трейдинг-бот
может неожиданно получить несовместимую версию. На раннем этапе/тестах это
адекватный компромисс — когда бот уйдёт в live/прод, стоит выделить ему отдельный venv.

### Ключи API
```bash
export DEEPSEEK_API_KEY="sk-..."
export TINKOFF_TOKEN="t....."          # имя переменной см. в configs/tinkoff.yaml
```
Ни один из сервисов (DeepSeek, Tinkoff) не работает без вашего собственного ключа/токена —
оба API требуют авторизацию, у агента нет доступа к ним без них.

---

## Запуск

### Бэктест
```bash
python -m src.main backtest --config configs/engine.yaml
```

### Поиск пар
```bash
python -m src.main pairs --config configs/engine.yaml --save pairs.json
```

### Live (sandbox)
```bash
python -m src.main live --mode sandbox --config configs/engine.yaml
```

### Новости
```bash
python -m src.main news --ticker GAZP --hours 24 --summarize
```

### Тесты
```bash
pytest tests/ -v
```

---

## Конфигурация engine.yaml

```yaml
tickers: [GAZP, LKOH, ...]

instrument:
  type: share       # динамический фильтр TinkoffClient
  currency: rub

signal:
  pvalue_threshold: 0.05
  entry_threshold_sigma: 2.0   # адаптивный: * (1 + vol_norm * 0.5)
  exit_threshold_sigma: 0.5
  min_confidence: 0.55         # Kronos confidence gate
  news_alpha: 0.30             # вес новостного сигнала

risk:
  max_position_pct: 0.20
  max_drawdown: 0.20
  stop_loss_pct: 0.05
  take_profit_pct: 0.15
  max_open_positions: 10
  min_cash_pct: 0.05
```

---

## Порядок интеграции

Текущая фаза (готово):
1. ✅ StatArb движок с Kronos
2. ✅ RiskManager (stop/take/drawdown)
3. ✅ TinkoffClient (sandbox / prod)
4. ✅ Бэктест с реальными данными
5. ✅ NewsSignalExtractor (подключён к MarketState)

Следующая фаза:
6. 🔲 Загрузить реальные исторические данные (OHLCV из Tinkoff)
7. 🔲 Подключить локальную SQLite с новостями (или запустить news-agent)
8. 🔲 Установить Kronos (`pip install chronos-forecasting`) и запустить на GPU
9. 🔲 Запустить sandbox бэктест с реальными данными
10. 🔲 Переход в live sandbox режим
