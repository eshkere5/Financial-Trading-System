Financial Trading System — Kronos + Regime Layer + News Risk Pipeline
Исследовательская торговая система: foundation-модель временных рядов Kronos,
детектор рыночного режима с послережимным сайзингом, событийный новостной
риск-контур (RSS → e5-фильтр → LLM) и полный журнал решений.
Рынки: акции MOEX (Tinkoff) и крипта (Bybit). Курсовой проект / research sandbox.

Статус: pre-soak frozen state. Бэктесты v2 завершены (режимный слой
измерен на двух рынках), следующий шаг — 21-дневный live soak-test
на VM с журналом решений как источником истины.

Архитектура
text
                        main.py CLI
              backtest | live | news
                 │              │
        ┌────────▼─────┐  ┌─────▼──────────┐
        │BacktestEngine│  │  LiveEngine    │  (async event loop, 60s tick)
        └────────┬─────┘  └─────┬──────────┘
                 └───────┬──────┘
                         │
            ┌────────────▼─────────────┐
            │   AITacticStrategy       │  ← on_bar(state) → List[Order]
            │                          │
            │  1. Kronos forecast      │  KronosAdapter → direction + expected_return
            │  2. Regime layer         │  RegimeDetector(OHLCV) → режим →
            │     (site sizing)        │  regime_multiplier(режим, сторона) [0.2–1.4]
            │  3. News risk dampener   │  effective_risk → factor 0.3–0.7 / block
            └────────────┬─────────────┘
                         │
            ┌────────────▼─────────────┐
            │   NewsRiskPipeline       │  (событийная модель)
            │  RSS(30s) → e5-filter →  │  DeepSeek только на релевантное,
            │  dedup → DeepSeek → TTL  │  cap 20 вызовов/цикл
            └────────────┬─────────────┘
                         │
            ┌────────────▼─────────────┐
            │      RiskManager         │  лимиты, stop/take, drawdown halt
            ├──────────────────────────┤
            │      TradeJournal        │  SQLite: КАЖДОЕ решение с контекстом
            │  (opened/skipped/closed, │  (режим, множитель, риск, rationale)
            │   outcomes T+1h/4h/24h)  │
            └────────────┬─────────────┘
                         │
            ┌────────────▼─────────────┐
            │     ExecutionRouter      │  Tinkoff (MOEX) / Bybit (testnet)
            └──────────────────────────┘
Ключевые компоненты
Regime layer (src/regime/regime_detector.py)
Офлайн-классификация бара в один из режимов (BULL_TREND, BEAR_TREND,
HIGH_VOL_CHOP, RANGE_BOUND, RANGE_SQUEEZE, RALLY_FRENZY, RECOVERY, CRASH)
по OHLCV-признакам. Измеренные в бэктесте множители сайзинга
(режим × сторона × класс актива) применяются к размеру позиции:
усиливаем плюсовые режимы (до ×1.4), душим минусовые (до ×0.2).
Чистая функция, детерминированная, walk-forward совместимая.

TradeJournal (src/engine/trade_journal.py)
SQLite-журнал всех решений (не только сделок): открытия, пропуски с причиной,
закрытия, исходы T+1h/4h/24h. Включая сделки, которые режимный слой
пропустил бы — для контрфактического what-if анализа.

NewsRiskPipeline (src/news_agent_client/)
rss_collector.py — непрерывный опрос фидов (Interfax, РБК, Finam,
МосБиржа, крипто-фиды), poll ~30 сек;

keyword_filter.py — локальный e5-small фильтр релевантности, бесплатно;

news_risk_pipeline.py — дедупликация по item_id, TTL-память
(short=3 дня, long=14 дней, линейное затухание), cap 20 LLM-вызовов/цикл;

news_deep_dive.py — единственная точка вызова LLM (OpenAI-совместимый
клиент, DeepSeek по умолчанию; base_url/model — из конфига).
Строгий JSON: risk_category, severity, materiality, sentiment, horizon.
Fail-closed: при ошибке API возвращает None, прежний риск сохраняется.
Ключ читается из env DEEPSEEK_API_KEY.

Стратегии
src/signals/ai_tactic_strategy.py — основная: Kronos + режимный
множитель + новостной демпфер;

src/engine/kronos_default_strategy.py — базовая Kronos-стратегия
(первичные watch_terms для новостного фильтра).

Аналитика (analysis/)
backtest_kronos.py, run_experiment.py — прогоны бэктестов;

regime_report.py — распределение режимов, hit-rate и PnL по ячейкам;

whatif_regime.py — контрфактический анализ режимных множителей;

smoke_regime.py — офлайн-проверка связки (1 минута, обязательна
перед любым запуском);

journal_report.py — отчёт по журналу решений (после soak'а).

Структура
text
_test_project/
├── analysis/            # бэктесты, режимные отчёты, smoke-тесты
├── configs/             # engine / kronos / news_agent / llm_* (без ключей!)
│                        # ключи: *.key.yaml и llm_shared.yaml — локально, в .gitignore
├── src/
│   ├── engine/          # live_engine, backtest_engine, risk_manager,
│   │                    # trade_journal, market_state, portfolio, orders
│   ├── executors/       # tinkoff_client, bybit_client, routers, lot_utils
│   ├── kronos_layer/    # kronos_adapter (KronosState), kronos_features
│   ├── regime/          # regime_detector
│   ├── news_agent_client/  # rss_collector, keyword_filter, news_risk_pipeline,
│   │                     # news_deep_dive, ticker_intelligence, llm_backend
│   ├── services/        # deepseek_news_analyzer, news_db_migration
│   ├── signals/         # ai_tactic_strategy и базовые сигналы
│   ├── llm_strategist/  # стратегический LLM-слой (posture/size)
│   └── utils/
└── tests/               # e2e и smoke тесты

Kronos/                  # модель — отдельный репо, клонируется рядом:
                         # git clone https://github.com/shiyu-coder/Kronos.git
Установка и запуск
bash
# 1. Код + модель
git clone <этот репо> && cd <репо>
git clone https://github.com/shiyu-coder/Kronos.git

# 2. Окружение (CPU-сборка torch — отдельно, до requirements)
cd _test_project
python3 -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r ../requirements.txt

# 3. Секреты (локально, не в git)
#    configs/*key*.yaml, configs/llm_shared.yaml
#    .env: DEEPSEEK_API_KEY=sk-...

# 4. Предстартовая проверка (офлайн, обязательна)
python analysis/smoke_regime.py

# 5. Запуск
python src/main.py live --mode sandbox          # soak / sandbox
python src/main.py backtest --config configs/engine.yaml
pytest tests/ -v
Запуск под tmux на VM — лог переживает разрыв SSH:
tmux new -s soak → запуск → Ctrl+B, D.

Методология (важнее кода)
Все изменения слоёв сначала измеряются на замороженных бэктестах
(v2: 245 сделок крипта, 127 акции), потом попадают в прод;

Магнитудам прогноза Kronos не доверяем (не откалиброваны) —
используем направление и hit-rate; ожидаемые доходности для
cost-гейта берём из измеренного EV ячеек журнала;

Live soak-test 21 день на testnet: журнал решений — единственный
источник истины для out-of-sample оценки;

Одна переменная за эксперимент: смена риск-LLM, порогов и слоёв
— только между soak'ами, никогда посередине.