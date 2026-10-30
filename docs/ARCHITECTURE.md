# Архитектура Finvex

Три контура с односторонней зависимостью: данные → решение → исполнение. Аналитика не знает о брокерах.

```text
finvex/
├── core/        модели (Instrument, Candle, Quote…), интерфейсы провайдеров, реестр, проверка качества
├── venues/      площадки и брокеры: moex.py, csv_file.py, tinkoff/, bybit/ (SDK подключаются по необходимости)
├── analysis/    kronos/ (адаптер, признаки, мост из свечей), news/ (RSS → фильтр → LLM → риск), regime.py
├── decision/    strategist (LLM, только понижает риск), ai_tactic, risk, universe, signal_types
├── execution/   disabled (по умолчанию), router (Т-Инвест), bybit_router
├── engine/      live/backtest-движки, состояние рынка, портфель, ордера, журнал сделок, загрузка конфигов
├── data/        сборщик исторических данных
├── modules/     реестр модулей платформы: зависимости, включение, состояние (зависит только от utils)
├── utils/
└── app.py       CLI: backtest | live | news   (python -m finvex)
```

## Правила зависимостей

- `core` не импортирует ничего из остальных пакетов.
- `analysis` может использовать `core` и `venues` (данные), но не `decision` и `execution`.
- `decision` использует `analysis` и `core`, но не вызывает брокеров.
- Только `engine` связывает решение с `execution`.
- Новая площадка = новый модуль в `venues/` с декоратором `@register("имя")`. Реестр найдёт его сам (`core/registry.py`).

## Конфигурация

`configs/providers.yaml` задаёт источники данных. Исполнение выключено, пока не указано `execution.enabled: true`.
Kronos ищется в `FINVEX_KRONOS_PATH`, затем `third_party/Kronos`, `Kronos/`, соседней папке `../Kronos`.

## Известные ограничения

- Движок `live_engine.py` (около 1000 строк) пока не разбит.
- Данные MOEX берутся из ISS; для коммерческого использования нужна лицензия биржи.
- Адаптеры Т-Инвест и Bybit пока подключены к движку напрямую через `execution/router.py`, а не через интерфейсы `core.base`.
