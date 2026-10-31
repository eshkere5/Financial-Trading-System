# LiteLLM и выбор модели в Finvex

Finvex подключает модель в «Настройки» → «Модель для аналитики, чата и агентов»: выберите провайдера, впишите название модели и ключ.
LiteLLM нужен, если хотите один адрес к нескольким провайдерам; для обычной работы он необязателен.

| Что | Куда |
|---|---|
| Ключ провайдера | В Finvex (поле «API-ключ») либо, при LiteLLM, в `.env.litellm` → `PROVIDER_API_KEY` |
| Название модели | В Finvex; для LiteLLM это `model_name` из конфига (`finvex-main`) |
| Ключ шлюза LiteLLM | `LITELLM_MASTER_KEY` из `.env.litellm`, вписывается в Finvex |
| Данные рынка (MOEX ISS, Bybit) | Нигде: ключи не нужны |

## Запуск (Windows, PowerShell, из корня проекта)

```powershell
python -m pip install "litellm[proxy]"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
Unblock-File .\scripts\start-litellm.ps1
.\scripts\start-litellm.ps1                    # создаст deploy\litellm\litellm_config.yaml
notepad .\deploy\litellm\litellm_config.yaml  # api_base и идентификатор модели
notepad .\.env.litellm                         # PROVIDER_API_KEY
.\scripts\start-litellm.ps1                    # шлюз на http://localhost:4000
```

В Finvex: провайдер «LiteLLM», модель `finvex-main`, ключ `LITELLM_MASTER_KEY`.

LiteLLM работает отдельным процессом. Режим без базы: модели в файле, один мастер-ключ; виртуальные ключи и учёт расходов требуют Postgres.
