"""Готовые подключения к моделям: пользователь выбирает провайдера из списка, а не вводит адрес вручную.

Все варианты говорят на OpenAI-совместимом интерфейсе, поэтому чат, анализ, песочница и новости работают с любым из них без изменений.
Идентификатор модели всегда берётся из кабинета провайдера.
"""
from __future__ import annotations

LOCAL_KEY = "local"      # заглушка для локальных моделей: клиенту OpenAI нужен непустой ключ, сервер его не проверяет

PRESETS = [
    {"id": "deepseek", "title": "DeepSeek", "base_url": "https://api.deepseek.com", "model_hint": "deepseek-v4-flash", "needs_key": True,
     "note": "Подключение по умолчанию. Ключ создаётся в кабинете DeepSeek."},
    {"id": "openai", "title": "OpenAI", "base_url": "https://api.openai.com/v1", "model_hint": "", "needs_key": True,
     "note": "Точное название модели возьмите в кабинете OpenAI."},
    {"id": "openrouter", "title": "OpenRouter (много моделей через один ключ)", "base_url": "https://openrouter.ai/api/v1", "model_hint": "", "needs_key": True,
     "note": "Модель вида провайдер/название, список в каталоге OpenRouter."},
    {"id": "ollama", "title": "Ollama (локально, без ключа)", "base_url": "http://localhost:11434/v1", "model_hint": "", "needs_key": False,
     "note": "Нужны установленный Ollama и скачанная модель. Название модели смотрите командой ollama list."},
    {"id": "litellm", "title": "LiteLLM (локальный шлюз)", "base_url": "http://localhost:4000/v1", "model_hint": "finvex-main", "needs_key": True,
     "note": "Ключ - LITELLM_MASTER_KEY из файла .env.litellm. Запуск: scripts/start-litellm.ps1."},
    {"id": "custom", "title": "Другой сервис (адрес вручную)", "base_url": "", "model_hint": "", "needs_key": True,
     "note": "Любой сервис с OpenAI-совместимым API."},
]


def preset(provider_id: str):
    return next((p for p in PRESETS if p["id"] == provider_id), None)
