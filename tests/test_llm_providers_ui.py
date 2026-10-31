"""Выбор провайдера модели вместо ручного адреса."""
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from finvex.core.urlguard import UnsafeUrl, check_llm_url
from finvex.web.llm_providers import LOCAL_KEY, PRESETS, preset
from finvex.web.settings import ENV_KEY, LlmSettings, SettingsError


def mk(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    return LlmSettings(tmp_path / "llm_shared.yaml")


def test_presets_are_valid_and_unique():
    ids = [p["id"] for p in PRESETS]
    assert len(ids) == len(set(ids)) and ids[-1] == "custom" and preset("ollama")["needs_key"] is False and preset("nope") is None
    for p in PRESETS:
        if p["base_url"]:
            check_llm_url(p["base_url"])


@pytest.mark.parametrize("url", ["http://api.example.com/v1", "ftp://localhost", "https://u:p@api.example.com", "http://10.0.0.5:8000", "http://169.254.169.254/v1", ""])
def test_unsafe_model_urls_rejected(url):
    with pytest.raises(UnsafeUrl):
        check_llm_url(url)


def test_local_model_needs_no_key_and_remote_never_gets_the_placeholder(tmp_path, monkeypatch):
    s = mk(tmp_path, monkeypatch)
    pub = s.save("http://localhost:11434/v1/", "llama", provider="ollama")
    assert pub["provider"] == "ollama" and pub["needs_key"] is False and s.has_key() and s.api_key() == LOCAL_KEY
    pub = s.save("https://api.openai.com/v1", "m", provider="openai")
    assert not s.has_key() and pub["needs_key"] is True
    s.save("https://api.openai.com/v1", "m", "sk-openai-1234567890", provider="openai")
    assert s.api_key() == "sk-openai-1234567890"
    with pytest.raises(SettingsError, match="http разрешён только для localhost"):
        s.save("http://api.example.com/v1", "m", provider="custom")
    with pytest.raises(SettingsError, match="неизвестный провайдер"):
        s.save("https://api.example.com/v1", "m", provider="evil")


def test_deepseek_env_key_is_not_sent_to_other_providers(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("cfg_loader_under_test", Path(__file__).resolve().parents[1] / "finvex/engine/config_loader.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    s = mk(tmp_path, monkeypatch)
    s.save("https://api.deepseek.com", "m", "sk-file-1234567890", provider="deepseek")
    monkeypatch.setenv(ENV_KEY, "sk-deepseek-env-0000")
    assert s.api_key() == "sk-deepseek-env-0000" and mod.load_llm_shared(str(tmp_path / "llm_shared.yaml"))["api_key"] == "sk-deepseek-env-0000"
    s.save("https://openrouter.ai/api/v1", "m", "sk-or-1234567890", provider="openrouter")
    assert s.api_key() == "sk-or-1234567890" and s.public()["source"] == "file"
    assert mod.load_llm_shared(str(tmp_path / "llm_shared.yaml"))["api_key"] == "sk-or-1234567890"


def test_old_settings_file_without_provider_still_works(tmp_path, monkeypatch):
    s = mk(tmp_path, monkeypatch)
    (tmp_path / "llm_shared.yaml").write_text(yaml.safe_dump({"base_url": "https://api.deepseek.com", "model": "m", "api_key": "sk-old-1234567890"}), encoding="utf-8")
    assert s.public()["provider"] == "deepseek" and s.has_key()
    s.save("https://openrouter.ai/api/v1", "m", provider="openrouter")
    assert yaml.safe_load((tmp_path / "llm_shared.yaml").read_text())["provider"] == "openrouter"
