import pytest
import yaml

from finvex.web.settings import ENV_KEY, LlmSettings, SettingsError


def mk(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    return LlmSettings(tmp_path / "llm_shared.yaml")


def test_key_saved_but_never_returned(tmp_path, monkeypatch):
    s = mk(tmp_path, monkeypatch)
    pub = s.save("https://api.deepseek.com/", "m", "sk-test-1234567890")
    assert pub["has_key"] and pub["key_hint"] == "…7890" and "sk-test-1234567890" not in str(pub)
    assert yaml.safe_load((tmp_path / "llm_shared.yaml").read_text())["api_key"] == "sk-test-1234567890"
    s.save("https://api.deepseek.com", "m2", "")
    assert s.api_key() == "sk-test-1234567890"
    s.save("https://api.deepseek.com", "m2", clear=True)
    assert not s.has_key()


@pytest.mark.parametrize("url,key", [("http://evil.example", ""), ("", ""), ("https://api.deepseek.com", "short"), ("https://api.deepseek.com", "has space 12345")])
def test_bad_input_rejected(tmp_path, monkeypatch, url, key):
    with pytest.raises(SettingsError):
        mk(tmp_path, monkeypatch).save(url, "m", key)


def test_env_wins(tmp_path, monkeypatch):
    s = mk(tmp_path, monkeypatch)
    s.save("https://api.deepseek.com", "m", "sk-file-1234567890")
    monkeypatch.setenv(ENV_KEY, "sk-env-abcdefghij")
    assert s.api_key() == "sk-env-abcdefghij" and s.public()["source"] == "env"
