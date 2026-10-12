import importlib

import pytest


def _reload(monkeypatch, **env):
    for k in ("QUANTCREW_PROFILE", "QUANTCREW_MODEL", "QUANTCREW_CRITIC_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from quantcrew import config
    return importlib.reload(config)


def test_claude_profile_uses_haiku_for_every_role(monkeypatch):
    c = _reload(monkeypatch, QUANTCREW_PROFILE="claude")
    assert c.MODEL == c.CRITIC_MODEL == "anthropic:claude-haiku-5-5"


def test_explicit_model_overrides_profile(monkeypatch):
    c = _reload(monkeypatch, QUANTCREW_PROFILE="claude", QUANTCREW_CRITIC_MODEL="anthropic:claude-sonnet-5-5")
    assert c.MODEL == "anthropic:claude-haiku-5-5" and c.CRITIC_MODEL == "anthropic:claude-sonnet-5-5"


def test_unknown_profile_is_rejected(monkeypatch):
    with pytest.raises(ValueError):
        _reload(monkeypatch, QUANTCREW_PROFILE="openai")
    _reload(monkeypatch)  # leave the default config loaded for other tests
