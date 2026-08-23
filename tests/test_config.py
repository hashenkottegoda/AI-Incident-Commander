import pytest
from pydantic import ValidationError

from backend.config import Settings


def test_settings_requires_anthropic_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_loads_from_env_with_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.anthropic_api_key == "test-key"
    assert settings.triage_model == "claude-haiku-4-5"
    assert settings.investigation_model == "claude-opus-4-8"
    assert settings.rca_model == "claude-opus-4-8"
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.voyage_api_key is None
