import pytest
from pydantic import ValidationError

from backend.config import Settings


def test_settings_requires_openrouter_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_loads_from_env_with_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.openrouter_api_key == "test-key"
    assert settings.triage_model == "nvidia/nemotron-3.5-lightning:free"
    assert settings.investigation_model == "nvidia/nemotron-3-super-120b-a12b:free"
    assert settings.rca_model == "nvidia/nemotron-3-super-120b-a12b:free"
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.voyage_api_key is None
