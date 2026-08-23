import os

# Some modules (e.g. backend.main) validate Settings at import time, so a
# required env var must be present before any test module's imports run.
# Centralized here instead of a per-file setdefault + import-order workaround.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-dummy")

import pytest  # noqa: E402

from backend.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Prevent a cached Settings instance from leaking between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
