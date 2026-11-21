"""Shared test setup. pytest loads this file automatically.

WHY: your real .env has Langfuse keys. Without this, every test that runs
answer_question would send traces to your Langfuse project. Tests must be
offline, free and private, so we blank the keys for every test.
"""

import pytest

from documind import config, tracing


@pytest.fixture(autouse=True)  # autouse = applies to every test without asking
def no_real_services(monkeypatch):
    # Environment variables beat .env in pydantic-settings, so these win.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    # Both are @lru_cache'd: clear them so they re-read the blanked settings.
    config.get_settings.cache_clear()
    tracing.get_langfuse.cache_clear()
    yield
    config.get_settings.cache_clear()
    tracing.get_langfuse.cache_clear()