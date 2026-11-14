"""Tests for config.py and llm.py — no network, no real key needed."""

import pytest

from documind.config import Settings
from documind.llm import MissingApiKeyError, get_llm


def make_settings(**overrides) -> Settings:
    # _env_file=None: ignore any real .env so tests behave the same on every PC.
    return Settings(_env_file=None, **overrides)


def test_defaults_are_sensible():
    s = make_settings()
    assert s.llm_provider == "groq"
    assert s.top_k == 4
    assert s.max_graph_steps == 6
    assert s.notes_dir.name == "notes"


def test_api_key_is_hidden_when_printed():
    s = make_settings(groq_api_key="gsk_super_secret")
    assert "gsk_super_secret" not in repr(s)
    assert s.groq_api_key.get_secret_value() == "gsk_super_secret"


def test_missing_key_gives_a_helpful_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError, match="console.groq.com"):
        get_llm(settings=make_settings())


def test_groq_model_is_built_from_settings():
    # Building the client does not call the network, so this is safe offline.
    llm = get_llm(temperature=0.5, settings=make_settings(groq_api_key="gsk_test", llm_model="some/model"))
    assert llm.model_name == "some/model"
    assert llm.temperature == 0.5
    assert llm.max_retries == 3


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        get_llm(settings=make_settings(llm_provider="nope"))
