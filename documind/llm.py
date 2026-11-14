"""The ONE place that creates a chat model.

WHY: every other file calls get_llm() instead of building ChatGroq itself.
Switching to Gemini or Ollama later means changing only this file (NFR7).
"""

from langchain_core.language_models.chat_models import BaseChatModel

from documind.config import Settings, get_settings


class MissingApiKeyError(RuntimeError):
    """Raised when a real LLM is requested but no API key is configured."""


def get_llm(temperature: float = 0.0, settings: Settings | None = None) -> BaseChatModel:
    """Return a ready-to-use chat model.

    temperature=0 by default: we want repeatable answers for study notes.
    Demos that teach temperature pass other values explicitly.
    """
    settings = settings or get_settings()

    if settings.llm_provider == "groq":
        if settings.groq_api_key is None:
            raise MissingApiKeyError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your free key "
                "from https://console.groq.com/keys"
            )
        # Imported here so tests that never need Groq don't import it.
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.llm_model,
            temperature=temperature,
            api_key=settings.groq_api_key,
            # WHY: the free tier returns HTTP 429 when you go too fast;
            # the client waits and retries instead of crashing (NFR6).
            max_retries=settings.llm_max_retries,
        )

    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")
