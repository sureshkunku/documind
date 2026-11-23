"""All settings in one place, read from environment variables / the .env file.

WHY a Settings class instead of os.getenv() everywhere:
- one place to see every knob the app has
- types are checked (top_k must be an int, etc.)
- secrets come from .env, never from code
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated variables in .env
    )

    # --- LLM ---
    # SecretStr hides the key if settings are ever printed or logged.
    groq_api_key: SecretStr | None = None
    llm_provider: str = "groq"
    # WHY gpt-oss: Groq retired its Llama 3.x models in 2026; gpt-oss-120b is a
    # production model there with tool calling and structured output.
    # Check https://console.groq.com/docs/models if this ever stops working.
    llm_model: str = "openai/gpt-oss-120b"
    llm_max_retries: int = 3  # free tier has rate limits -> retry with backoff

    # --- Embeddings ---
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Paths & retrieval (used from Phase 1) ---
    notes_dir: Path = PROJECT_ROOT / "notes"
    chroma_dir: Path = PROJECT_ROOT / ".chroma"
    top_k: int = 4
    skills_dir: Path = PROJECT_ROOT / "skills"  # Phase 2: Agent Skills
    # --- Tracing (Phase 1+). Both keys missing = tracing off, app still works. ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Agents (used from Phase 3) ---
    max_graph_steps: int = 6
    max_agent_steps: int = 8  # Phase 2: stop a tool-calling loop that never finishes
    # --- Wikipedia (Phase 2). Their robot policy requires contact info in
    # the User-Agent; without it, requests get HTTP 403. ---
    wiki_contact: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Cached so .env is read once, not on every call."""
    return Settings()
