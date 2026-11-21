"""Langfuse tracing: see every step of every request in a web dashboard.

CONCEPTS
- TRACE: one request from start to finish (e.g. one question answered).
- SPAN / OBSERVATION: one step inside a trace (retrieve, LLM call...).
  Spans nest, so a trace reads like a tree:
      answer_question
      ├── retrieve          (which chunks, what scores)
      └── generate          (LangChain chain)
          └── ChatGroq      (exact prompt sent, reply, tokens, cost, latency)
- Two ways to create spans, both used in Phase 1:
    @observe            -> decorate any Python function you wrote
    CallbackHandler     -> pass to LangChain; it records chains and LLM calls
- Traces are sent in the BACKGROUND in batches. A short script must call
  flush() before exiting, or the last traces are lost.
- No keys in .env -> tracing is off and everything still runs (tests too).
"""

import os
from functools import lru_cache

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from documind.config import get_settings


@lru_cache
def get_langfuse() -> Langfuse | None:
    """Create the Langfuse client once, or None when keys are missing."""
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        # WHY: @observe would otherwise create its own keyless client that
        # keeps trying (and failing) to upload. This switches it off cleanly.
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
        return None
    # WHY pass keys explicitly: Langfuse normally reads os.environ, but our
    # keys live in .env, which only our Settings class reads.
    return Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key.get_secret_value(),
        host=s.langfuse_host,
    )


def get_langfuse_handler() -> CallbackHandler | None:
    """LangChain callback that records chains and LLM calls, or None if tracing is off."""
    if get_langfuse() is None:
        return None
    return CallbackHandler(public_key=get_settings().langfuse_public_key)


def langchain_callbacks() -> list:
    """Ready to use as config={"callbacks": langchain_callbacks()}."""
    handler = get_langfuse_handler()
    return [handler] if handler else []


def flush() -> None:
    """Send any queued traces now. Call at the end of short scripts."""
    client = get_langfuse()
    if client is not None:
        client.flush()


if __name__ == "__main__":
    client = get_langfuse()
    if client is None:
        print("Tracing is OFF: add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to .env")
    else:
        # auth_check() makes one real call to Langfuse to verify the keys.
        try:
            ok = client.auth_check()
        except Exception as error:  # the SDK raises on HTTP 401 instead of returning False
            ok = False
            print(f"Langfuse said: {type(error).__name__}")
        host = get_settings().langfuse_host
        if ok:
            print(f"Langfuse keys OK ({host})")
        else:
            print(f"Langfuse keys REJECTED by {host}. Copy BOTH keys from the same new key pair, "
                  "and use the host shown on that page.")