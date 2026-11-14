# DocuMind

A small multi-agent study assistant over your own notes, built **one AI topic at a time** as a learning project.

- Requirements: [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)
- Design: [docs/DESIGN.md](docs/DESIGN.md)

| Phase | Topic | Status |
|---|---|---|
| 0 | LLM basics, LangChain, embeddings | ✅ built |
| 1 | RAG + Langfuse | next |
| 2 | MCP + Skills | |
| 3 | LangGraph multi-agent | |
| 4 | RAGAS + agent evals | |
| 5 | Streamlit UI (optional) | |

## Setup (once)

1. Install Python 3.11+ and [uv](https://docs.astral.sh/uv/).
2. Install dependencies (the first run downloads PyTorch, so it takes a few minutes):
   ```bash
   uv sync
   ```
3. Get a free Groq API key at https://console.groq.com/keys.
4. Copy `.env.example` to `.env` and put your key after `GROQ_API_KEY=`. `.env` is git-ignored, so the key never gets committed.

## Phase 0: LLM basics

Six demos, each one short file. Read the `CONCEPTS` comment at the top of each, run it, then try the `TRY THIS` ideas at the bottom.

| Demo | You learn | Run |
|---|---|---|
| d1 | Messages and roles, no memory between calls, tokens | `uv run python -m documind.phase0_basics.d1_first_call` |
| d2 | Temperature: repeatable vs creative answers | `uv run python -m documind.phase0_basics.d2_temperature` |
| d3 | Streaming vs waiting for the whole answer | `uv run python -m documind.phase0_basics.d3_streaming` |
| d4 | Structured output into a Pydantic object | `uv run python -m documind.phase0_basics.d4_structured` |
| d5 | Prompt templates and LCEL chains (`prompt \| llm \| parser`), batch | `uv run python -m documind.phase0_basics.d5_lcel_chain` |
| d6 | Embeddings and cosine similarity (local, no key needed) | `uv run python -m documind.phase0_basics.d6_embeddings` |

**Shared code used by every later phase:**
- `documind/config.py`: all settings, read from `.env`
- `documind/llm.py`: `get_llm()`, the only place a chat model is created

## Tests

```bash
uv run pytest
```
Runs offline with a fake LLM: free and fast.

```bash
uv run pytest -m live
```
Also runs the real Groq call and the real embedding model. Needs `.env`.
