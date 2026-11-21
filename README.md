# DocuMind

A small multi-agent study assistant over your own notes, built **one AI topic at a time** as a learning project.

- Requirements: [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)
- Design: [docs/DESIGN.md](docs/DESIGN.md)

| Phase | Topic | Status |
|---|---|---|
| 0 | LLM basics, LangChain, embeddings | ✅ built |
| 1 | RAG + Langfuse | ✅ built |
| 2 | MCP + Skills | next |
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
5. *(Optional, for tracing)* Create a free project at https://cloud.langfuse.com, then **Settings → API Keys → Create**. Copy all three values from that one dialog into `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://us.cloud.langfuse.com
   ```
   Use the host shown on that page: EU projects use `https://cloud.langfuse.com`. Langfuse may label it `BASE_URL`, but in `.env` it must be `LANGFUSE_HOST`. Without keys, tracing is off and everything else still works.

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

## Phase 1: RAG + Langfuse

Ask questions about your notes. Answers come **only** from the notes, with citations, and the system says "not found" instead of making things up.

```
notes/*.md, *.pdf → loader → splitter → MiniLM embeddings → ChromaDB (.chroma/)
question → embed → top-4 chunks → numbered context → LLM → answer + [n] citations
```

| File | You learn | Run on its own |
|---|---|---|
| `rag/loader.py` | Documents = text + metadata; file hashes | `uv run python -m documind.rag.loader` |
| `rag/splitter.py` | Chunking by heading, then by size, with overlap | `uv run python -m documind.rag.splitter` |
| `rag/store.py` | Vector DB, incremental indexing, relevance scores | `uv run python -m documind.rag.store` |
| `rag/qa.py` | The full RAG loop, citations, hallucination guards | `uv run python -m documind.rag.qa` |
| `tracing.py` | Traces and spans, `@observe` + LangChain callbacks | `uv run python -m documind.tracing` (checks your keys) |

**Your own notes:** put `.md` or `.pdf` files in `notes/mine/`, which is git-ignored. `qa.py` re-indexes only new or changed files.

### Citations

A citation says **where an answer came from**, so you can check it instead of trusting it.

```
A: Use dropout, early stopping and more training data [1].
   source: samples/neural_networks.md › Neural Networks › Overfitting
```

How `qa.py` does it:
1. `format_context()` numbers each retrieved chunk: `[1] (file › heading)`, then the text.
2. The prompt tells the LLM to cite the numbers it used, e.g. `[1]` or `[2][3]`.
3. `cited_sources()` turns each `[n]` back into `file › heading`, keeping the order and dropping duplicates.

Why it matters:
- **Verify:** open the cited file and section, and check the answer is really there.
- **Spot hallucinations:** an answer without `[n]` wasn't grounded in your notes.
- **No fake sources:** numbers the LLM invents, like `[7]` when only 4 chunks were sent, are ignored.
- **Debug:** a wrong answer citing `[2]` points straight at chunk 2. Either the note was wrong, or the wrong chunk was retrieved.

### Hallucination guards

All four layers live in `documind/rag/qa.py`. Each one catches something the others miss.

| Layer | Where | Stops | Limit |
|---|---|---|---|
| 1. Strict prompt | `PROMPT` | Answering from the LLM's own memory ("ONLY the numbered context", "no outside knowledge") | Only an instruction; the LLM usually obeys but isn't forced to |
| 2. Relevance score | Guard (a) in `answer_question` | Unrelated questions: best score below `MIN_RELEVANCE` (0.2), so **the LLM is never called** | Can't catch "same topic, missing fact" |
| 3. LLM says `NOT_FOUND` | Guard (b) in `answer_question` | Related topic but the fact isn't in the notes, e.g. "Who invented the transformer?" (score 0.43, but the inventors aren't named) | Relies on the LLM noticing |
| 4. Citation check | `cited_sources()` | Made-up source numbers | Doesn't check the answer matches the chunk |

```
question
   │
   ▼
search → best score < 0.2 ? ──yes──► "not found"   (layer 2: LLM never called)
   │ no
   ▼
LLM + strict prompt                                (layer 1)
   │
   ▼
reply contains NOT_FOUND ? ──yes──► "not found"    (layer 3)
   │ no
   ▼
answer + only real [n] sources                     (layer 4)
```

Tests that prove each guard, in `tests/test_rag.py`:
- Layer 2: `test_low_score_returns_not_found_without_calling_llm` uses an "exploding" LLM that fails the test if it's called.
- Layer 3: `test_llm_not_found_reply_returns_not_found`
- Layer 4: `test_made_up_citation_numbers_are_ignored`

**Still missing:** none of these check that a "found" answer is actually *correct*. The LLM could cite `[1]` and still misstate it. Phase 4's RAGAS **faithfulness** score checks every claim in the answer against the cited chunks.

### What I learned building it
- gpt-oss sometimes ignores function-calling structured output, so use `method="json_schema"`.
- Small embedding models rank imperfectly, which is why we send the top 4 chunks, not 1.
- `MIN_RELEVANCE` is a trade-off: 0.3 rejected a real question (WHERE vs HAVING, 0.23). Phase 4's evals will tune it.
- Langfuse rejects keys sent to the wrong region, so check the host first.

## Tests

```bash
uv run pytest
```
39 offline tests: fake LLM, fake embeddings, temporary folders. Langfuse keys are blanked by `tests/conftest.py`, so tests never send traces. Free, fast and private.

```bash
uv run pytest -m live
```
3 live tests: a real Groq call, real MiniLM embeddings and an end-to-end RAG run on `notes/samples/`. Needs `GROQ_API_KEY` in `.env`, and takes about 40 seconds.
