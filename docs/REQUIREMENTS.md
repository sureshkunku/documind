# DocuMind — Requirements (SRS v0.1, draft)

> SDLC stage 1 of 6: **Requirements** → Design → Build (phase by phase) → Test → Evaluate → Release
> Status: **Draft — waiting for your approval.** Nothing gets built until this is approved.

## 1. Purpose

DocuMind is a **learning project**. It is a small multi-agent study assistant: you ask questions about your own study notes, and it can turn them into summaries or flashcards and save them as files.

The real goal is to **understand every core AI-engineering topic through a small, working POC**. Each phase adds one or two topics. Simplicity beats features.

## 2. Topics the project must cover

| # | Topic | Covered in |
|---|---|---|
| T1 | LLM basics: prompts, roles, temperature, tokens, streaming, structured output | Phase 0 |
| T2 | LangChain: prompt templates, LCEL chains, document loaders, text splitters | Phases 0–1 |
| T3 | Embeddings and vector DB (ChromaDB) | Phase 1 |
| T4 | RAG with source citations | Phase 1 |
| T5 | Observability with Langfuse (traces, tokens, latency, cost) | Phase 1, then everywhere |
| T6 | Tool calling | Phase 2 |
| T7 | MCP server and client (FastMCP) | Phase 2 |
| T8 | Agent Skills (`SKILL.md`, loaded on demand) | Phase 2 |
| T9 | LangGraph: state, nodes, edges, conditional routing, memory, human-in-the-loop | Phase 3 |
| T10 | Multi-agent: Supervisor, Researcher, Writer | Phase 3 |
| T11 | RAG evaluation with RAGAS | Phase 4 |
| T12 | Agent evaluation (DeepEval and/or Langfuse datasets) | Phase 4 |
| T13 | *Optional:* Streamlit chat UI | Phase 5 |

## 3. Example user stories

- **US1, ask:** "What is backpropagation?" → an answer drawn **only from my notes**, with the source file and section named.
- **US2, flashcards:** "Make flashcards from my Neural Networks notes and save them to cards.md." → the system researches, writes, **asks me to approve**, then saves.
- **US3, summary:** "Summarize my SQL notes in 5 bullets and save the summary." → a summary file in `notes/output/`.
- **US4, fact check:** "Check my note on transformers against Wikipedia." → the answer says what matches and what differs, with links.
- **US5, not in notes:** a question my notes don't cover → it says "I couldn't find this in your notes" and doesn't make anything up.
- **US6, debug:** I can open Langfuse and see every agent step, tool call, token count and timing for any request.
- **US7, quality:** I can run one command that prints RAGAS and agent-eval scores.

## 4. Functional requirements

### Phase 0: LLM basics
- **FR0.1** Call the Groq LLM with a system and user prompt, and print the reply plus token usage.
- **FR0.2** Show how temperature changes output (same prompt, 3 temperatures).
- **FR0.3** Stream a response token by token.
- **FR0.4** Get structured output into a Pydantic model (for example, a question → `{topic, difficulty, keywords}`).
- **FR0.5** Build a LangChain LCEL chain: `prompt | llm | parser`.
- **FR0.6** Embed 5 sentences and print a similarity table, to show what "similar meaning" looks like as numbers.

### Phase 1: RAG + Langfuse
- **FR1.1** Load `.md` and `.pdf` files from `notes/`.
- **FR1.2** Split them into chunks, keeping the source file and heading as metadata.
- **FR1.3** Embed chunks locally (`all-MiniLM-L6-v2`) into a persistent ChromaDB.
- **FR1.4** Re-indexing only processes new or changed files.
- **FR1.5** Answer a question using the top-k chunks, **citing sources**.
- **FR1.6** If nothing relevant is retrieved, reply "not found in your notes" (US5).
- **FR1.7** Every run is traced in Langfuse.

### Phase 2: Tools, MCP, Skills
- **FR2.1** An MCP server (FastMCP) exposes `list_notes()`, `read_note(name)`, `save_note(name, text)` and `wiki_search(query)`.
- **FR2.2** File tools only work **inside `notes/`**. Paths like `../` or absolute paths are rejected.
- **FR2.3** `save_note` only writes into `notes/output/` and never overwrites a file without `overwrite=true`.
- **FR2.4** An LLM agent connects to the MCP server and picks the right tool for a request.
- **FR2.5** The same MCP server can be added to Claude Desktop (manual check).
- **FR2.6** Skills live in `skills/<name>/SKILL.md`. Two to start: **flashcards** and **summary**.
- **FR2.7** At startup the agent sees only each skill's name and description, and loads the full skill text **only when it picks that skill**.

### Phase 3: LangGraph multi-agent
- **FR3.1** A LangGraph `StateGraph` with a shared state: question, plan, retrieved chunks, draft, files written, messages.
- **FR3.2** The **Supervisor** decides the next step: Researcher, Writer, or finish.
- **FR3.3** The **Researcher** uses RAG (and `wiki_search` for fact checks).
- **FR3.4** The **Writer** uses Skills and the MCP `save_note` tool.
- **FR3.5** **Human-in-the-loop:** the graph pauses before any file write and waits for yes or no.
- **FR3.6** Conversation memory per thread (checkpointer), so a follow-up like "make it shorter" works.
- **FR3.7** A loop guard: at most 6 steps per request, so agents can't loop forever.
- **FR3.8** Supervisor, Researcher and Writer appear as separate spans in Langfuse.

### Phase 4: Evaluation
- **FR4.1** A test set in `evals/dataset.json`: 10 questions with expected answers and the expected source file. At least 2 of them are "not in notes" cases.
- **FR4.2** RAGAS reports **faithfulness, answer relevancy, context precision and context recall**.
- **FR4.3** Agent evals check correct routing, the right tool called (for example `save_note` for flashcard requests), tools in the right order, and no tool calls for plain questions.
- **FR4.4** A single command, `python -m evals.run`, prints a scores table and saves it to `evals/results/`.
- **FR4.5** Scores are also pushed to Langfuse.

### Phase 5 (optional): UI
- **FR5.1** A Streamlit chat page with approve/reject buttons for file writes and a link to each trace.

## 5. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR1 | **Free:** Groq free tier, local embeddings, local ChromaDB, Langfuse Cloud free tier. No paid services. |
| NFR2 | **Simple:** each phase's main script is about 150 lines or fewer, with "why" comments explaining each concept. |
| NFR3 | **Runnable on its own:** each phase runs with one command and has a README section saying what it teaches and suggesting 2–3 exercises. |
| NFR4 | **Tested:** pytest tests for every phase. Unit tests mock the LLM, so they run offline and for free. Real-LLM tests are opt-in. |
| NFR5 | **Secrets:** API keys live only in `.env`, which is git-ignored. A `.env.example` lists the variable names only. |
| NFR6 | **Rate limits:** Groq free-tier limits are handled with retry and backoff. Evals run sequentially. |
| NFR7 | **Swappable LLM:** the model name and provider are set in config, so switching to Gemini or Ollama later is a one-line change. |
| NFR8 | **Windows-friendly:** works on Windows 11 with Python 3.11+ and `uv`. |

## 6. Tech stack

| Area | Choice |
|---|---|
| Language / env | Python 3.11+, `uv` |
| LLM | Groq (Llama model with tool calling) via `langchain-groq` |
| Framework | LangChain, LangGraph |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector DB | ChromaDB (persistent, local) |
| MCP | `fastmcp` server + `langchain-mcp-adapters` client |
| Observability | Langfuse Cloud (free tier) |
| Evaluation | RAGAS, DeepEval |
| Tests | pytest |
| UI (optional) | Streamlit |

## 7. Data

- `notes/samples/`: 4 short sample notes I'll write (Neural Networks, Python Basics, SQL Basics, Transformers). These make tests and eval scores repeatable.
- `notes/mine/`: your own PDFs and markdown, which are not committed to git.
- `notes/output/`: files the agents write.

## 8. Out of scope (for now)

- User accounts, login, multiple users
- Cloud deployment and CI/CD
- Paid LLMs
- Web scraping (only the Wikipedia API)
- Fine-tuning models

## 9. Acceptance criteria (what "done" means)

| Phase | Done when |
|---|---|
| 0 | All 6 demos run, tests pass, and you can explain tokens, temperature, structured output and embeddings |
| 1 | US1 and US5 work with citations, and the trace is visible in Langfuse |
| 2 | The agent picks the correct MCP tool, path escape is blocked (tested), a skill loads only when used, and the server works in Claude Desktop |
| 3 | US2, US3 and US4 work end to end, the approval pause works, and the follow-up "make it shorter" works |
| 4 | One command prints RAGAS and agent-eval scores. Target: faithfulness ≥ 0.8 and routing accuracy ≥ 80% on the sample notes |

## 10. What you need to set up

1. A free **Groq** API key at console.groq.com
2. A free **Langfuse Cloud** account and project keys at cloud.langfuse.com
3. Python 3.11+ and `uv` installed

Keys go into `.env` on your machine. Never paste them in chat.

## 11. Open questions

- Should **US4 (fact check against Wikipedia)** stay in v1, or move to "later"?
- Do you want **PDF support** from Phase 1, or start with markdown only (simpler) and add PDFs later?
