"""Phase 1 tests: loader, splitter, store, Q&A, tracing.

WHY these fakes:
- DeterministicFakeEmbedding: same text -> same random-looking vector,
  instantly, without loading MiniLM. Enough to test indexing logic.
- FakeListChatModel: replies we choose, no Groq call.
- Every test gets its own tmp_path folder, so tests never touch your
  real notes/ or .chroma/.
"""

import os

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from pypdf import PdfWriter

from documind import tracing
from documind.rag import qa
from documind.rag.loader import find_note_files, load_notes
from documind.rag.splitter import CHUNK_SIZE, heading_path, split_document
from documind.rag.store import get_store, index_notes, search


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def notes_dir(tmp_path):
    """A tiny notes folder: 2 real notes + things the loader must skip."""
    root = tmp_path / "notes"
    write(root / "samples" / "sql.md", "# SQL\n\n## Joins\nINNER JOIN matches rows.\n\n## Indexes\nIndexes speed up reads.\n")
    write(root / "samples" / "python.md", "# Python\n\n## Lists\nLists are mutable.\n")
    write(root / "output" / "cards.md", "# Agent output\nmust NOT be indexed\n")
    write(root / "samples" / "todo.txt", "not a supported type")
    write(root / "samples" / "empty.md", "   \n")
    return root


@pytest.fixture
def store(tmp_path):
    return get_store(chroma_dir=tmp_path / "chroma", embeddings=DeterministicFakeEmbedding(size=32))


# ---------- loader ----------

def test_loader_finds_md_and_skips_output_and_other_types(notes_dir):
    names = [p.relative_to(notes_dir).as_posix() for p in find_note_files(notes_dir)]
    assert names == ["samples/empty.md", "samples/python.md", "samples/sql.md"]  # sorted, no output/, no .txt


def test_loader_skips_empty_files_and_sets_metadata(notes_dir):
    docs = load_notes(notes_dir)
    assert [d.metadata["source"] for d in docs] == ["samples/python.md", "samples/sql.md"]
    assert all(len(d.metadata["file_hash"]) == 16 for d in docs)


def test_file_hash_changes_only_when_content_changes(notes_dir):
    before = {d.metadata["source"]: d.metadata["file_hash"] for d in load_notes(notes_dir)}
    write(notes_dir / "samples" / "sql.md", "# SQL\n\n## Joins\nchanged text\n")
    after = {d.metadata["source"]: d.metadata["file_hash"] for d in load_notes(notes_dir)}
    assert after["samples/sql.md"] != before["samples/sql.md"]
    assert after["samples/python.md"] == before["samples/python.md"]


def test_pdf_without_text_is_skipped(notes_dir):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # like a scanned page: no text layer
    with open(notes_dir / "samples" / "scan.pdf", "wb") as f:
        writer.write(f)
    assert "samples/scan.pdf" not in [d.metadata["source"] for d in load_notes(notes_dir)]


# ---------- splitter ----------

def test_heading_path():
    assert heading_path({"h1": "SQL", "h2": "Joins"}) == "SQL › Joins"
    assert heading_path({"h1": "SQL"}) == "SQL"
    assert heading_path({}) == ""


def test_markdown_splits_per_heading_with_metadata(notes_dir):
    doc = next(d for d in load_notes(notes_dir) if d.metadata["source"] == "samples/sql.md")
    chunks = split_document(doc)
    assert [c.metadata["heading"] for c in chunks] == ["SQL › Joins", "SQL › Indexes"]
    assert [c.metadata["chunk_id"] for c in chunks] == ["samples/sql.md#0", "samples/sql.md#1"]
    assert all(c.metadata["file_hash"] == doc.metadata["file_hash"] for c in chunks)  # carried over
    assert "## Joins" in chunks[0].page_content  # strip_headers=False keeps the heading words


def test_long_section_is_cut_into_pieces_within_chunk_size():
    long_text = "# T\n\n## Long\n" + ("word " * 400)  # ~2000 characters
    chunks = split_document(Document(page_content=long_text, metadata={"source": "long.md", "file_hash": "x"}))
    assert len(chunks) > 1
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)  # ids unique


def test_pdf_text_is_split_by_size_only():
    chunks = split_document(Document(page_content="plain pdf text", metadata={"source": "a.pdf", "file_hash": "x"}))
    assert chunks[0].metadata["heading"] == ""


# ---------- store: incremental indexing ----------

def count_chunks(store, source):
    return len(store.get(where={"source": source})["ids"])


def test_first_index_adds_everything(store, notes_dir):
    report = index_notes(store, notes_dir)
    assert report.added == ["samples/python.md", "samples/sql.md"]
    assert report.chunks_written == 3
    assert count_chunks(store, "samples/sql.md") == 2


def test_second_index_changes_nothing(store, notes_dir):
    index_notes(store, notes_dir)
    report = index_notes(store, notes_dir)
    assert report.unchanged == ["samples/python.md", "samples/sql.md"]
    assert report.chunks_written == 0


def test_changed_file_is_replaced_not_duplicated(store, notes_dir):
    index_notes(store, notes_dir)
    write(notes_dir / "samples" / "sql.md", "# SQL\n\n## Joins\nonly one section now\n")
    report = index_notes(store, notes_dir)
    assert report.updated == ["samples/sql.md"]
    assert count_chunks(store, "samples/sql.md") == 1  # old 2 chunks gone, 1 new


def test_deleted_file_is_removed(store, notes_dir):
    index_notes(store, notes_dir)
    (notes_dir / "samples" / "python.md").unlink()
    report = index_notes(store, notes_dir)
    assert report.removed == ["samples/python.md"]
    assert count_chunks(store, "samples/python.md") == 0


# Fake vectors are random, so cosine can dip below 0 and LangChain warns. Expected here.
@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1")
def test_search_returns_k_chunks_with_scores_and_metadata(store, notes_dir):
    index_notes(store, notes_dir)
    results = search(store, "joins", k=2)
    assert len(results) == 2
    for doc, score in results:
        assert {"source", "heading", "chunk_id"} <= doc.metadata.keys()
        assert isinstance(score, float)


# ---------- Q&A ----------

def chunk(source, heading, text="some text"):
    return Document(page_content=text, metadata={"source": source, "heading": heading})


@pytest.fixture
def fake_search(monkeypatch):
    """Replace search() inside qa with chunks we choose. Returns a setter."""
    def set_results(results):
        monkeypatch.setattr(qa, "search", lambda store, question: results)
    return set_results


def exploding_llm():
    """An 'LLM' that fails the test if it is ever called."""
    def boom(_):
        raise AssertionError("LLM should not have been called")
    return RunnableLambda(boom)


def test_low_score_returns_not_found_without_calling_llm(fake_search):
    fake_search([(chunk("a.md", "A"), 0.05)])
    answer = qa.answer_question("capital of France?", store=None, llm=exploding_llm())
    assert answer.found is False
    assert answer.text == qa.NOT_FOUND_MESSAGE


def test_no_chunks_returns_not_found(fake_search):
    fake_search([])
    assert qa.answer_question("anything", store=None, llm=exploding_llm()).found is False


def test_llm_not_found_reply_returns_not_found(fake_search):
    fake_search([(chunk("t.md", "Transformers"), 0.43)])
    answer = qa.answer_question("who invented it?", store=None, llm=FakeListChatModel(responses=["NOT_FOUND"]))
    assert answer.found is False
    assert answer.retrieved  # chunks kept for debugging / evals


def test_found_answer_maps_citations_to_sources(fake_search):
    fake_search([(chunk("sql.md", "Where"), 0.30), (chunk("sql.md", "Having"), 0.25)])
    llm = FakeListChatModel(responses=["WHERE filters rows [1], HAVING filters groups [2][1]."])
    answer = qa.answer_question("where vs having?", store=None, llm=llm)
    assert answer.found is True
    assert answer.sources == ["sql.md › Where", "sql.md › Having"]  # cited order, no duplicates


def test_made_up_citation_numbers_are_ignored():
    chunks = [(chunk("a.md", "A"), 0.5)]
    assert qa.cited_sources("see [1] and [7]", chunks) == ["a.md › A"]


def test_format_context_numbers_chunks_from_one():
    text = qa.format_context([(chunk("a.md", "A", "alpha"), 0.5), (chunk("b.md", "B", "beta"), 0.4)])
    assert text.startswith("[1] (a.md › A)\nalpha")
    assert "[2] (b.md › B)\nbeta" in text


# ---------- tracing ----------

def test_tracing_off_without_keys():
    # conftest blanked the keys
    assert tracing.get_langfuse() is None
    assert tracing.get_langfuse_handler() is None
    assert tracing.langchain_callbacks() == []
    assert os.environ["LANGFUSE_TRACING_ENABLED"] == "false"
    tracing.flush()  # must not crash when tracing is off


# ---------- live (opt-in: uv run pytest -m live) ----------

@pytest.mark.live
def test_live_rag_on_sample_notes(tmp_path):
    from documind.config import get_settings

    store = get_store(chroma_dir=tmp_path / "chroma")  # real MiniLM embeddings
    index_notes(store, get_settings().notes_dir)

    found = qa.answer_question("How do I stop my model from memorising the training data?", store)
    assert found.found
    assert any("Overfitting" in s for s in found.sources)

    assert qa.answer_question("What is the capital of France?", store).found is False