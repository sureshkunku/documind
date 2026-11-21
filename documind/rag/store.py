"""Store note chunks in ChromaDB and search them by meaning.

CONCEPTS
- VECTOR DATABASE: stores each chunk's text + metadata + embedding, and can
  quickly find the chunks whose embeddings are closest to a question's.
- ChromaDB runs inside your Python process and saves to the .chroma/ folder,
  so the index survives restarts. No server, no account, free.
- INCREMENTAL INDEXING: we compare each file's hash with the hash stored in
  its chunks. Only new or changed files are embedded again; deleted files
  are removed. Embedding is the slow part, so this matters as notes grow.
- RELEVANCE SCORE: 1.0 = same meaning, ~0 = unrelated (cosine similarity,
  exactly like Phase 0's d6 demo). Step 4 uses it to say "not found".
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langfuse import observe
from documind.config import get_settings
from documind.rag.loader import load_notes
from documind.rag.splitter import split_documents
from langchain_core.embeddings import Embeddings

COLLECTION = "notes"


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local MiniLM model once (loading takes a few seconds)."""
    return HuggingFaceEmbeddings(model_name=get_settings().embed_model)


def get_store(chroma_dir: Path | None = None, embeddings: Embeddings | None = None) -> Chroma:
    """embeddings: tests pass a fast fake model; normal code uses local MiniLM."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(chroma_dir or get_settings().chroma_dir),
        # WHY cosine: Chroma's default is L2 distance, whose numbers are hard
        # to read. Cosine gives the 0..1 similarity you saw in Phase 0.
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


@dataclass
class IndexReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    chunks_written: int = 0


def indexed_hashes(store: Chroma) -> dict[str, str]:
    """{source: file_hash} for every file already in the store."""
    metadatas = store.get(include=["metadatas"])["metadatas"]
    return {m["source"]: m["file_hash"] for m in metadatas}


def delete_source(store: Chroma, source: str) -> None:
    ids = store.get(where={"source": source})["ids"]
    if ids:
        store.delete(ids=ids)


def index_notes(store: Chroma, notes_dir: Path | None = None) -> IndexReport:
    """Bring the store in line with the files on disk, touching only what changed."""
    report = IndexReport()
    old = indexed_hashes(store)
    docs = load_notes(notes_dir)
    current_sources = {d.metadata["source"] for d in docs}

    for doc in docs:
        source, new_hash = doc.metadata["source"], doc.metadata["file_hash"]
        if old.get(source) == new_hash:
            report.unchanged.append(source)
            continue
        if source in old:
            delete_source(store, source)  # changed: drop old chunks first
            report.updated.append(source)
        else:
            report.added.append(source)
        chunks = split_documents([doc])
        # chunk_id as the database id: re-adding the same chunk can't duplicate it.
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        report.chunks_written += len(chunks)

    for source in sorted(old.keys() - current_sources):
        delete_source(store, source)  # file was deleted from notes/
        report.removed.append(source)
    return report


@observe(name="retrieve", as_type="retriever")
def search(store: Chroma, question: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Top-k chunks for the question, best first, with a 0..1 relevance score."""
    return store.similarity_search_with_relevance_scores(question, k=k or get_settings().top_k)


if __name__ == "__main__":
    store = get_store()
    report = index_notes(store)
    print(
        f"added={len(report.added)} updated={len(report.updated)} removed={len(report.removed)} "
        f"unchanged={len(report.unchanged)} chunks_written={report.chunks_written}"
    )

    for question in ["How do I stop my model from memorising the training data?", "What is the capital of France?"]:
        print(f"\nQ: {question}")
        for doc, score in search(store, question):
            print(f"   {score:.2f}  {doc.metadata['source']} › {doc.metadata['heading']}")