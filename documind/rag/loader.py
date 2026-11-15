"""Load study notes (.md and .pdf) into LangChain Documents.

CONCEPTS
- A Document is just text + metadata (a dict). Metadata travels with the
  text through every RAG step, so at the end we can say WHERE an answer
  came from (citations).
- We also store a hash (fingerprint) of each file. If the file changes,
  the hash changes, so Step 3 can re-index only changed files.
"""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from documind.config import get_settings

SUPPORTED_SUFFIXES = {".md", ".pdf"}
# WHY skip "output": that's where agents save their own files (Phase 2+).
# Indexing them would make the AI quote its own summaries back to you.
SKIPPED_FOLDERS = {"output"}


def file_hash(path: Path) -> str:
    """Short fingerprint of the file's bytes. Same bytes -> same hash."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        # extract_text() returns None for pages with no text (e.g. scanned images).
        pages = [page.extract_text() or "" for page in PdfReader(path).pages]
        return "\n\n".join(pages)
    return path.read_text(encoding="utf-8")


def load_file(path: Path, notes_dir: Path) -> Document:
    return Document(
        page_content=read_text(path),
        metadata={
            # relative path, e.g. "samples/sql_basics.md" — unique and readable
            "source": path.relative_to(notes_dir).as_posix(),
            "file_hash": file_hash(path),
        },
    )


def find_note_files(notes_dir: Path) -> list[Path]:
    """All supported files under notes_dir, skipping agent output. Sorted = stable order."""
    files = []
    for path in notes_dir.rglob("*"):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if SKIPPED_FOLDERS & set(path.relative_to(notes_dir).parts):
            continue
        files.append(path)
    return sorted(files)


def load_notes(notes_dir: Path | None = None) -> list[Document]:
    notes_dir = notes_dir or get_settings().notes_dir
    docs = [load_file(p, notes_dir) for p in find_note_files(notes_dir)]
    # Skip empty files (e.g. a scanned PDF with no text layer).
    return [d for d in docs if d.page_content.strip()]


if __name__ == "__main__":
    for doc in load_notes():
        print(f"{doc.metadata['source']:<35} {len(doc.page_content):>5} chars  hash={doc.metadata['file_hash']}")