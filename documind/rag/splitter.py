"""Split notes into small chunks for embedding.

CONCEPTS
- CHUNK: a small piece of a note, ideally about ONE idea. One idea -> a
  sharp embedding -> better search. A whole file -> a blurry average.
- Pass 1 splits markdown by heading, following the note's own structure.
  Pass 2 cuts any section still too long into ~800-character pieces.
- OVERLAP: neighbouring pieces share 100 characters, so a sentence cut at
  a boundary is still complete in one of them.
- Every chunk keeps its metadata (source, file_hash) and gains a
  "heading" for citations and a "chunk_id" so we can find it again.
"""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# "#" lines become metadata["h1"], "##" lines become metadata["h2"].
HEADERS = [("#", "h1"), ("##", "h2")]

# WHY strip_headers=False: keep "## Joins" inside the chunk text too, so the
# heading words help the embedding match questions like "what is a join?".
_by_heading = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
# Tries to cut at paragraphs, then lines, then spaces, before cutting mid-word.
_by_size = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def heading_path(metadata: dict) -> str:
    """{'h1': 'SQL Basics', 'h2': 'Joins'} -> 'SQL Basics › Joins'."""
    return " › ".join(metadata[key] for key in ("h1", "h2") if metadata.get(key))


def split_document(doc: Document) -> list[Document]:
    """One loaded note -> many chunks, each with source, heading and chunk_id."""
    if doc.metadata["source"].endswith(".md"):
        sections = _by_heading.split_text(doc.page_content)
    else:
        # PDFs have no markdown headings: treat the whole text as one section.
        sections = [Document(page_content=doc.page_content)]

    chunks = []
    for section in sections:
        heading = heading_path(section.metadata)
        for piece in _by_size.split_text(section.page_content):
            chunks.append(Document(page_content=piece, metadata={**doc.metadata, "heading": heading}))

    # chunk_id like "samples/sql_basics.md#2": unique, readable, stable.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}#{i}"
    return chunks


def split_documents(docs: list[Document]) -> list[Document]:
    return [chunk for doc in docs for chunk in split_document(doc)]


if __name__ == "__main__":
    from documind.rag.loader import load_notes

    chunks = split_documents(load_notes())
    print(f"{len(chunks)} chunks\n")
    for c in chunks:
        first_line_after_heading = c.page_content.split("\n", 1)[-1][:50]
        print(f"{c.metadata['chunk_id']:<28} {len(c.page_content):>4} chars  [{c.metadata['heading']}]  {first_line_after_heading}...")