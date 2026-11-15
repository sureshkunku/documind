"""Answer questions from your notes, with citations — the full RAG loop.

CONCEPTS
- RAG = Retrieval-Augmented Generation:
    1. RETRIEVE the chunks most similar to the question (Step 3)
    2. AUGMENT the prompt by pasting those chunks in as numbered context
    3. GENERATE: the LLM answers using ONLY that context
- The LLM's own training knowledge is deliberately NOT used. That's what
  lets it answer about YOUR notes, and makes "I don't know" possible.
- CITATIONS: each chunk is numbered [1], [2]... and the LLM must cite them.
  We map the numbers back to "file › heading" so you can check the answer.
- TWO "NOT FOUND" GUARDS (either can miss on its own):
    a) best relevance score below MIN_RELEVANCE -> skip the LLM entirely
       (saves a call; catches clearly unrelated questions)
    b) the LLM replies NOT_FOUND when the context doesn't hold the answer
       (catches "related topic, but the fact isn't there")
"""

import re
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from documind.llm import get_llm
from documind.rag.store import search

# Picked from Step 3's output: relevant chunks scored ~0.3-0.4, the France
# question scored at most ~0.11. Tune it after watching real questions.
MIN_RELEVANCE = 0.3
NOT_FOUND = "NOT_FOUND"
NOT_FOUND_MESSAGE = "I couldn't find this in your notes."

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the numbered context from the student's notes.\n"
            "- Cite the context you used with its number in square brackets, e.g. [1] or [2][3].\n"
            "- Do not use any outside knowledge.\n"
            f"- If the context does not contain the answer, reply with exactly: {NOT_FOUND}",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@dataclass
class Answer:
    text: str
    found: bool
    sources: list[str] = field(default_factory=list)  # "file › heading" actually cited
    # Every retrieved chunk + score. Phase 4's RAGAS needs these "contexts".
    retrieved: list[tuple[Document, float]] = field(default_factory=list)


def citation(doc: Document) -> str:
    return f"{doc.metadata['source']} › {doc.metadata['heading']}"


def format_context(chunks: list[tuple[Document, float]]) -> str:
    """Number each chunk so the LLM can cite it: '[1] (file › heading)\\ntext'."""
    return "\n\n".join(
        f"[{i}] ({citation(doc)})\n{doc.page_content}" for i, (doc, _score) in enumerate(chunks, start=1)
    )


def cited_sources(answer_text: str, chunks: list[tuple[Document, float]]) -> list[str]:
    """Turn the [n] markers in the answer back into 'file › heading', in order, no duplicates."""
    sources = []
    for number in re.findall(r"\[(\d+)\]", answer_text):
        index = int(number) - 1
        if 0 <= index < len(chunks):  # ignore numbers the LLM made up
            source = citation(chunks[index][0])
            if source not in sources:
                sources.append(source)
    return sources


def answer_question(question: str, store: Chroma, llm: BaseChatModel | None = None) -> Answer:
    chunks = search(store, question)

    # Guard (a): nothing relevant at all -> don't even ask the LLM.
    if not chunks or chunks[0][1] < MIN_RELEVANCE:
        return Answer(text=NOT_FOUND_MESSAGE, found=False, retrieved=chunks)

    chain = PROMPT | (llm or get_llm()) | StrOutputParser()
    text = chain.invoke({"context": format_context(chunks), "question": question}).strip()

    # Guard (b): the LLM read the context and found no answer.
    if NOT_FOUND in text:
        return Answer(text=NOT_FOUND_MESSAGE, found=False, retrieved=chunks)

    return Answer(text=text, found=True, sources=cited_sources(text, chunks), retrieved=chunks)


if __name__ == "__main__":
    from documind.rag.store import get_store, index_notes

    store = get_store()
    index_notes(store)  # cheap when nothing changed

    questions = [
        "How do I stop my model from memorising the training data?",
        "What is the difference between WHERE and HAVING?",
        "Who invented the transformer architecture?",  # related topic, fact not in notes
        "What is the capital of France?",  # unrelated
    ]
    for q in questions:
        ans = answer_question(q, store)
        best = ans.retrieved[0][1] if ans.retrieved else 0.0
        print(f"\nQ: {q}   (best score {best:.2f})")
        print(f"A: {ans.text}")
        for s in ans.sources:
            print(f"   source: {s}")