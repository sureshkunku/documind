"""Demo 3 — streaming.

CONCEPTS
- invoke() waits for the WHOLE answer, then returns it.
- stream() returns pieces (chunks) as soon as the model produces them.
- Same total time, but the user sees text immediately — that's why every
  chat app streams. The answer is still made one token after another.

Run:  uv run python -m documind.phase0_basics.d3_streaming
"""

import time
from collections.abc import Iterator

from langchain_core.language_models.chat_models import BaseChatModel

from documind.llm import get_llm


def stream_answer(llm: BaseChatModel, question: str) -> Iterator[str]:
    """Yield the answer text piece by piece."""
    for chunk in llm.stream(question):
        # Some chunks carry no text (e.g. the final one with token counts).
        if chunk.content:
            yield str(chunk.content)


def main() -> None:
    llm = get_llm()
    question = "Explain gradient descent in 5 short sentences."
    print(f"Q: {question}\nA: ", end="")

    start = time.perf_counter()
    first_piece_at = None
    pieces = 0
    for piece in stream_answer(llm, question):
        if first_piece_at is None:
            first_piece_at = time.perf_counter() - start
        print(piece, end="", flush=True)  # flush = show it NOW, don't buffer
        pieces += 1
    total = time.perf_counter() - start

    print(f"\n\nfirst text after {first_piece_at or 0:.2f}s, finished after {total:.2f}s, {pieces} chunks")
    # TRY THIS:
    # 1. Replace stream_answer with llm.invoke(question) and notice the wait.
    # 2. Print each piece inside [brackets] to see how the text is split up.


if __name__ == "__main__":
    main()
