"""Demo 4 — structured output.

CONCEPTS
- Normally an LLM returns free text. Code can't reliably use free text.
- STRUCTURED OUTPUT: you describe the shape you want as a Pydantic model,
  and the model fills it in. LangChain sends the schema to the LLM and
  validates the reply, so you get a real Python object back.
- Phase 3's Supervisor uses exactly this to decide which agent runs next.

Run:  uv run python -m documind.phase0_basics.d4_structured
"""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from documind.llm import get_llm


class QuestionInfo(BaseModel):
    """What a study question is about. The docstring and the field
    descriptions are sent to the LLM — they ARE the instructions."""

    topic: str = Field(description="Main subject, 1-3 words, e.g. 'neural networks'")
    difficulty: Literal["beginner", "intermediate", "advanced"] = Field(
        description="How hard the question is for a student"
    )
    keywords: list[str] = Field(description="3-5 key terms needed to answer it")


# WHY a system prompt: without it the model sees "What is a for loop?" and
# simply ANSWERS it. We must say its job is to describe the question.
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You classify study questions. Do NOT answer the question; only describe it."),
        ("human", "{question}"),
    ]
)


def make_classifier() -> Runnable:
    """prompt -> LLM that returns QuestionInfo objects instead of text.

    WHY method="json_schema": the default ("function_calling") only ASKS the
    model to call a tool, and gpt-oss sometimes just answers in prose instead
    (Groq error 400 "tool_use_failed"). json_schema makes Groq constrain the
    reply to the schema's JSON shape, so the model can't wander off.
    """
    return PROMPT | get_llm().with_structured_output(QuestionInfo, method="json_schema")


def classify(classifier: Runnable, question: str) -> QuestionInfo:
    result = classifier.invoke({"question": question})
    # Pydantic already validated it; this guards against a wrong classifier.
    if not isinstance(result, QuestionInfo):
        raise TypeError(f"expected QuestionInfo, got {type(result).__name__}")
    return result


def main() -> None:
    classifier = make_classifier()
    questions = [
        "What is a for loop in Python?",
        "How does multi-head attention differ from single-head attention?",
        "Write a SQL query that uses a window function to rank sales by region.",
    ]
    for q in questions:
        info = classify(classifier, q)
        print(f"\nQ: {q}")
        print(f"   topic={info.topic!r}  difficulty={info.difficulty}  keywords={info.keywords}")
    # TRY THIS:
    # 1. Add a field `estimated_minutes: int` and run again — no prompt change needed.
    # 2. Remove a Field description and see whether answers get worse.


if __name__ == "__main__":
    main()
