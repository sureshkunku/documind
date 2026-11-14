"""Demo 5 — a LangChain chain (LCEL).

CONCEPTS
- PROMPT TEMPLATE: a prompt with {placeholders} you fill at run time.
  Keeps prompts in one place instead of f-strings scattered in code.
- LCEL ("LangChain Expression Language"): join steps with `|`, like a
  Unix pipe:   prompt | llm | parser
    prompt  -> turns {"topic": ..., "level": ...} into messages
    llm     -> turns messages into an AI message
    parser  -> turns the AI message into a plain string
- Every chain gets invoke(), stream() and batch() for free.

Run:  uv run python -m documind.phase0_basics.d5_lcel_chain
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from documind.llm import get_llm

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You explain topics to a {level} student in at most 3 sentences."),
        ("human", "Explain: {topic}"),
    ]
)


def build_chain(llm: BaseChatModel) -> Runnable:
    """prompt -> llm -> plain text."""
    return PROMPT | llm | StrOutputParser()


def main() -> None:
    chain = build_chain(get_llm())

    # invoke: one input
    print("== invoke ==")
    print(chain.invoke({"topic": "overfitting", "level": "beginner"}))

    # batch: many inputs at once (LangChain runs them in parallel)
    print("\n== batch: same topic, three levels ==")
    levels = ["10-year-old", "university", "PhD"]
    answers = chain.batch([{"topic": "overfitting", "level": lvl} for lvl in levels])
    for level, answer in zip(levels, answers):
        print(f"\n[{level}]\n{answer}")
    # TRY THIS:
    # 1. print(PROMPT.invoke({"topic": "x", "level": "y"})) to see the filled-in messages.
    # 2. Add a third step after the parser: `| (lambda text: text.upper())`.


if __name__ == "__main__":
    main()
