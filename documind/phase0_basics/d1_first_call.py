"""Demo 1 — your first LLM call.

CONCEPTS
- Messages have ROLES:
    system = standing instructions ("who you are, how to answer")
    human  = the user's message
    ai     = the model's reply
- The model only sees the text you send it. It has no memory between calls
  unless YOU send the earlier messages again (Phase 3 automates that).
- TOKENS are the pieces of text models read and write (~4 characters each in
  English). Free tiers and prices are measured in tokens, so we print them.

Run:  uv run python -m documind.phase0_basics.d1_first_call
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from documind.llm import get_llm

SYSTEM_PROMPT = "You are a patient tutor. Answer in at most 3 short sentences."


def ask(llm: BaseChatModel, question: str) -> AIMessage:
    """Send one system + one human message and return the AI message."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    # invoke() = send the messages, wait for the full reply.
    return llm.invoke(messages)


def format_usage(reply: AIMessage) -> str:
    """Turn the token counts LangChain attaches to the reply into one line."""
    usage = reply.usage_metadata  # None if the provider didn't report it
    if not usage:
        return "token usage: not reported"
    return (
        f"tokens -> input: {usage['input_tokens']}, "
        f"output: {usage['output_tokens']}, total: {usage['total_tokens']}"
    )


def main() -> None:
    llm = get_llm()
    question = "What is a Python?"
    reply = ask(llm, question)

    print(f"Q: {question}\n")
    print(f"A: {reply.content}\n")
    print(format_usage(reply))
    # TRY THIS:
    # 1. Change SYSTEM_PROMPT to "Answer like a pirate." Same question, new style.
    # 2. Ask a follow-up like "Give an example of one" in a new run. It won't
    #    know what "one" means: no memory between calls.
    # 3. Ask for a long answer and watch the output token count grow.


if __name__ == "__main__":
    main()
