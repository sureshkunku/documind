"""Demo 2 — temperature.

CONCEPTS
- An LLM picks the next token from a list of likely options.
- TEMPERATURE controls how adventurous that pick is:
    0     -> almost always the most likely token (repeatable, factual)
    ~0.7  -> some variety (good for brainstorming)
    1.5+  -> very random (often odd)
- For study notes we use 0: the same question should give the same answer.

Run:  uv run python -m documind.phase0_basics.d2_temperature
"""

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from documind.llm import get_llm

PROMPT = "what is python in one word?."


def sample_at_temperatures(
    make_llm: Callable[[float], BaseChatModel],
    prompt: str,
    temperatures: list[float],
    runs: int = 3,
) -> dict[float, list[str]]:
    """Ask the same prompt `runs` times at each temperature.

    make_llm is a function (temperature -> model) so tests can pass a fake.
    """
    results: dict[float, list[str]] = {}
    for t in temperatures:
        llm = make_llm(t)
        results[t] = [str(llm.invoke(prompt).content).strip() for _ in range(runs)]
    return results


def main() -> None:
    results = sample_at_temperatures(
        make_llm=lambda t: get_llm(temperature=t),
        prompt=PROMPT,
        temperatures=[0.0, 0.7, 1.5],
    )
    for temperature, answers in results.items():
        distinct = len(set(answers))
        print(f"\ntemperature={temperature}  ({distinct} different answers out of {len(answers)})")
        for answer in answers:
            print(f"   - {answer}")
    # TRY THIS:
    # 1. Use a factual prompt ("Capital of France?"). Temperature matters less
    #    when one answer is far more likely than any other.
    # 2. Raise runs to 5 and compare how many distinct answers each level gives.


if __name__ == "__main__":
    main()
