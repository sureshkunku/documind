"""Phase 0 demo tests.

WHY a fake LLM: FakeListChatModel returns the replies we give it. Tests run
offline, free and the same every time, while still exercising the real
LangChain plumbing (messages, invoke, stream, chains).
"""

import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from documind.phase0_basics import d1_first_call, d2_temperature, d3_streaming, d4_structured, d5_lcel_chain
from documind.phase0_basics.d6_embeddings import cosine_similarity_matrix, most_similar_pairs


# ---------- d1: first call ----------

def test_ask_returns_the_model_reply():
    llm = FakeListChatModel(responses=["A network of layers."])
    reply = d1_first_call.ask(llm, "What is a neural network?")
    assert reply.content == "A network of layers."


def test_format_usage_with_and_without_counts():
    with_usage = AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    assert d1_first_call.format_usage(with_usage) == "tokens -> input: 10, output: 5, total: 15"
    assert d1_first_call.format_usage(AIMessage(content="x")) == "token usage: not reported"


# ---------- d2: temperature ----------

def test_sample_at_temperatures_asks_each_level_n_times():
    seen_temps = []

    def make_llm(t):
        seen_temps.append(t)
        return FakeListChatModel(responses=[" StudyBot ", "NoteNinja"])

    results = d2_temperature.sample_at_temperatures(make_llm, "name?", [0.0, 1.0], runs=3)

    assert seen_temps == [0.0, 1.0]
    assert list(results) == [0.0, 1.0]
    assert results[0.0] == ["StudyBot", "NoteNinja", "StudyBot"]  # stripped, cycles


# ---------- d3: streaming ----------

def test_stream_answer_yields_pieces_that_join_to_full_text():
    llm = FakeListChatModel(responses=["Gradient descent walks downhill."])
    pieces = list(d3_streaming.stream_answer(llm, "q"))
    assert len(pieces) > 1  # arrived in several chunks, not one
    assert "".join(pieces) == "Gradient descent walks downhill."


# ---------- d4: structured output ----------

def test_classify_returns_question_info():
    expected = d4_structured.QuestionInfo(topic="python", difficulty="beginner", keywords=["loop", "range"])
    fake_classifier = RunnableLambda(lambda _q: expected)
    assert d4_structured.classify(fake_classifier, "What is a for loop?") == expected


def test_classify_rejects_non_structured_reply():
    with pytest.raises(TypeError):
        d4_structured.classify(RunnableLambda(lambda _q: "plain text"), "q")


def test_schema_rejects_unknown_difficulty():
    # This validation is what makes structured output safe to use in code.
    with pytest.raises(ValidationError):
        d4_structured.QuestionInfo(topic="x", difficulty="expert", keywords=[])


# ---------- d5: LCEL chain ----------

def test_prompt_template_fills_placeholders():
    messages = d5_lcel_chain.PROMPT.invoke({"topic": "overfitting", "level": "beginner"}).to_messages()
    assert "beginner student" in messages[0].content
    assert messages[1].content == "Explain: overfitting"


def test_chain_returns_plain_string():
    chain = d5_lcel_chain.build_chain(FakeListChatModel(responses=["Memorising, not learning."]))
    assert chain.invoke({"topic": "overfitting", "level": "beginner"}) == "Memorising, not learning."


def test_chain_batch_returns_one_answer_per_input():
    chain = d5_lcel_chain.build_chain(FakeListChatModel(responses=["a", "b"]))
    answers = chain.batch([{"topic": "t", "level": "l"}] * 2)
    # batch runs inputs in parallel, so which fake reply each gets can vary
    assert len(answers) == 2
    assert set(answers) <= {"a", "b"}


# ---------- d6: embeddings math ----------

def test_cosine_similarity_known_values():
    sims = cosine_similarity_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    assert np.allclose(np.diag(sims), 1.0)           # every vector matches itself
    assert sims[0, 1] == pytest.approx(0.0)           # 90 degrees apart = unrelated
    assert sims[0, 2] == pytest.approx(1 / np.sqrt(2))  # 45 degrees apart
    assert np.allclose(sims, sims.T)                  # symmetric


def test_cosine_similarity_ignores_vector_length():
    sims = cosine_similarity_matrix(np.array([[1.0, 2.0], [10.0, 20.0]]))
    assert sims[0, 1] == pytest.approx(1.0)  # same direction, different length


def test_zero_vector_does_not_produce_nan():
    sims = cosine_similarity_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert not np.isnan(sims).any()


def test_most_similar_pairs_sorted_and_skips_self():
    sentences = ["a", "b", "c"]
    sims = np.array([[1.0, 0.2, 0.9], [0.2, 1.0, 0.5], [0.9, 0.5, 1.0]])
    pairs = most_similar_pairs(sentences, sims, top=2)
    assert pairs == [(0.9, "a", "c"), (0.5, "b", "c")]


# ---------- live (opt-in: uv run pytest -m live) ----------

@pytest.mark.live
def test_live_groq_first_call():
    from documind.llm import get_llm

    reply = d1_first_call.ask(get_llm(), "Reply with the single word: ready")
    assert "ready" in str(reply.content).lower()
    assert reply.usage_metadata["total_tokens"] > 0


@pytest.mark.live
def test_live_embeddings_capture_meaning():
    from documind.phase0_basics.d6_embeddings import SENTENCES, embed

    sims = cosine_similarity_matrix(embed(SENTENCES))
    # neural-net sentences are closer to each other than to the pasta sentence
    assert sims[0, 1] > sims[0, 4]
    # the two SQL sentences are closer to each other than to neural networks
    assert sims[2, 3] > sims[2, 0]
