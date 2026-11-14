"""Demo 6 — embeddings.

CONCEPTS
- An EMBEDDING turns text into a list of numbers (a vector). MiniLM gives
  384 numbers per text.
- Texts with similar MEANING get vectors that point in a similar direction,
  even when they share no words ("car" vs "automobile").
- COSINE SIMILARITY measures that direction: 1.0 = same meaning,
  ~0 = unrelated. This is how RAG (Phase 1) finds the right notes.
- The model runs locally on your PC: free, private, no API key.
  First run downloads it (~90 MB), then it's cached.

Run:  uv run python -m documind.phase0_basics.d6_embeddings
"""

import numpy as np

from documind.config import get_settings

SENTENCES = [
    "A neural network learns by adjusting its weights.",
    "Backpropagation updates the model's parameters using gradients.",
    "SQL joins combine rows from two tables.",
    "INNER JOIN returns only rows that match in both tables.",
    "I had pasta for dinner yesterday.",
]


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity of every vector with every other vector.

    cos(a, b) = (a . b) / (|a| * |b|)
    Dividing each row by its length first turns that into a plain dot product.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.clip(norms, 1e-12, None)  # clip avoids dividing by 0
    return unit @ unit.T


def most_similar_pairs(sentences: list[str], sims: np.ndarray, top: int = 3) -> list[tuple[float, str, str]]:
    """The `top` most similar DIFFERENT sentence pairs, best first."""
    pairs = [
        (float(sims[i, j]), sentences[i], sentences[j])
        for i in range(len(sentences))
        for j in range(i + 1, len(sentences))  # j > i: skip self-pairs and duplicates
    ]
    return sorted(pairs, reverse=True)[:top]


def embed(sentences: list[str]) -> np.ndarray:
    # Imported here: loading torch is slow, and tests of the math don't need it.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(get_settings().embed_model)
    return model.encode(sentences)


def main() -> None:
    vectors = embed(SENTENCES)
    print(f"{len(SENTENCES)} sentences -> vectors of {vectors.shape[1]} numbers each")
    print(f"first 5 numbers of sentence 1: {np.round(vectors[0][:5], 3)}\n")

    sims = cosine_similarity_matrix(vectors)
    labels = [f"S{i + 1}" for i in range(len(SENTENCES))]
    print("similarity table:")
    print("      " + "  ".join(f"{label:>5}" for label in labels))
    for label, row in zip(labels, sims):
        print(f"{label:>5} " + "  ".join(f"{v:5.2f}" for v in row))

    print("\nmost similar pairs:")
    for score, a, b in most_similar_pairs(SENTENCES, sims):
        print(f"  {score:.2f}  {a!r}  <->  {b!r}")
    # TRY THIS:
    # 1. Add "Weights are tuned during training." Which sentence is it closest to?
    # 2. Add "A table at a restaurant." Does it get confused with SQL tables?


if __name__ == "__main__":
    main()
