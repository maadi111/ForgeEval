"""
Hidden test 2 — Embedding scale and length normalization invariance.

Why this test exists:
Cosine similarity measures angular direction, not vector magnitude. In unnormalized
systems, documents with repeated keywords or long descriptions produce artificially
high dot-product scores, overpowering short, highly relevant documents.
"""

import pathlib
import sys

import pytest

WORKSPACE = pathlib.Path(__file__).parents[2] / "workspace"
DATA_DIR = pathlib.Path(__file__).parents[2] / "data"
sys.path.insert(0, str(WORKSPACE))

from retriever import DenseRetriever


@pytest.fixture(scope="module")
def retriever():
    return DenseRetriever(corpus_path=DATA_DIR / "corpus.json")


def test_scores_are_bounded_like_cosine(retriever):
    """Normalized cosine similarity scores must be bounded in [-1.0, 1.0] (or [0, 1] for TF-IDF)."""
    results = retriever.retrieve("kubernetes gpu deployment", top_k=5)
    for r in results:
        assert r["score"] <= 1.0001, (
            f"Score {r['score']} exceeds 1.0. Embeddings appear unnormalized; "
            "scores must reflect normalized cosine similarity."
        )


def test_repeated_tokens_in_query_do_not_distort_relative_rank(retriever):
    """Repeating words in the query should not change the top retrieved document."""
    res_single = retriever.retrieve("kubernetes cluster", top_k=1)
    res_repeat = retriever.retrieve("kubernetes kubernetes cluster cluster", top_k=1)

    assert res_single[0]["id"] == res_repeat[0]["id"]
