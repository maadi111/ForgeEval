"""
Hidden test 1 — Generalization under distribution drift.

Why this test exists:
Validates that when the query vocabulary shifts to new domain concepts (e.g. MLOps,
Kubernetes, GPU operators), retrieval quality remains high (Recall@5 >= 80%).
A buggy pipeline with unnormalized vector dot products or aggressive token truncation
suffers severe recall collapse on drifted domains.
"""

import json
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


@pytest.fixture(scope="module")
def drifted_queries():
    with open(DATA_DIR / "queries.json", "r", encoding="utf-8") as f:
        return json.load(f)["drifted"]


def test_drifted_domain_recall_at_5(retriever, drifted_queries):
    """Drifted domain queries must achieve Recall@5 >= 80%."""
    hits = 0
    total = len(drifted_queries)

    for item in drifted_queries:
        query = item["query"]
        relevant_ids = set(item["relevant_ids"])

        results = retriever.retrieve(query, top_k=5)
        retrieved_ids = {r["id"] for r in results}

        if relevant_ids.intersection(retrieved_ids):
            hits += 1

    recall = hits / total
    assert recall >= 0.80, (
        f"Recall@5 on drifted domain = {recall:.1%} (threshold 80%). "
        "The retriever failed under distribution drift. Verify embedding normalization "
        "and token preservation in query preprocessing."
    )


def test_drifted_domain_top_1_precision(retriever, drifted_queries):
    """At least 60% of drifted queries must have relevant document in rank 1."""
    hits = 0
    for item in drifted_queries:
        query = item["query"]
        relevant_ids = set(item["relevant_ids"])
        results = retriever.retrieve(query, top_k=1)
        if results and results[0]["id"] in relevant_ids:
            hits += 1

    precision = hits / len(drifted_queries)
    assert precision >= 0.60, f"Top-1 Precision on drifted domain is too low: {precision:.1%}"
