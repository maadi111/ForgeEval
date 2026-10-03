"""
Public test 2 — Historical retrieval baseline.

Checks that the retriever achieves basic recall on familiar historical queries.
Passing this test alone does not guarantee generalization under distribution drift.
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
def historical_queries():
    with open(DATA_DIR / "queries.json", "r", encoding="utf-8") as f:
        return json.load(f)["historical"]


def test_historical_recall_at_5(retriever, historical_queries):
    """Historical domain queries must meet baseline recall (>= 60%)."""
    hits = 0
    total = len(historical_queries)

    for item in historical_queries:
        query = item["query"]
        relevant_ids = set(item["relevant_ids"])

        results = retriever.retrieve(query, top_k=5)
        retrieved_ids = {r["id"] for r in results}

        if relevant_ids.intersection(retrieved_ids):
            hits += 1

    recall = hits / total
    assert recall >= 0.60, f"Historical Recall@5 is too low: {recall:.2%}"
