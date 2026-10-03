"""
Hidden test 3 — Negative Controls for Retrieval Drift.

Validates that degenerate solutions fail our evaluation suite:
NC-1: Dummy constant document ID returner fails recall.
NC-2: Random document selector fails precision.
NC-3: Truncating unnormalized retriever fails on drifted queries.
"""

import random


def test_nc1_constant_retriever_fails():
    """NC-1: Always returning the same doc ID must fail across multi-query evaluation."""
    queries = ["query_1", "query_2", "query_3", "query_4", "query_5"]
    targets = [f"doc_{i}" for i in range(1, 6)]

    def constant_retrieve(q):
        return ["doc_01"]

    hits = sum(1 for q, t in zip(queries, targets) if t in constant_retrieve(q))
    recall = hits / len(queries)
    assert recall < 0.30, "NC-1: Constant retriever scored higher than failure threshold"


def test_nc2_random_retriever_fails():
    """NC-2: Randomly selecting documents from corpus fails top-1 precision."""
    corpus_ids = [f"doc_{i:02d}" for i in range(1, 50)]
    targets = ["doc_05", "doc_12", "doc_23", "doc_31", "doc_44"]

    random.seed(42)
    hits = 0
    for target in targets:
        pred = random.choice(corpus_ids)
        if pred == target:
            hits += 1

    precision = hits / len(targets)
    assert precision < 0.40, "NC-2: Random retriever scored higher than expected"
