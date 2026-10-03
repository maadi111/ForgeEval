"""
Public test 1 — Retrieval contract and response schema.

Visible to the candidate: verifies basic search interface and cardinality.
"""

import pathlib
import sys

import pytest

WORKSPACE = pathlib.Path(__file__).parents[2] / "workspace"
sys.path.insert(0, str(WORKSPACE))

from retriever import DenseRetriever


@pytest.fixture(scope="module")
def retriever():
    return DenseRetriever()


def test_retrieve_cardinality(retriever):
    results = retriever.retrieve("virtual machine scaling", top_k=5)
    assert isinstance(results, list)
    assert len(results) == 5


def test_retrieve_schema(retriever):
    results = retriever.retrieve("cloud storage bucket permissions", top_k=3)
    for res in results:
        assert "id" in res
        assert "score" in res
        assert "title" in res
        assert isinstance(res["id"], str)
        assert isinstance(res["score"], float)


def test_empty_query_fallback(retriever):
    results = retriever.retrieve("", top_k=5)
    assert len(results) == 5
