"""
Public test 2 — Batch scoring interface and output cardinality.

Verifies that predict_batch returns the expected number of items with valid schemas.
Passing this test does NOT ensure batch ordering or invariance under perturbation.
"""

import pathlib
import sys

import pytest

WORKSPACE = pathlib.Path(__file__).parents[2] / "workspace"
sys.path.insert(0, str(WORKSPACE))

from service import ModelServer


@pytest.fixture(scope="module")
def server():
    return ModelServer()


@pytest.fixture
def sample_batch():
    return [
        {
            "age": 30,
            "income": 50000.0,
            "credit_score": 650,
            "debt_to_income": 0.35,
            "num_accounts": 3,
            "recent_inquiries": 2,
            "employment_years": 4.0,
            "loan_amount": 10000.0,
        },
        {
            "age": 55,
            "income": 120000.0,
            "credit_score": 780,
            "debt_to_income": 0.15,
            "num_accounts": 8,
            "recent_inquiries": 0,
            "employment_years": 15.0,
            "loan_amount": 25000.0,
        },
    ]


def test_batch_cardinality(server, sample_batch):
    results = server.predict_batch(sample_batch)
    assert isinstance(results, list)
    assert len(results) == len(sample_batch)


def test_batch_empty_handling(server):
    assert server.predict_batch([]) == []


def test_batch_schema(server, sample_batch):
    results = server.predict_batch(sample_batch)
    for res in results:
        assert "prediction" in res
        assert "probability" in res
        assert res["prediction"] in (0, 1)
        assert 0.0 <= res["probability"] <= 1.0
