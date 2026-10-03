"""
Public test 1 — Single prediction contract and output format.

Visible to the candidate: verifies basic functionality of predict_one.
Passing this test is necessary but not sufficient.
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
def sample_payload():
    return {
        "age": 42,
        "income": 75000.0,
        "credit_score": 720,
        "debt_to_income": 0.28,
        "num_accounts": 5,
        "recent_inquiries": 1,
        "employment_years": 8.5,
        "loan_amount": 15000.0,
    }


def test_predict_one_contract(server, sample_payload):
    result = server.predict_one(sample_payload)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0


def test_probability_corresponds_to_prediction(server, sample_payload):
    result = server.predict_one(sample_payload)
    if result["prediction"] == 1:
        assert result["probability"] >= 0.5
    else:
        assert result["probability"] < 0.5
