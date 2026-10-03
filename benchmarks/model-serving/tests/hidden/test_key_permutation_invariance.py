"""
Hidden test 2 — JSON Payload Key Permutation Invariance.

Why this test exists:
JSON object keys have no guaranteed order per RFC 8259. Different client SDKs,
curl requests, or upstream JSON serializers may serialize keys in arbitrary orders.
A robust serving preprocessor must parse features by explicit column name, not by
dictionary iteration order.
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
def canonical_payload():
    return {
        "age": 35,
        "income": 95000.0,
        "credit_score": 710,
        "debt_to_income": 0.22,
        "num_accounts": 4,
        "recent_inquiries": 1,
        "employment_years": 9.0,
        "loan_amount": 12000.0,
    }


def test_reversed_key_order(server, canonical_payload):
    """Payload with reversed key order must yield the exact same prediction."""
    reversed_keys = list(reversed(list(canonical_payload.keys())))
    reversed_payload = {k: canonical_payload[k] for k in reversed_keys}

    baseline = server.predict_one(canonical_payload)
    test_result = server.predict_one(reversed_payload)

    assert test_result["prediction"] == baseline["prediction"], (
        f"Prediction differed when JSON keys were reversed: "
        f"canonical={baseline['prediction']}, reversed={test_result['prediction']}. "
        "The preprocessor likely iterates over dict.keys() instead of canonical column order."
    )
    assert abs(test_result["probability"] - baseline["probability"]) < 1e-4, (
        f"Probability changed when JSON keys were reversed: "
        f"{baseline['probability']} vs {test_result['probability']}."
    )


def test_alphabetical_key_order(server, canonical_payload):
    """Payload with alphabetically sorted keys must yield the exact same prediction."""
    sorted_keys = sorted(canonical_payload.keys())
    sorted_payload = {k: canonical_payload[k] for k in sorted_keys}

    baseline = server.predict_one(canonical_payload)
    test_result = server.predict_one(sorted_payload)

    assert test_result["prediction"] == baseline["prediction"]
    assert abs(test_result["probability"] - baseline["probability"]) < 1e-4
