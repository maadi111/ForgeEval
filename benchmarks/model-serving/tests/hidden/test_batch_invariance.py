"""
Hidden test 1 — Batch Invariance and Request Ordering.

Why this test exists:
A model inference service must be invariant to batching. Scoring an item in a batch
must produce the exact same prediction and probability as scoring that item in isolation:
predict_batch([item_A, item_B])[0] == predict_one(item_A)

A common bug in dynamic micro-batching is sorting items (e.g. by size or sequence length)
to optimize execution, but failing to invert the permutation before responding.
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
def test_profiles():
    # Intentionally unsorted loan amounts to expose batch ordering bugs
    return [
        {
            "age": 28,
            "income": 45000.0,
            "credit_score": 580,
            "debt_to_income": 0.45,
            "num_accounts": 2,
            "recent_inquiries": 4,
            "employment_years": 2.0,
            "loan_amount": 42000.0,  # High loan amount
        },
        {
            "age": 52,
            "income": 140000.0,
            "credit_score": 810,
            "debt_to_income": 0.12,
            "num_accounts": 9,
            "recent_inquiries": 0,
            "employment_years": 22.0,
            "loan_amount": 3500.0,  # Low loan amount
        },
        {
            "age": 39,
            "income": 82000.0,
            "credit_score": 690,
            "debt_to_income": 0.31,
            "num_accounts": 6,
            "recent_inquiries": 1,
            "employment_years": 7.5,
            "loan_amount": 18000.0,  # Medium loan amount
        },
    ]


def test_batch_matches_single_predictions(server, test_profiles):
    """Each item in a batch must yield identical predictions to predict_one."""
    batch_results = server.predict_batch(test_profiles)
    single_results = [server.predict_one(item) for item in test_profiles]

    assert len(batch_results) == len(single_results)

    for i, (b_res, s_res) in enumerate(zip(batch_results, single_results)):
        assert b_res["prediction"] == s_res["prediction"], (
            f"Prediction mismatch at index {i}: batch got {b_res['prediction']}, "
            f"single got {s_res['prediction']}. Probabilities: batch={b_res['probability']} "
            f"vs single={s_res['probability']}. Check for batch reordering bugs."
        )
        assert abs(b_res["probability"] - s_res["probability"]) < 1e-4, (
            f"Probability mismatch at index {i}: batch={b_res['probability']} "
            f"vs single={s_res['probability']}."
        )


def test_order_invariance_under_permutation(server, test_profiles):
    """Permuting the batch input must permute the output identically."""
    permuted_profiles = [test_profiles[1], test_profiles[0], test_profiles[2]]
    permuted_results = server.predict_batch(permuted_profiles)
    expected_results = [server.predict_one(p) for p in permuted_profiles]

    for i, (act, exp) in enumerate(zip(permuted_results, expected_results)):
        assert act["prediction"] == exp["prediction"], (
            f"Permuted batch mismatch at index {i}: expected {exp['prediction']}, got {act['prediction']}."
        )
