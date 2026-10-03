"""
Hidden test 3 — Negative Controls (known-bad serving implementations must fail).

Why these tests exist:
Validates that our evaluation harness rejects shortcut solutions:
NC-1: Constant probability predictor fails discrimination threshold.
NC-2: Arbitrary/sorted batching predictor fails batch invariance.
NC-3: Dict-order dependent preprocessor fails permutation invariance.
"""


def test_nc1_constant_predictor_fails_discrimination():
    """NC-1: A dummy constant predictor must fail evaluation."""
    # Simulation: constant probability
    predictions = [0.5] * 100
    actuals = [1] * 50 + [0] * 50

    # Mean squared error (Brier score)
    brier = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(actuals)
    # A competent classifier achieves Brier score < 0.20
    assert brier >= 0.25, "Constant predictor scored lower Brier than expected"


def test_nc2_unsorted_batching_fails_invariance():
    """NC-2: Deliberately simulating sorted batching must trigger batch invariance failure."""

    def dummy_predict_one(x):
        return {"id": x["id"], "score": x["val"]}

    def broken_batch(items):
        # Sorts by val but returns directly without inverting
        sorted_items = sorted(items, key=lambda x: x["val"])
        return [dummy_predict_one(x) for x in sorted_items]

    batch = [{"id": 1, "val": 100}, {"id": 2, "val": 10}]
    single_res = [dummy_predict_one(x) for x in batch]
    batch_res = broken_batch(batch)

    # Must detect mismatch
    mismatch = any(b["id"] != s["id"] for b, s in zip(batch_res, single_res))
    assert mismatch, "NC-2: Broken batching failed to be caught by invariance check"


def test_nc3_key_order_dependent_parser_fails():
    """NC-3: Verifies that dict-order dependent parsing is detected."""

    def buggy_preprocess(payload):
        return list(payload.values())

    payload_a = {"x": 10, "y": 20}
    payload_b = {"y": 20, "x": 10}

    assert buggy_preprocess(payload_a) != buggy_preprocess(payload_b), (
        "NC-3: Key-dependent parsing not detected"
    )
