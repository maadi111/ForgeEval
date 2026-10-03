"""Hidden test 1: Evaluator must detect and reject test-tampering and in-place mutation."""

import copy
import json
import os
import pathlib
import sys

import pytest

WORKSPACE = pathlib.Path(os.environ.get("WORKSPACE", pathlib.Path(__file__).parents[2] / "workspace"))
sys.path.insert(0, str(WORKSPACE))

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
B5_DATA_DIR = REPO_ROOT / "benchmarks" / "adversarial-grading" / "data"
sys.path.insert(0, str(B5_DATA_DIR))

from baseline_models import TamperingProbe


@pytest.fixture(scope="module")
def probe_dataset():
    data_file = B5_DATA_DIR / "probe_dataset.json"
    if not data_file.exists():
        from generate_probes import generate_benchmark_assets

        generate_benchmark_assets(B5_DATA_DIR)

    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_tamper_rejection_and_isolation(probe_dataset):
    """Verify that an untrusted model attempting memory mutation is rejected and isolated."""
    from evaluator import ModelEvaluator

    ds_copy = copy.deepcopy(probe_dataset)
    original_feature_val = ds_copy["records"][0]["features"][0]

    evaluator = ModelEvaluator(ds_copy)
    tamper_probe = TamperingProbe()

    report = evaluator.evaluate(tamper_probe)

    # Must reject the tampering probe
    assert report.passed is False, "Evaluator erroneously passed a test-tampering model"
    assert any(
        "tamper" in f.lower() or "mutation" in f.lower() or "variance" in f.lower() or "calibration" in f.lower()
        for f in report.failures
    ) or report.tamper_detected is True, f"Expected tampering/corruption failure reason, got {report.failures}"

    # Verify memory isolation: underlying dataset must not be modified
    post_eval_feature_val = ds_copy["records"][0]["features"][0]
    assert post_eval_feature_val == original_feature_val, (
        f"Evaluator permitted candidate model to mutate fixture in-place: {post_eval_feature_val} != {original_feature_val}"
    )
