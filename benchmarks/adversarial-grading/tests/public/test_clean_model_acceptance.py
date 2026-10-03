"""Public test 2: Genuine machine learning models should be accepted with high scores."""

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

from baseline_models import train_genuine_models


@pytest.fixture(scope="module")
def probe_dataset():
    data_file = B5_DATA_DIR / "probe_dataset.json"
    if not data_file.exists():
        from generate_probes import generate_benchmark_assets

        generate_benchmark_assets(B5_DATA_DIR)

    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_genuine_model_acceptance(probe_dataset):
    """Verify that a genuine, trained classifier passes evaluation."""
    from evaluator import ModelEvaluator

    lr_model, rf_model = train_genuine_models(probe_dataset)
    evaluator = ModelEvaluator(probe_dataset)

    # Evaluate Logistic Regression
    report_lr = evaluator.evaluate(lr_model)
    assert report_lr.passed is True, f"Genuine Logistic Regression failed: {report_lr.failures}"
    assert report_lr.score >= 75.0, f"Expected score >= 75.0, got {report_lr.score}"

    # Evaluate Random Forest
    report_rf = evaluator.evaluate(rf_model)
    assert report_rf.passed is True, f"Genuine Random Forest failed: {report_rf.failures}"
    assert report_rf.score >= 75.0, f"Expected score >= 75.0, got {report_rf.score}"
