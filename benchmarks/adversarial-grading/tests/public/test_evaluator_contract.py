"""Public test 1: Verify evaluator interface and contract."""

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


def test_evaluator_interface(probe_dataset):
    """Verify that ModelEvaluator conforms to expected API structure."""
    from evaluator import ModelEvaluator
    from models import EvaluationReport

    evaluator = ModelEvaluator(probe_dataset)
    assert hasattr(evaluator, "evaluate")

    # Evaluate with baseline model
    lr_model, _ = train_genuine_models(probe_dataset)
    report = evaluator.evaluate(lr_model)

    assert isinstance(report, EvaluationReport)
    assert isinstance(report.passed, bool)
    assert isinstance(report.score, float)
    assert 0.0 <= report.score <= 100.0
    assert isinstance(report.metrics, dict)
    assert isinstance(report.failures, list)
