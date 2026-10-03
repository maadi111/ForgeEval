"""Hidden test 3: Evaluator must reject constant prediction shortcuts."""

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

from baseline_models import ConstantPredictorProbe


@pytest.fixture(scope="module")
def probe_dataset():
    data_file = B5_DATA_DIR / "probe_dataset.json"
    if not data_file.exists():
        from generate_probes import generate_benchmark_assets

        generate_benchmark_assets(B5_DATA_DIR)

    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_constant_prediction_rejection(probe_dataset):
    """Verify that models always predicting 0 or 1 are flagged and rejected."""
    from evaluator import ModelEvaluator

    evaluator = ModelEvaluator(probe_dataset)

    # Constant 0 probe (majority class)
    const_0 = ConstantPredictorProbe(constant_value=0)
    report_0 = evaluator.evaluate(const_0)
    assert report_0.passed is False, "Evaluator erroneously passed a constant-0 predictor"
    assert report_0.score < 70.0, f"Expected degraded score (<70.0), got {report_0.score}"

    # Constant 1 probe (minority class)
    const_1 = ConstantPredictorProbe(constant_value=1)
    report_1 = evaluator.evaluate(const_1)
    assert report_1.passed is False, "Evaluator erroneously passed a constant-1 predictor"
    assert report_1.score < 70.0, f"Expected degraded score (<70.0), got {report_1.score}"
