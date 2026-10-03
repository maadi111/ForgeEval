"""Negative controls for B5: Verify that known shortcuts and exploits fail on reference evaluators."""

import copy
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
B5_DATA_DIR = REPO_ROOT / "benchmarks" / "adversarial-grading" / "data"
sys.path.insert(0, str(B5_DATA_DIR))

from baseline_models import (
    ConstantPredictorProbe,
    MemorizedLookupProbe,
    OrderExploitProbe,
    TamperingProbe,
)


@pytest.fixture(scope="module")
def probe_dataset():
    data_file = B5_DATA_DIR / "probe_dataset.json"
    if not data_file.exists():
        from generate_probes import generate_benchmark_assets

        generate_benchmark_assets(B5_DATA_DIR)

    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_evaluator_cls(sol_name: str):
    """Dynamically load ModelEvaluator from the specified reference solution, flushing module cache."""
    sol_dir = REPO_ROOT / "benchmarks" / "adversarial-grading" / "reference" / sol_name
    for mod_name in ["evaluator", "security", "models"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    sys.path.insert(0, str(sol_dir))
    try:
        import evaluator

        return evaluator.ModelEvaluator
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("sol_name", ["solution-a", "solution-b"])
def test_negative_control_memorization_fails(probe_dataset, sol_name):
    """Negative control: Memorized lookup probe must fail on reference evaluators."""
    evaluator_cls = _load_evaluator_cls(sol_name)
    evaluator = evaluator_cls(probe_dataset)
    probe = MemorizedLookupProbe(probe_dataset)
    report = evaluator.evaluate(probe)
    assert report.passed is False
    assert any(
        "memoriz" in f.lower() or "perturb" in f.lower() or "jitter" in f.lower()
        for f in report.failures
    )


@pytest.mark.parametrize("sol_name", ["solution-a", "solution-b"])
def test_negative_control_constant_predictor_fails(probe_dataset, sol_name):
    """Negative control: Constant predictor probe must fail on reference evaluators."""
    evaluator_cls = _load_evaluator_cls(sol_name)
    evaluator = evaluator_cls(probe_dataset)
    probe = ConstantPredictorProbe(constant_value=0)
    report = evaluator.evaluate(probe)
    assert report.passed is False
    assert report.score < 70.0


@pytest.mark.parametrize("sol_name", ["solution-a", "solution-b"])
def test_negative_control_order_exploit_fails(probe_dataset, sol_name):
    """Negative control: Order exploit probe must fail on reference evaluators."""
    evaluator_cls = _load_evaluator_cls(sol_name)
    evaluator = evaluator_cls(probe_dataset)
    labels = [r["label"] for r in probe_dataset["records"]]
    probe = OrderExploitProbe(ground_truth_labels=labels)
    report = evaluator.evaluate(probe)
    assert report.passed is False
    assert any("order" in f.lower() or "permutation" in f.lower() for f in report.failures)


@pytest.mark.parametrize("sol_name", ["solution-a", "solution-b"])
def test_negative_control_tampering_fails(probe_dataset, sol_name):
    """Negative control: Tampering probe must fail and preserve dataset integrity."""
    evaluator_cls = _load_evaluator_cls(sol_name)
    ds = copy.deepcopy(probe_dataset)
    orig_val = ds["records"][0]["features"][0]
    evaluator = evaluator_cls(ds)
    probe = TamperingProbe()
    report = evaluator.evaluate(probe)
    assert report.passed is False
    assert ds["records"][0]["features"][0] == orig_val
