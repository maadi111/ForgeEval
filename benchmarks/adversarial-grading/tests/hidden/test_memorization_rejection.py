"""Hidden test 2: Evaluator must detect and reject memorized lookup shortcuts."""

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

from baseline_models import MemorizedLookupProbe


@pytest.fixture(scope="module")
def probe_dataset():
    data_file = B5_DATA_DIR / "probe_dataset.json"
    if not data_file.exists():
        from generate_probes import generate_benchmark_assets

        generate_benchmark_assets(B5_DATA_DIR)

    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_memorization_shortcut_rejection(probe_dataset):
    """Verify that a model with hardcoded/memorized feature lookup is rejected."""
    from evaluator import ModelEvaluator

    evaluator = ModelEvaluator(probe_dataset)
    mem_probe = MemorizedLookupProbe(probe_dataset)

    report = evaluator.evaluate(mem_probe)

    assert report.passed is False, "Evaluator erroneously passed a memorized hash-lookup probe"
    assert any(
        "memoriz" in f.lower()
        or "perturb" in f.lower()
        or "jitter" in f.lower()
        or "invariance" in f.lower()
        for f in report.failures
    ), f"Expected memorization or perturbation failure recorded, got: {report.failures}"
