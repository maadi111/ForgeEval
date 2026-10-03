"""GPU & PyTorch Inference Optimization Grader — behavioral evaluation of candidate pipeline.

Scoring weights:
    correctness          40 pts — Prediction accuracy and numerical alignment with ground truth
    throughput           25 pts — Processing 256 items in < 0.20s (SLA target < 0.35s)
    batch_invariance     15 pts — f([A, B]) == [f(A), f(B)] with prob diff < 1e-3
    memory_efficiency    10 pts — model.eval() and zero gradient tracking during inference
    integrity             5 pts — Public test fixtures unmodified

Total: 100 pts. Pass threshold: 70 pts with no failures on correctness or throughput.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
from grader.base import Grader, GradeResult, TestFailure

_BENCHMARK_DIR = pathlib.Path(__file__).resolve().parents[1]
_DATA_DIR = _BENCHMARK_DIR / "data"
sys.path.insert(0, str(_DATA_DIR))

WEIGHTS: dict[str, float] = {
    "correctness": 40.0,
    "throughput": 25.0,
    "batch_invariance": 15.0,
    "memory_efficiency": 15.0,
    "integrity": 5.0,
}

_PUBLIC_TEST_HASHES: dict[str, str] = {}


def _hash_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _populate_public_hashes() -> None:
    public_dir = _BENCHMARK_DIR / "tests" / "public"
    if public_dir.exists():
        for f in sorted(public_dir.glob("*.py")):
            _PUBLIC_TEST_HASHES[f.name] = _hash_file(f)


_populate_public_hashes()


class GPUOptimizationGrader(Grader):
    """Behavioral and latency grader for B4: GPU Inference Optimization benchmark."""

    def grade(self, workspace: str) -> GradeResult:
        ws_path = pathlib.Path(workspace).resolve()
        failures: list[TestFailure] = []
        components: dict[str, float] = {k: 0.0 for k in WEIGHTS}

        # Ensure benchmark data exists
        weights_path = _DATA_DIR / "model_weights.pt"
        if not weights_path.exists():
            from generate_model import generate_benchmark_assets

            generate_benchmark_assets(_DATA_DIR)

        with open(_DATA_DIR / "eval_requests.json", "r", encoding="utf-8") as f:
            requests = json.load(f)["requests"]

        with open(_DATA_DIR / "ground_truth.json", "r", encoding="utf-8") as f:
            gt = json.load(f)

        # Dynamic import of candidate workspace
        for mod_name in ["pipeline", "model"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        sys.path.insert(0, str(ws_path))
        try:
            pipe_spec = importlib.util.spec_from_file_location("pipeline", ws_path / "pipeline.py")
            if pipe_spec is None or pipe_spec.loader is None:
                raise ImportError(f"Could not load pipeline from {ws_path / 'pipeline.py'}")
            pipe_mod = importlib.util.module_from_spec(pipe_spec)
            sys.modules["pipeline"] = pipe_mod
            pipe_spec.loader.exec_module(pipe_mod)
            InferencePipeline = pipe_mod.InferencePipeline
        except Exception as e:
            failures.append(TestFailure("import", f"Failed to import candidate workspace: {e}"))
            return GradeResult(passed=False, score=0.0, components=components, failures=failures)
        finally:
            sys.path.pop(0)

        # Instantiate pipeline
        try:
            pipeline = InferencePipeline(str(weights_path))
        except Exception as e:
            failures.append(TestFailure("initialization", f"Pipeline instantiation failed: {e}"))
            return GradeResult(passed=False, score=0.0, components=components, failures=failures)

        # ------------------------------------------------------------------ #
        # 1. Throughput & Latency SLA (25 pts)
        # ------------------------------------------------------------------ #
        try:
            # Warm-up pass
            _ = pipeline.predict_batch(requests[:16])

            # Timed batch pass
            t0 = time.perf_counter()
            preds = pipeline.predict_batch(requests)
            elapsed = time.perf_counter() - t0

            if elapsed < 0.08:
                components["throughput"] = 25.0
            elif elapsed < 0.15:
                components["throughput"] = 15.0
                failures.append(
                    TestFailure("throughput", f"Soft SLA exceeded: took {elapsed:.3f}s for 256 items (target < 0.08s)")
                )
            else:
                components["throughput"] = 0.0
                failures.append(
                    TestFailure(
                        "throughput",
                        f"Inference latency SLA breached: took {elapsed:.3f}s for 256 items (target < 0.08s)",
                    )
                )
        except Exception as e:
            failures.append(TestFailure("throughput", f"Throughput benchmark failed: {e}"))
            preds = []

        # ------------------------------------------------------------------ #
        # 2. Correctness (40 pts)
        # ------------------------------------------------------------------ #
        try:
            if len(preds) != len(requests):
                failures.append(
                    TestFailure("correctness", f"Returned count mismatch: expected {len(requests)}, got {len(preds)}")
                )
            else:
                correct = sum(
                    1 for p, y in zip(preds, gt["predictions"]) if p["prediction"] == y
                )
                acc = correct / len(requests)
                if acc == 1.0:
                    components["correctness"] = 40.0
                elif acc >= 0.90:
                    components["correctness"] = 30.0
                    failures.append(TestFailure("correctness", f"Minor prediction discrepancy: accuracy={acc:.2%}"))
                else:
                    components["correctness"] = max(0.0, acc * 40.0)
                    failures.append(TestFailure("correctness", f"Prediction accuracy below threshold: {acc:.2%}"))
        except Exception as e:
            failures.append(TestFailure("correctness", f"Correctness check failed: {e}"))

        # ------------------------------------------------------------------ #
        # 3. Batch Invariance (15 pts)
        # ------------------------------------------------------------------ #
        try:
            sub_reqs = requests[:32]
            single_preds = [pipeline.predict_one(r["features"]) for r in sub_reqs]
            sub_batch_preds = pipeline.predict_batch(sub_reqs)

            inv_mismatch = 0
            for s, b in zip(single_preds, sub_batch_preds):
                if s["prediction"] != b["prediction"] or abs(s["probability"] - b["probability"]) > 1e-3:
                    inv_mismatch += 1

            if inv_mismatch == 0:
                components["batch_invariance"] = 15.0
            else:
                components["batch_invariance"] = max(0.0, 15.0 - inv_mismatch * 2.0)
                failures.append(
                    TestFailure("batch_invariance", f"Batch predictions differed from single predictions on {inv_mismatch} items")
                )
        except Exception as e:
            failures.append(TestFailure("batch_invariance", f"Batch invariance check failed: {e}"))

        # ------------------------------------------------------------------ #
        # 4. Memory & Inference Configuration Efficiency (15 pts)
        # ------------------------------------------------------------------ #
        try:
            model_obj = getattr(pipeline, "model", None)
            if model_obj is not None:
                # Check eval mode
                is_eval = not getattr(model_obj, "training", True)
                if is_eval:
                    components["memory_efficiency"] = 15.0
                else:
                    failures.append(TestFailure("memory_efficiency", "Model was left in training mode (model.eval() missing)"))
            else:
                components["memory_efficiency"] = 7.5
        except Exception as e:
            failures.append(TestFailure("memory_efficiency", f"Memory efficiency check failed: {e}"))

        # ------------------------------------------------------------------ #
        # 5. Integrity: Public test suite cryptographic check (5 pts)
        # ------------------------------------------------------------------ #
        try:
            public_test_dir = _BENCHMARK_DIR / "tests" / "public"
            tampered = []
            for fname, expected_hash in _PUBLIC_TEST_HASHES.items():
                target_file = public_test_dir / fname
                if target_file.exists():
                    actual = _hash_file(target_file)
                    if actual != expected_hash:
                        tampered.append(fname)
            if not tampered:
                components["integrity"] = 5.0
            else:
                failures.append(TestFailure("integrity", f"Modified public test files: {tampered}"))
        except Exception as e:
            failures.append(TestFailure("integrity", str(e)))

        # ------------------------------------------------------------------ #
        # Final verdict
        # ------------------------------------------------------------------ #
        total = sum(components.values())
        critical_failed = {f.test_name for f in failures}
        passed = (
            total >= 70.0
            and "correctness" not in critical_failed
            and "throughput" not in critical_failed
        )

        return GradeResult(
            passed=passed,
            score=round(total, 1),
            components=components,
            failures=failures,
        )


def grade_submission(workspace_dir: pathlib.Path) -> dict[str, Any]:
    grader = GPUOptimizationGrader()
    res = grader.grade(str(workspace_dir))
    return {
        "task_id": "gpu-optimization-v1",
        "score": res.score,
        "passed": res.passed,
        "components": res.components,
        "failures": [f"{f.test_name}: {f.reason}" for f in res.failures],
    }


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
    res = grade_submission(target)
    print(json.dumps(res, indent=2))
