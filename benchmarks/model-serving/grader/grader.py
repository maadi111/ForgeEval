"""
Model Serving Grader — behavioral evaluation of candidate serving pipelines.

Scoring weights (from task.yaml):
    correctness          40 pts — Classification accuracy on held-out test data
    batch_invariance     20 pts — f([A, B]) == [f(A), f(B)] under unordered batches
    schema_robustness    15 pts — Invariant to JSON key order permutation
    concurrency_safety   10 pts — Thread-safe concurrent execution without state corruption
    performance          10 pts — Batch of 500 items processed within 1.0s
    integrity             5 pts — Public test files unmodified

Total: 100 pts. Pass threshold: 70 pts with no failures on correctness,
batch_invariance, or schema_robustness.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import pathlib
import sys
import time
from typing import Any

import numpy as np

# Repo root for base Grader
_REPO_ROOT = pathlib.Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT))
from grader.base import Grader, GradeResult, TestFailure

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
sys.path.insert(0, str(_DATA_DIR))

WEIGHTS: dict[str, float] = {
    "correctness": 40.0,
    "batch_invariance": 20.0,
    "schema_robustness": 15.0,
    "concurrency_safety": 10.0,
    "performance": 10.0,
    "integrity": 5.0,
}

_PUBLIC_TEST_HASHES: dict[str, str] = {}


def _hash_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _populate_public_hashes() -> None:
    public_dir = pathlib.Path(__file__).parent.parent / "tests" / "public"
    for f in sorted(public_dir.glob("*.py")):
        _PUBLIC_TEST_HASHES[f.name] = _hash_file(f)


_populate_public_hashes()


def _generate_test_samples(n: int = 100, seed: int = 123) -> list[dict[str, Any]]:
    from generate_model import generate_tabular_data

    df, _ = generate_tabular_data(n_samples=n, seed=seed)
    return df.to_dict(orient="records")


def _load_workspace_server(workspace: str):
    path = pathlib.Path(workspace) / "service.py"
    spec = importlib.util.spec_from_file_location("candidate_service", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ModelServer(model_path=pathlib.Path(workspace) / "model.pkl")


class ModelServingGrader(Grader):
    """Grader for the model-serving-v1 benchmark."""

    def grade(self, workspace: str) -> GradeResult:
        components: dict[str, float] = {k: 0.0 for k in WEIGHTS}
        failures: list[TestFailure] = []

        # 1. Load server
        try:
            server = _load_workspace_server(workspace)
        except Exception as exc:
            failures.append(TestFailure("load", f"Cannot instantiate ModelServer: {exc}"))
            return GradeResult(passed=False, score=0.0, components=components, failures=failures)

        # 2. Correctness (40 pts)
        try:
            from generate_model import generate_tabular_data

            df_test, y_test = generate_tabular_data(n_samples=200, seed=777)
            records = df_test.to_dict(orient="records")
            preds = [server.predict_one(r)["prediction"] for r in records]
            acc = float(np.mean(np.array(preds) == y_test))

            if acc >= 0.70:
                components["correctness"] = 40.0
            elif acc >= 0.60:
                components["correctness"] = 25.0
            else:
                failures.append(
                    TestFailure(
                        "correctness",
                        f"Accuracy {acc:.2f} is below acceptable threshold (0.60)",
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("correctness", str(exc)))

        # 3. Batch Invariance (20 pts)
        try:
            samples = _generate_test_samples(n=30, seed=999)
            # Ensure non-monotonic loan_amount to expose ordering bugs
            batch_res = server.predict_batch(samples)
            single_res = [server.predict_one(s) for s in samples]

            mismatches = []
            for i, (b, s) in enumerate(zip(batch_res, single_res)):
                if (
                    b["prediction"] != s["prediction"]
                    or abs(b["probability"] - s["probability"]) > 1e-4
                ):
                    mismatches.append(i)

            if not mismatches:
                components["batch_invariance"] = 20.0
            else:
                failures.append(
                    TestFailure(
                        "batch_invariance",
                        f"Predictions mismatch under batching at indices {mismatches[:5]} "
                        f"(total mismatches: {len(mismatches)}/{len(samples)}). "
                        "Serving must guarantee f([A, B]) == [f(A), f(B)].",
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("batch_invariance", str(exc)))

        # 4. Schema Robustness — Key Permutation Invariance (15 pts)
        try:
            samples = _generate_test_samples(n=20, seed=456)
            permuted_failures = 0
            for item in samples:
                base_pred = server.predict_one(item)

                # Reversed keys
                rev_keys = list(reversed(list(item.keys())))
                rev_item = {k: item[k] for k in rev_keys}
                rev_pred = server.predict_one(rev_item)

                # Alphabetical keys
                alpha_keys = sorted(item.keys())
                alpha_item = {k: item[k] for k in alpha_keys}
                alpha_pred = server.predict_one(alpha_item)

                if (
                    rev_pred["prediction"] != base_pred["prediction"]
                    or alpha_pred["prediction"] != base_pred["prediction"]
                    or abs(rev_pred["probability"] - base_pred["probability"]) > 1e-4
                ):
                    permuted_failures += 1

            if permuted_failures == 0:
                components["schema_robustness"] = 15.0
            else:
                failures.append(
                    TestFailure(
                        "schema_robustness",
                        f"Predictions altered by JSON key permutation on {permuted_failures}/{len(samples)} samples. "
                        "Preprocessor must adhere to canonical feature order.",
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("schema_robustness", str(exc)))

        # 5. Concurrency Safety (10 pts)
        try:
            samples = _generate_test_samples(n=15, seed=111)
            baseline = server.predict_batch(samples)

            def worker():
                return server.predict_batch(samples)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(worker) for _ in range(8)]
                results = [f.result() for f in futures]

            concurrency_ok = True
            for r in results:
                if len(r) != len(baseline):
                    concurrency_ok = False
                    break
                for item_r, item_b in zip(r, baseline):
                    if item_r["prediction"] != item_b["prediction"]:
                        concurrency_ok = False
                        break

            if concurrency_ok:
                components["concurrency_safety"] = 10.0
            else:
                failures.append(
                    TestFailure(
                        "concurrency_safety", "State corruption detected under concurrent execution"
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("concurrency_safety", str(exc)))

        # 6. Performance (10 pts)
        try:
            large_batch = _generate_test_samples(n=500, seed=555)
            t0 = time.perf_counter()
            server.predict_batch(large_batch)
            elapsed = time.perf_counter() - t0

            if elapsed < 1.0:
                components["performance"] = 10.0
            elif elapsed < 2.5:
                components["performance"] = 5.0
                failures.append(
                    TestFailure(
                        "performance", f"Batch of 500 took {elapsed:.2f}s (soft limit 1.0s)"
                    )
                )
            else:
                failures.append(
                    TestFailure(
                        "performance", f"Batch of 500 took {elapsed:.2f}s (hard limit 2.5s)"
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("performance", str(exc)))

        # 7. Integrity (5 pts)
        try:
            ws_path = pathlib.Path(workspace)
            public_dir = ws_path.parent / "tests" / "public"
            tampered = []
            for fname, exp_hash in _PUBLIC_TEST_HASHES.items():
                fpath = public_dir / fname
                if fpath.exists() and _hash_file(fpath) != exp_hash:
                    tampered.append(fname)
            if not tampered:
                components["integrity"] = 5.0
            else:
                failures.append(TestFailure("integrity", f"Public test files modified: {tampered}"))
        except Exception as exc:
            failures.append(TestFailure("integrity", str(exc)))

        total = sum(components.values())
        crit_fails = {f.test_name for f in failures}
        passed = (
            total >= 70.0
            and "correctness" not in crit_fails
            and "batch_invariance" not in crit_fails
            and "schema_robustness" not in crit_fails
        )

        return GradeResult(
            passed=passed,
            score=round(total, 2),
            components=components,
            failures=failures,
        )
