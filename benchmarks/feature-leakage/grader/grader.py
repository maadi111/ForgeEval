"""
Feature Leakage Grader — behavioral evaluation of a candidate workspace.

Scoring weights (from task.yaml):
    correctness    40 pts   — MAE on held-out data
    regression     15 pts   — all declared feature columns still present
    generalization 15 pts   — works on unseen timestamps
    adversarial    15 pts   — temporal integrity (future-row perturbation)
    performance    10 pts   — feature build completes within 5 s
    integrity       5 pts   — public test files were not modified

Total                100 pts.  Pass threshold: 70 pts with no failures on
                               correctness or adversarial components.
"""

import hashlib
import importlib.util
import pathlib
import sys
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

# grader.base is in the repo root
_REPO_ROOT = pathlib.Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT))
from grader.base import Grader, GradeResult, TestFailure

# Data generator lives alongside this grader's benchmark
_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
sys.path.insert(0, str(_DATA_DIR))

WEIGHTS: dict[str, float] = {
    "correctness": 40.0,
    "regression": 15.0,
    "generalization": 15.0,
    "adversarial": 15.0,
    "performance": 10.0,
    "integrity": 5.0,
}

# SHA-256 hashes of the unmodified public test files
_PUBLIC_TEST_HASHES: dict[str, str] = {}  # populated at import time


def _hash_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _populate_public_hashes() -> None:
    public_dir = pathlib.Path(__file__).parent.parent / "tests" / "public"
    for f in sorted(public_dir.glob("*.py")):
        _PUBLIC_TEST_HASHES[f.name] = _hash_file(f)


_populate_public_hashes()


def _generate(start: str, end: str, seed: int) -> pd.DataFrame:
    from generate_data import generate_sales_data  # noqa: WPS433

    return generate_sales_data(start=start, end=end, seed=seed)


def _load_workspace_module(workspace: str, module_name: str = "features"):
    path = pathlib.Path(workspace) / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FeatureLeakageGrader(Grader):
    """Grades the feature-leakage-v1 benchmark."""

    def grade(self, workspace: str) -> GradeResult:
        components: dict[str, float] = {k: 0.0 for k in WEIGHTS}
        failures: list[TestFailure] = []

        # ------------------------------------------------------------------ #
        # Load candidate module
        # ------------------------------------------------------------------ #
        try:
            mod = _load_workspace_module(workspace)
            build_features: Callable = mod.build_features
            FEATURE_COLS: list[str] = mod.FEATURE_COLS
            TARGET_COL: str = mod.TARGET_COL
        except Exception as exc:
            failures.append(TestFailure("load", f"Cannot import features.py: {exc}"))
            return GradeResult(passed=False, score=0.0, components=components, failures=failures)

        # ------------------------------------------------------------------ #
        # Correctness (40 pts) — MAE on held-out data
        # ------------------------------------------------------------------ #
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # type: ignore

            train_df = _generate("2023-01-01", "2024-06-30", seed=42)
            holdout_df = _generate("2024-07-01", "2024-12-31", seed=42)

            ft_train = build_features(train_df)
            ft_holdout = build_features(holdout_df)

            model = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
            )
            model.fit(ft_train[FEATURE_COLS], ft_train[TARGET_COL])
            preds = model.predict(ft_holdout[FEATURE_COLS])
            mae = float(np.abs(preds - ft_holdout[TARGET_COL].values).mean())

            if mae < 10.0:
                components["correctness"] = 40.0
            elif mae < 15.0:
                components["correctness"] = 30.0
            elif mae < 20.0:
                components["correctness"] = 15.0
            else:
                failures.append(
                    TestFailure(
                        "correctness",
                        f"MAE on holdout = {mae:.2f} (threshold 20.0); "
                        "model likely trained on a leaky pipeline.",
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("correctness", str(exc)))

        # ------------------------------------------------------------------ #
        # Regression (15 pts) — declared feature columns still present
        # ------------------------------------------------------------------ #
        try:
            sample_df = _generate("2023-06-01", "2023-12-31", seed=1)
            featured = build_features(sample_df)
            missing = [c for c in FEATURE_COLS if c not in featured.columns]
            if not missing:
                components["regression"] = 15.0
            else:
                failures.append(TestFailure("regression", f"Missing feature columns: {missing}"))
        except Exception as exc:
            failures.append(TestFailure("regression", str(exc)))

        # ------------------------------------------------------------------ #
        # Generalization (15 pts) — unseen timestamps (seed=99)
        # ------------------------------------------------------------------ #
        try:
            unseen_df = _generate("2025-01-01", "2025-06-30", seed=99)
            featured_unseen = build_features(unseen_df)
            assert len(featured_unseen) > 0, "No rows returned for unseen timestamps"
            missing_unseen = [c for c in FEATURE_COLS if c not in featured_unseen.columns]
            if not missing_unseen:
                components["generalization"] = 15.0
            else:
                failures.append(
                    TestFailure(
                        "generalization", f"Missing columns on unseen data: {missing_unseen}"
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("generalization", str(exc)))

        # ------------------------------------------------------------------ #
        # Adversarial (15 pts) — temporal integrity via future-row perturbation
        # ------------------------------------------------------------------ #
        try:
            base = _generate("2023-01-01", "2024-06-30", seed=42)
            base = base.sort_values("date").reset_index(drop=True)
            n = len(base)
            pivot = n // 2
            cutoff = base["date"].iloc[pivot]

            ft_clean = build_features(base)

            perturbed = base.copy()
            perturbed.loc[pivot:, "sales"] += 9999
            ft_perturbed = build_features(perturbed)

            orig_sub = ft_clean[ft_clean["date"].astype(str) <= str(cutoff)]
            pert_sub = ft_perturbed[ft_perturbed["date"].astype(str) <= str(cutoff)]
            merged = orig_sub.merge(pert_sub, on="date", suffixes=("_o", "_p"))

            leak_cols = []
            for col in FEATURE_COLS:
                co, cp = f"{col}_o", f"{col}_p"
                if co in merged.columns:
                    max_diff = (merged[co] - merged[cp]).abs().max()
                    if max_diff > 1e-6:
                        leak_cols.append(f"{col}(diff={max_diff:.2f})")

            if not leak_cols:
                components["adversarial"] = 15.0
            else:
                failures.append(
                    TestFailure(
                        "adversarial",
                        "Feature(s) leak future data: " + ", ".join(leak_cols),
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("adversarial", str(exc)))

        # ------------------------------------------------------------------ #
        # Performance (10 pts) — build_features on 6 years of data in < 5 s
        # ------------------------------------------------------------------ #
        try:
            large_df = _generate("2020-01-01", "2025-12-31", seed=7)
            t0 = time.perf_counter()
            build_features(large_df)
            elapsed = time.perf_counter() - t0

            if elapsed < 5.0:
                components["performance"] = 10.0
            elif elapsed < 10.0:
                components["performance"] = 5.0
                failures.append(
                    TestFailure(
                        "performance", f"build_features took {elapsed:.1f}s (soft limit 5s)"
                    )
                )
            else:
                failures.append(
                    TestFailure(
                        "performance", f"build_features took {elapsed:.1f}s (hard limit 10s)"
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("performance", str(exc)))

        # ------------------------------------------------------------------ #
        # Integrity (5 pts) — public test files not modified
        # ------------------------------------------------------------------ #
        try:
            ws_path = pathlib.Path(workspace)
            public_test_dir = ws_path.parent / "tests" / "public"
            tampered = []
            for fname, expected_hash in _PUBLIC_TEST_HASHES.items():
                candidate_file = public_test_dir / fname
                if candidate_file.exists():
                    actual = _hash_file(candidate_file)
                    if actual != expected_hash:
                        tampered.append(fname)
            if not tampered:
                components["integrity"] = 5.0
            else:
                failures.append(TestFailure("integrity", f"Modified public test files: {tampered}"))
        except Exception as exc:
            failures.append(TestFailure("integrity", str(exc)))

        # ------------------------------------------------------------------ #
        # Final verdict
        # ------------------------------------------------------------------ #
        total = sum(components.values())
        critical_failures = {f.test_name for f in failures}
        passed = (
            total >= 70.0
            and "correctness" not in critical_failures
            and "adversarial" not in critical_failures
        )

        return GradeResult(
            passed=passed,
            score=round(total, 2),
            components=components,
            failures=failures,
        )
