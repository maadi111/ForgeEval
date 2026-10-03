"""Adversarial Grading Grader — behavioral evaluation of candidate evaluation harness.

Scoring weights:
    correctness          40 pts — Accepts genuine generalizing ML models (Logistic Regression & Random Forest)
    regression           15 pts — Preserves public ModelEvaluator API contract and EvaluationReport structure
    generalization       15 pts — Strict memory isolation: candidate cannot mutate evaluation fixtures
    adversarial          15 pts — Systematically rejects memorization, constant prediction, order exploits, and tampering
    performance          10 pts — Evaluation latency across 500 samples < 2.0s
    integrity             5 pts — Public test fixtures unmodified

Total: 100 pts. Pass threshold: 70 pts with no failures on correctness or adversarial tests.
"""

from __future__ import annotations

import copy
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

from baseline_models import (
    ConstantPredictorProbe,
    MemorizedLookupProbe,
    OrderExploitProbe,
    TamperingProbe,
    train_genuine_models,
)

WEIGHTS: dict[str, float] = {
    "correctness": 40.0,
    "regression": 15.0,
    "generalization": 15.0,
    "adversarial": 15.0,
    "performance": 10.0,
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


class AdversarialGrader(Grader):
    """Behavioral and adversarial grader for B5: Adversarial Grading benchmark."""

    def grade(self, workspace: str) -> GradeResult:
        ws_path = pathlib.Path(workspace).resolve()
        failures: list[TestFailure] = []
        components: dict[str, float] = {k: 0.0 for k in WEIGHTS}

        # Load probe dataset
        data_file = _DATA_DIR / "probe_dataset.json"
        if not data_file.exists():
            from generate_probes import generate_benchmark_assets

            generate_benchmark_assets(_DATA_DIR)

        with open(data_file, "r", encoding="utf-8") as f:
            probe_dataset = json.load(f)

        # Dynamic import of candidate workspace
        for mod_name in ["evaluator", "security", "models"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        sys.path.insert(0, str(ws_path))
        try:
            eval_spec = importlib.util.spec_from_file_location("evaluator", ws_path / "evaluator.py")
            if eval_spec is None or eval_spec.loader is None:
                raise ImportError(f"Could not load evaluator from {ws_path / 'evaluator.py'}")
            eval_mod = importlib.util.module_from_spec(eval_spec)
            sys.modules["evaluator"] = eval_mod
            eval_spec.loader.exec_module(eval_mod)
            ModelEvaluator = eval_mod.ModelEvaluator
        except Exception as e:
            failures.append(TestFailure("import", f"Failed to import candidate workspace: {e}"))
            return GradeResult(passed=False, score=0.0, components=components, failures=failures)
        finally:
            sys.path.pop(0)

        # ------------------------------------------------------------------ #
        # 1. Regression & Contract Verification (15 pts)
        # ------------------------------------------------------------------ #
        try:
            ds_sample = copy.deepcopy(probe_dataset)
            evaluator_instance = ModelEvaluator(ds_sample)
            if hasattr(evaluator_instance, "evaluate"):
                components["regression"] = 15.0
            else:
                failures.append(TestFailure("regression", "ModelEvaluator missing evaluate() method"))
        except Exception as e:
            failures.append(TestFailure("regression", f"Evaluator initialization failed: {e}"))

        # ------------------------------------------------------------------ #
        # 2. Correctness: Genuine Baseline Acceptance (40 pts)
        # ------------------------------------------------------------------ #
        t_eval = 999.0
        try:
            lr_model, rf_model = train_genuine_models(probe_dataset)

            t0 = time.perf_counter()
            report_lr = evaluator_instance.evaluate(lr_model)
            t_eval = time.perf_counter() - t0

            report_rf = evaluator_instance.evaluate(rf_model)

            lr_ok = (getattr(report_lr, "passed", False) is True) and (getattr(report_lr, "score", 0.0) >= 75.0)
            rf_ok = (getattr(report_rf, "passed", False) is True) and (getattr(report_rf, "score", 0.0) >= 75.0)

            if lr_ok and rf_ok:
                components["correctness"] = 40.0
            elif lr_ok or rf_ok:
                components["correctness"] = 20.0
                failures.append(TestFailure("correctness", "Only 1 of 2 genuine baseline models achieved passing grade"))
            else:
                failures.append(
                    TestFailure(
                        "correctness",
                        f"Genuine models rejected by evaluator (LR score={getattr(report_lr, 'score', 0)}, RF score={getattr(report_rf, 'score', 0)})",
                    )
                )
        except Exception as e:
            failures.append(TestFailure("correctness", f"Genuine baseline model evaluation failed: {e}"))

        # ------------------------------------------------------------------ #
        # 3. Generalization: Defensive Isolation & State Immutability (15 pts)
        # ------------------------------------------------------------------ #
        try:
            ds_test = copy.deepcopy(probe_dataset)
            initial_val = ds_test["records"][0]["features"][0]
            eval_isolated = ModelEvaluator(ds_test)
            tamper_probe = TamperingProbe()
            _ = eval_isolated.evaluate(tamper_probe)
            post_val = ds_test["records"][0]["features"][0]

            if post_val == initial_val:
                components["generalization"] = 15.0
            else:
                failures.append(
                    TestFailure(
                        "generalization",
                        "Evaluator failed defensive isolation: untrusted model mutated evaluation dataset in memory",
                    )
                )
        except Exception as e:
            failures.append(TestFailure("generalization", f"Generalization / isolation check failed: {e}"))

        # ------------------------------------------------------------------ #
        # 4. Adversarial Probes: Catch Shortcuts & Exploits (15 pts)
        # ------------------------------------------------------------------ #
        adv_fails = []
        try:
            eval_adv = ModelEvaluator(copy.deepcopy(probe_dataset))

            # Probe A: Memorization shortcut
            mem_probe = MemorizedLookupProbe(probe_dataset)
            rep_mem = eval_adv.evaluate(mem_probe)
            if getattr(rep_mem, "passed", False) is True:
                adv_fails.append("Evaluator passed a memorized lookup shortcut (must reject)")

            # Probe B: Constant 0 predictor
            c0_probe = ConstantPredictorProbe(0)
            rep_c0 = eval_adv.evaluate(c0_probe)
            if getattr(rep_c0, "passed", False) is True or getattr(rep_c0, "score", 0.0) >= 75.0:
                adv_fails.append("Evaluator passed a constant predictor shortcut (must reject)")

            # Probe C: Order exploit
            labels = [r["label"] for r in probe_dataset["records"]]
            ord_probe = OrderExploitProbe(labels)
            rep_ord = eval_adv.evaluate(ord_probe)
            if getattr(rep_ord, "passed", False) is True:
                adv_fails.append("Evaluator passed an order-dependent exploit (must reject)")

            # Probe D: Tamper probe
            rep_tmp = eval_adv.evaluate(TamperingProbe())
            if getattr(rep_tmp, "passed", False) is True:
                adv_fails.append("Evaluator passed a test-tampering probe (must reject)")

            if not adv_fails:
                components["adversarial"] = 15.0
            else:
                for msg in adv_fails:
                    failures.append(TestFailure("adversarial", msg))
                components["adversarial"] = max(0.0, 15.0 - len(adv_fails) * 3.75)

        except Exception as e:
            failures.append(TestFailure("adversarial", f"Adversarial probe testing error: {e}"))

        # ------------------------------------------------------------------ #
        # 5. Performance Check (10 pts)
        # ------------------------------------------------------------------ #
        if t_eval < 2.0:
            components["performance"] = 10.0
        elif t_eval < 5.0:
            components["performance"] = 5.0
            failures.append(TestFailure("performance", f"Evaluation took {t_eval:.2f}s (soft limit 2.0s)"))
        else:
            failures.append(TestFailure("performance", f"Evaluation took {t_eval:.2f}s (hard limit 5.0s)"))

        # ------------------------------------------------------------------ #
        # 6. Integrity: Public test suite cryptographic check (5 pts)
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
            and "adversarial" not in critical_failed
        )

        return GradeResult(
            passed=passed,
            score=round(total, 1),
            components=components,
            failures=failures,
        )


def grade_submission(workspace_dir: pathlib.Path) -> dict[str, Any]:
    grader = AdversarialGrader()
    res = grader.grade(str(workspace_dir))
    return {
        "task_id": "adversarial-grading-v1",
        "score": res.score,
        "passed": res.passed,
        "components": res.components,
        "failures": [f"{f.test_name}: {f.reason}" for f in res.failures],
    }


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
    res = grade_submission(target)
    print(json.dumps(res, indent=2))
