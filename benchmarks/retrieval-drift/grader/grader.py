"""
Retrieval Drift Grader — behavioral evaluation of semantic retrieval systems.

Scoring weights (from task.yaml):
    recall_at_k               40 pts — Overall Recall@5 across all benchmark queries
    drift_robustness          20 pts — Recall@5 specifically on drifted domain queries
    precision_at_k            15 pts — Top-1 Precision on drifted domain queries
    normalization_integrity   10 pts — Cosine boundedness & scale invariance
    latency                   10 pts — Average retrieval latency < 15ms per query
    integrity                  5 pts — Public test files unmodified

Total: 100 pts. Pass threshold: 70 pts with no failures on recall_at_k or drift_robustness.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import time

# Repo root for base Grader
_REPO_ROOT = pathlib.Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT))
from grader.base import Grader, GradeResult, TestFailure

_BENCHMARK_DIR = pathlib.Path(__file__).parent.parent
_DATA_DIR = _BENCHMARK_DIR / "data"

WEIGHTS: dict[str, float] = {
    "recall_at_k": 40.0,
    "drift_robustness": 20.0,
    "precision_at_k": 15.0,
    "normalization_integrity": 10.0,
    "latency": 10.0,
    "integrity": 5.0,
}

_PUBLIC_TEST_HASHES: dict[str, str] = {}


def _hash_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _populate_public_hashes() -> None:
    public_dir = _BENCHMARK_DIR / "tests" / "public"
    for f in sorted(public_dir.glob("*.py")):
        _PUBLIC_TEST_HASHES[f.name] = _hash_file(f)


_populate_public_hashes()


def _load_workspace_retriever(workspace: str):
    path = pathlib.Path(workspace) / "retriever.py"
    spec = importlib.util.spec_from_file_location("candidate_retriever", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DenseRetriever(corpus_path=_DATA_DIR / "corpus.json")


class RetrievalDriftGrader(Grader):
    """Grader for the retrieval-drift-v1 benchmark."""

    def grade(self, workspace: str) -> GradeResult:
        components: dict[str, float] = {k: 0.0 for k in WEIGHTS}
        failures: list[TestFailure] = []

        # 1. Load retriever
        try:
            retriever = _load_workspace_retriever(workspace)
        except Exception as exc:
            failures.append(TestFailure("load", f"Cannot instantiate DenseRetriever: {exc}"))
            return GradeResult(passed=False, score=0.0, components=components, failures=failures)

        with open(_DATA_DIR / "queries.json", "r", encoding="utf-8") as f:
            query_data = json.load(f)

        historical = query_data["historical"]
        drifted = query_data["drifted"]
        all_queries = historical + drifted

        # 2. Overall Recall@5 (40 pts)
        try:
            hits = 0
            for item in all_queries:
                res = retriever.retrieve(item["query"], top_k=5)
                retrieved_ids = {r["id"] for r in res}
                if set(item["relevant_ids"]).intersection(retrieved_ids):
                    hits += 1

            recall = hits / len(all_queries)
            if recall >= 0.80:
                components["recall_at_k"] = 40.0
            elif recall >= 0.60:
                components["recall_at_k"] = 25.0
            else:
                failures.append(
                    TestFailure("recall_at_k", f"Overall Recall@5 is {recall:.1%} (threshold 60%)")
                )
        except Exception as exc:
            failures.append(TestFailure("recall_at_k", str(exc)))

        # 3. Drift Robustness (20 pts)
        try:
            drift_hits = 0
            for item in drifted:
                res = retriever.retrieve(item["query"], top_k=5)
                retrieved_ids = {r["id"] for r in res}
                if set(item["relevant_ids"]).intersection(retrieved_ids):
                    drift_hits += 1

            drift_recall = drift_hits / len(drifted)
            if drift_recall >= 0.80:
                components["drift_robustness"] = 20.0
            elif drift_recall >= 0.60:
                components["drift_robustness"] = 10.0
                failures.append(
                    TestFailure(
                        "drift_robustness",
                        f"Drift Recall@5 is degraded: {drift_recall:.1%} (soft threshold 80%)",
                    )
                )
            else:
                failures.append(
                    TestFailure(
                        "drift_robustness",
                        f"Severe drift collapse: Recall@5 is {drift_recall:.1%} (threshold 60%)",
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("drift_robustness", str(exc)))

        # 4. Precision@1 on Drifted Domain (15 pts)
        try:
            p1_hits = 0
            for item in drifted:
                res = retriever.retrieve(item["query"], top_k=1)
                if res and res[0]["id"] in item["relevant_ids"]:
                    p1_hits += 1

            p1 = p1_hits / len(drifted)
            if p1 >= 0.60:
                components["precision_at_k"] = 15.0
            elif p1 >= 0.40:
                components["precision_at_k"] = 8.0
            else:
                failures.append(
                    TestFailure(
                        "precision_at_k",
                        f"Top-1 Precision on drifted queries = {p1:.1%} (threshold 40%)",
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("precision_at_k", str(exc)))

        # 5. Normalization Integrity (10 pts)
        try:
            sample_res = retriever.retrieve("kubernetes gpu deployment", top_k=5)
            bounded = all(r["score"] <= 1.0001 for r in sample_res)
            if bounded:
                components["normalization_integrity"] = 10.0
            else:
                failures.append(
                    TestFailure(
                        "normalization_integrity",
                        "Scores exceed 1.0; embeddings appear unnormalized",
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("normalization_integrity", str(exc)))

        # 6. Latency (10 pts) — 10 queries executed in < 0.20s total (< 20ms/query)
        try:
            t0 = time.perf_counter()
            for item in all_queries:
                retriever.retrieve(item["query"], top_k=5)
            elapsed = time.perf_counter() - t0

            if elapsed < 0.20:
                components["latency"] = 10.0
            elif elapsed < 0.50:
                components["latency"] = 5.0
                failures.append(
                    TestFailure(
                        "latency", f"10 queries took {elapsed * 1000:.1f}ms (soft limit 200ms)"
                    )
                )
            else:
                failures.append(
                    TestFailure(
                        "latency", f"10 queries took {elapsed * 1000:.1f}ms (hard limit 500ms)"
                    )
                )
        except Exception as exc:
            failures.append(TestFailure("latency", str(exc)))

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
            and "recall_at_k" not in crit_fails
            and "drift_robustness" not in crit_fails
        )

        return GradeResult(
            passed=passed,
            score=round(total, 2),
            components=components,
            failures=failures,
        )
