"""CLI runner for evaluating candidate workspaces against ForgeEval benchmarks."""

import argparse
import importlib.util
import inspect
import json
import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from grader.base import Grader, GradeResult


def find_benchmark_dir(task_id: str) -> pathlib.Path:
    benchmarks_dir = REPO_ROOT / "benchmarks"
    for task_yaml in benchmarks_dir.rglob("task.yaml"):
        with open(task_yaml) as fh:
            data = yaml.safe_load(fh)
        if data.get("id") == task_id:
            return task_yaml.parent
    raise ValueError(f"Task '{task_id}' not found in benchmarks/")


def load_grader(benchmark_dir: pathlib.Path) -> Grader:
    grader_file = benchmark_dir / "grader" / "grader.py"
    if not grader_file.exists():
        raise FileNotFoundError(f"Grader file not found at {grader_file}")

    spec = importlib.util.spec_from_file_location("benchmark_grader", grader_file)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(benchmark_dir / "grader"))
    sys.path.insert(0, str(benchmark_dir / "data"))
    spec.loader.exec_module(mod)

    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, Grader) and obj is not Grader:
            return obj()
    raise RuntimeError("No Grader subclass found in grader.py")


def print_report(task_id: str, workspace: str, result: GradeResult) -> None:
    print("=" * 60)
    print(f" ForgeEval Evaluation Report: {task_id}")
    print("=" * 60)
    print(f"Workspace: {workspace}")
    print(f"Status:    {'PASSED [OK]' if result.passed else 'FAILED [X]'}")
    print(f"Score:     {result.score:.1f} / 100.0")
    print("-" * 60)
    print("Component Breakdown:")
    for comp, score in result.components.items():
        print(f"  * {comp:<16} : {score:5.1f} pts")
    print("-" * 60)
    if result.failures:
        print("Failures & Warnings:")
        for fail in result.failures:
            print(f"  [!] [{fail.test_name}] {fail.reason}")
    else:
        print("No test failures detected.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="ForgeEval Benchmark Runner")
    parser.add_argument("--task", required=True, help="Task ID (e.g., feature-leakage-v1)")
    parser.add_argument("--workspace", required=True, help="Path to workspace directory")
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of text report"
    )
    args = parser.parse_args()

    bench_dir = find_benchmark_dir(args.task)
    grader = load_grader(bench_dir)
    result = grader.grade(str(pathlib.Path(args.workspace).resolve()))

    if args.json:
        report = {
            "task_id": args.task,
            "workspace": args.workspace,
            "passed": result.passed,
            "score": result.score,
            "components": result.components,
            "failures": [{"test": f.test_name, "reason": f.reason} for f in result.failures],
        }
        print(json.dumps(report, indent=2))
    else:
        print_report(args.task, args.workspace, result)

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
