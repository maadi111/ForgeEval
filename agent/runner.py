"""Agent Evaluation Harness — runs AI agents against ForgeEval benchmarks."""

import argparse
import importlib.util
import pathlib
import shutil
import sys
import tempfile
import time
from typing import Any

import yaml

from agent.mock_agent import HeuristicBenchmarkAgent
from agent.protocol import AgentMetrics, AgentRunResult, BenchmarkAgent

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class AgentEvaluationHarness:
    """Evaluates benchmark agents in isolated sandboxed workspaces."""

    def __init__(self, repo_root: pathlib.Path = REPO_ROOT):
        self.repo_root = repo_root
        self.benchmarks_dir = repo_root / "benchmarks"

    def run_task(
        self,
        task_id: str,
        agent: BenchmarkAgent,
        max_iterations: int = 5,
    ) -> AgentRunResult:
        """Run agent against specified task in an isolated workspace."""
        # Locate task
        task_dir = None
        for b_dir in self.benchmarks_dir.iterdir():
            if b_dir.is_dir() and (b_dir / "task.yaml").exists():
                with open(b_dir / "task.yaml", "r", encoding="utf-8") as f:
                    spec = yaml.safe_load(f)
                    if spec.get("id") == task_id or b_dir.name == task_id:
                        task_dir = b_dir
                        task_spec = dict(spec)
                        task_spec["benchmark_dir"] = str(task_dir)
                        task_spec["repo_root"] = str(self.repo_root)
                        break

        if task_dir is None:
            raise ValueError(f"Task '{task_id}' not found in {self.benchmarks_dir}")

        src_ws = task_dir / task_spec.get("paths", {}).get("workspace", "workspace")

        # Create isolated temporary workspace
        temp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"forge_agent_{task_id}_"))
        temp_ws = temp_dir / "workspace"
        shutil.copytree(src_ws, temp_ws)

        # Copy data folder if present
        if (task_dir / "data").exists():
            shutil.copytree(task_dir / "data", temp_dir / "data")

        t0 = time.perf_counter()
        try:
            # Execute agent
            trajectory = agent.solve_task(
                task_id=task_id,
                workspace_dir=str(temp_ws),
                task_spec=task_spec,
                max_iterations=max_iterations,
            )
            duration = time.perf_counter() - t0

            # Grade the agent's resulting workspace
            grader_path = task_dir / task_spec.get("paths", {}).get("grader", "grader/grader.py")
            grade_result = self._grade_workspace(grader_path, str(temp_ws))

            # Assemble metrics
            metrics = AgentMetrics(
                total_iterations=len({t.iteration for t in trajectory}),
                total_duration_seconds=round(duration, 3),
                public_tests_passed=True,
                adversarial_tests_passed=grade_result.get("passed", False),
                forgeeval_score=grade_result.get("score", 0.0),
                tool_calls_count=len(trajectory),
            )

            result = AgentRunResult(
                task_id=task_id,
                agent_name=agent.name,
                success=grade_result.get("passed", False),
                score=grade_result.get("score", 0.0),
                grade_report=grade_result,
                metrics=metrics,
                trajectory=trajectory,
            )

            return result

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _grade_workspace(self, grader_file: pathlib.Path, ws_path: str) -> dict[str, Any]:
        """Dynamically invoke task grader."""
        sys.path.insert(0, str(grader_file.parent))
        try:
            spec = importlib.util.spec_from_file_location("dynamic_grader", grader_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load grader at {grader_file}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Find Grader class
            grader_cls = None
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and attr.endswith("Grader") and attr != "Grader":
                    grader_cls = obj
                    break

            if grader_cls is None:
                raise RuntimeError(f"No Grader class found in {grader_file}")

            grader = grader_cls()
            res = grader.grade(ws_path)
            return {
                "score": res.score,
                "passed": res.passed,
                "components": res.components,
                "failures": [f"{f.test_name}: {f.reason}" for f in res.failures],
            }
        finally:
            sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser(description="ForgeEval AI Agent Runner")
    parser.add_argument("--task", default="feature-leakage-v1", help="Task ID to evaluate")
    parser.add_argument("--max-iters", type=int, default=5, help="Maximum agent iterations")
    parser.add_argument("--output", default=None, help="Path to save result JSON")
    args = parser.parse_args()

    harness = AgentEvaluationHarness()
    agent = HeuristicBenchmarkAgent()

    print("=" * 60)
    print(f" ForgeEval Agent Evaluation: {args.task}")
    print(f" Agent: {agent.name}")
    print("=" * 60)

    res = harness.run_task(task_id=args.task, agent=agent, max_iterations=args.max_iters)

    print(f"Status:   {'PASSED [OK]' if res.success else 'FAILED [X]'}")
    print(f"Score:    {res.score:.1f} / 100.0")
    print(f"Duration: {res.metrics.total_duration_seconds:.2f}s")
    print(f"Steps:    {res.metrics.tool_calls_count} actions")
    print("-" * 60)
    print("Component Breakdown:")
    for k, v in res.grade_report.get("components", {}).items():
        print(f"  * {k:<18}: {v:5.1f} pts")
    print("=" * 60)

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(res.model_dump_json(indent=2))
        print(f"Saved run log -> {out_path}")


if __name__ == "__main__":
    main()
