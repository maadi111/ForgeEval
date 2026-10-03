"""Submission routes — POST /submissions, GET /submissions/{submission_id}."""

import importlib.util
import pathlib
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.models import SubmissionRequest, SubmissionResponse

router = APIRouter(prefix="/submissions", tags=["submissions"])

# In-memory store (replace with DB in production)
_store: dict[str, dict[str, Any]] = {}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _run_grader(submission_id: str, task_id: str, workspace: str) -> None:
    """Locate and execute the appropriate grader, then persist results."""
    _store[submission_id]["status"] = "running"
    try:
        # Discover the grader module for this task
        task_dir = _find_task_dir(task_id)
        if task_dir is None:
            raise FileNotFoundError(f"Task directory for '{task_id}' not found")

        grader_path = task_dir / "grader" / "grader.py"
        spec = importlib.util.spec_from_file_location("task_grader", grader_path)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(grader_path.parent))
        spec.loader.exec_module(mod)

        # Instantiate and run
        grader_class = _find_grader_class(mod)
        result = grader_class().grade(workspace)

        _store[submission_id].update(
            {
                "status": "done",
                "score": result.score,
                "components": result.components,
                "failures": [{"test": f.test_name, "reason": f.reason} for f in result.failures],
                "finished_at": _utcnow(),
            }
        )
    except Exception as exc:
        _store[submission_id].update(
            {
                "status": "failed",
                "failures": [{"test": "system", "reason": str(exc)}],
                "finished_at": _utcnow(),
            }
        )


def _find_task_dir(task_id: str) -> pathlib.Path | None:
    benchmarks = pathlib.Path(__file__).parents[2] / "benchmarks"
    for task_yaml in benchmarks.rglob("task.yaml"):
        import yaml  # lazy import to avoid hard dependency at module load

        with open(task_yaml) as fh:
            data = yaml.safe_load(fh)
        if data.get("id") == task_id:
            return task_yaml.parent
    return None


def _find_grader_class(mod: Any):
    """Return the first Grader subclass found in the module."""
    import inspect

    from grader.base import Grader

    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, Grader) and obj is not Grader:
            return obj
    raise RuntimeError("No Grader subclass found in grader module")


@router.post("", response_model=SubmissionResponse, status_code=202)
def create_submission(
    req: SubmissionRequest, background_tasks: BackgroundTasks
) -> SubmissionResponse:
    """
    Queue a candidate submission for grading.

    The grader runs asynchronously; poll GET /submissions/{id} for results.
    """
    sid = req.submission_id or str(uuid.uuid4())
    now = _utcnow()

    _store[sid] = {
        "submission_id": sid,
        "task_id": req.task_id,
        "status": "pending",
        "score": None,
        "components": None,
        "failures": None,
        "created_at": now,
        "finished_at": None,
    }

    # Use the Docker image name as the workspace path key for now
    # In production: DockerExecutor would run the image and mount the workspace
    background_tasks.add_task(_run_grader, sid, req.task_id, req.image)

    return SubmissionResponse(**_store[sid])


@router.get("/{submission_id}", response_model=SubmissionResponse)
def get_submission(submission_id: str) -> SubmissionResponse:
    """Poll grading status and retrieve score report."""
    if submission_id not in _store:
        raise HTTPException(status_code=404, detail=f"Submission '{submission_id}' not found")
    return SubmissionResponse(**_store[submission_id])
