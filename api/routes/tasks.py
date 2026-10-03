"""Task registry routes — GET /tasks, GET /tasks/{task_id}."""

import pathlib

import yaml
from fastapi import APIRouter, HTTPException

from api.models import TaskListResponse, TaskSchema

router = APIRouter(prefix="/tasks", tags=["tasks"])

_TASK_ROOT = pathlib.Path(__file__).parents[2] / "benchmarks"


def _load_all_tasks() -> list[TaskSchema]:
    tasks: list[TaskSchema] = []
    for task_yaml in _TASK_ROOT.rglob("task.yaml"):
        try:
            with open(task_yaml) as fh:
                data = yaml.safe_load(fh)
            tasks.append(
                TaskSchema(
                    id=data["id"],
                    title=data["title"],
                    version=data["version"],
                    difficulty=data["difficulty"],
                    category=data["category"],
                    description=data.get("description", "").strip(),
                )
            )
        except Exception:
            pass  # skip malformed task files
    return tasks


@router.get("", response_model=TaskListResponse)
def list_tasks() -> TaskListResponse:
    """Return all registered benchmark tasks."""
    tasks = _load_all_tasks()
    return TaskListResponse(tasks=tasks, total=len(tasks))


@router.get("/{task_id}", response_model=TaskSchema)
def get_task(task_id: str) -> TaskSchema:
    """Return a single task by ID."""
    for task in _load_all_tasks():
        if task.id == task_id:
            return task
    raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
