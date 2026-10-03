"""API data models (Pydantic schemas)."""

from datetime import datetime

from pydantic import BaseModel, Field


class TaskSchema(BaseModel):
    id: str
    title: str
    version: str
    difficulty: str
    category: str
    description: str


class TaskListResponse(BaseModel):
    tasks: list[TaskSchema]
    total: int


class SubmissionRequest(BaseModel):
    task_id: str = Field(..., description="Task identifier, e.g. feature-leakage-v1")
    image: str = Field(..., description="Docker image tag to execute")
    submission_id: str | None = Field(None, description="Optional client-supplied idempotency key")


class SubmissionResponse(BaseModel):
    submission_id: str
    task_id: str
    status: str  # pending | running | done | failed
    score: float | None = None
    components: dict[str, float] | None = None
    failures: list[dict] | None = None
    created_at: datetime
    finished_at: datetime | None = None
