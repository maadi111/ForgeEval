"""Data contracts and schemas for the evaluation platform."""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class PredictorProtocol(Protocol):
    """Protocol for candidate models evaluated by the harness."""

    def predict(self, X: list[list[float]]) -> list[int]:
        ...

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        ...


class EvaluationReport(BaseModel):
    """Detailed evaluation report returned by the grading harness."""

    passed: bool
    score: float = Field(ge=0.0, le=100.0)
    metrics: dict[str, float] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    tamper_detected: bool = False
    invariance_verified: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
