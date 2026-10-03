"""Protocols, data contracts, and trajectory models for AI agents."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class AgentTrajectoryStep(BaseModel):
    """A single step in the agent's reasoning/action trajectory."""

    iteration: int
    action: str
    target_file: str | None = None
    command: str | None = None
    diff: str | None = None
    observation: str | None = None
    timestamp: float


class AgentMetrics(BaseModel):
    """Execution and cost metrics for the agent run."""

    total_iterations: int = 0
    total_duration_seconds: float = 0.0
    public_tests_passed: bool = False
    adversarial_tests_passed: bool = False
    forgeeval_score: float = 0.0
    tool_calls_count: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)


class AgentRunResult(BaseModel):
    """Complete evaluation outcome for an agent on a ForgeEval benchmark task."""

    task_id: str
    agent_name: str
    success: bool
    score: float
    grade_report: dict[str, Any]
    metrics: AgentMetrics
    trajectory: list[AgentTrajectoryStep] = Field(default_factory=list)


class BenchmarkAgent(ABC):
    """Abstract interface for benchmark agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name/version of the agent."""
        ...

    @abstractmethod
    def solve_task(
        self,
        task_id: str,
        workspace_dir: str,
        task_spec: dict[str, Any],
        max_iterations: int = 5,
    ) -> list[AgentTrajectoryStep]:
        """Inspect, modify, test, and iterate on the candidate workspace."""
        ...
