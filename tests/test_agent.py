"""Tests for the AI Agent Evaluation Harness."""

from agent.mock_agent import HeuristicBenchmarkAgent
from agent.protocol import AgentRunResult
from agent.runner import AgentEvaluationHarness


def test_agent_eval_feature_leakage():
    """Verify that HeuristicBenchmarkAgent solves feature-leakage-v1 and passes grading."""
    harness = AgentEvaluationHarness()
    agent = HeuristicBenchmarkAgent()

    result = harness.run_task("feature-leakage-v1", agent=agent, max_iterations=3)

    assert isinstance(result, AgentRunResult)
    assert result.task_id == "feature-leakage-v1"
    assert result.success is True
    assert result.score >= 85.0
    assert result.metrics.total_iterations >= 2
    assert len(result.trajectory) >= 3


def test_agent_eval_model_serving():
    """Verify that HeuristicBenchmarkAgent solves model-serving-v1 and passes grading."""
    harness = AgentEvaluationHarness()
    agent = HeuristicBenchmarkAgent()

    result = harness.run_task("model-serving-v1", agent=agent, max_iterations=3)

    assert isinstance(result, AgentRunResult)
    assert result.task_id == "model-serving-v1"
    assert result.success is True
    assert result.score >= 85.0
