"""Tests for DockerExecutor sandboxing abstractions."""

from runner.executor import DockerExecutor, ExecutionLimits, ExecutionResult


def test_executor_initialization():
    """Verify default and custom limit initialization."""
    default_exec = DockerExecutor()
    assert default_exec.limits.timeout_seconds == 300
    assert default_exec.limits.memory_mb == 4096
    assert default_exec.limits.network_disabled is True

    custom_limits = ExecutionLimits(
        timeout_seconds=60,
        memory_mb=1024,
        cpu_limit=1,
        network_disabled=False,
    )
    custom_exec = DockerExecutor(limits=custom_limits)
    assert custom_exec.limits.timeout_seconds == 60
    assert custom_exec.limits.memory_mb == 1024
    assert custom_exec.limits.cpu_limit == 1
    assert custom_exec.limits.network_disabled is False


def test_execution_result_dataclass():
    """Verify ExecutionResult fields."""
    res = ExecutionResult(
        exit_code=0,
        stdout="Output",
        stderr="",
        duration_seconds=1.25,
        timed_out=False,
    )
    assert res.exit_code == 0
    assert res.stdout == "Output"
    assert res.stderr == ""
    assert res.duration_seconds == 1.25
    assert not res.timed_out
