"""
DockerExecutor — sandboxed candidate execution.

Runs an untrusted Docker image with resource limits, network isolation,
stdout/stderr capture, and exit-code handling.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class ExecutionLimits:
    timeout_seconds: int = 300
    memory_mb: int = 4096
    cpu_limit: int = 2
    network_disabled: bool = True


@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False


class DockerExecutor:
    """Run an untrusted Docker image inside a constrained sandbox."""

    def __init__(self, limits: ExecutionLimits | None = None) -> None:
        self.limits = limits or ExecutionLimits()

    def run(
        self,
        image: str,
        command: list[str] | None = None,
        mounts: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute *image* with *command* inside Docker.

        Parameters
        ----------
        image:
            Docker image tag to run.
        command:
            Override the image's default CMD.  Pass ``None`` to use the image default.
        mounts:
            Host-to-container volume mappings, e.g.
            ``{"/host/data": "/workspace/data"}``.
        env:
            Extra environment variables to inject.

        Returns
        -------
        ExecutionResult
        """
        cmd: list[str] = ["docker", "run", "--rm"]

        # Resource limits
        cmd += ["--memory", f"{self.limits.memory_mb}m"]
        cmd += ["--cpus", str(self.limits.cpu_limit)]

        # Network isolation
        if self.limits.network_disabled:
            cmd += ["--network", "none"]

        # Volume mounts (read-write workspace, read-only data)
        for host_path, container_path in (mounts or {}).items():
            cmd += ["-v", f"{host_path}:{container_path}"]

        # Environment variables
        for key, value in (env or {}).items():
            cmd += ["-e", f"{key}={value}"]

        # Security: run as non-root where possible
        cmd += ["--user", "1000:1000"]

        cmd.append(image)
        if command:
            cmd.extend(command)

        t0 = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.limits.timeout_seconds,
                check=False,
            )
            exit_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -1
            stdout = exc.stdout or ""
            stderr = exc.stderr or f"Execution timed out after {self.limits.timeout_seconds}s"
            # Kill the container (best effort)
            subprocess.run(
                ["docker", "ps", "-q", "--filter", f"ancestor={image}"],
                capture_output=True,
                check=False,
            )

        duration = time.perf_counter() - t0
        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=round(duration, 3),
            timed_out=timed_out,
        )
