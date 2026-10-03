from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TestFailure:
    test_name: str
    reason: str


@dataclass
class GradeResult:
    passed: bool
    score: float
    components: dict[str, float] = field(default_factory=dict)
    failures: list[TestFailure] = field(default_factory=list)


class Grader(ABC):
    @abstractmethod
    def grade(self, workspace: str) -> GradeResult:
        raise NotImplementedError
