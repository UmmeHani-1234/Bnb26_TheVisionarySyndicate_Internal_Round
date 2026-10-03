"""Base classes and data contracts for failure intelligence and localization."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    """A concrete observable fact supporting a failure diagnosis."""

    type: str  # e.g. "constraint_violation", "tool_error", "data_mismatch"
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepSuspicion:
    """Suspicion score assigned to a specific execution step."""

    step_id: int
    step_type: str
    tool_name: Optional[str]
    score: float  # Normalized between 0.0 and 1.0
    status: str = "success"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisResult:
    """Structured diagnosis of an agent execution trace."""

    run_id: str
    failure_detected: bool
    failure_type: str  # e.g. "budget_violation", "tool_failure", "unexpected_output", "none"
    likely_responsible_step: Optional[StepSuspicion]
    top_suspected_steps: List[StepSuspicion] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "failure_detected": self.failure_detected,
            "failure_type": self.failure_type,
            "likely_responsible_step": self.likely_responsible_step.to_dict() if self.likely_responsible_step else None,
            "top_suspected_steps": [s.to_dict() for s in self.top_suspected_steps],
            "evidence": [e.to_dict() for e in self.evidence],
            "evidence_descriptions": [e.description for e in self.evidence],
            "confidence": round(self.confidence, 2),
            "summary": self.summary,
        }


class BaseLocalizer(ABC):
    """Abstract interface for failure localization algorithms.
    
    Permits rule-based localizers now and future ML localizers (MLLocalizer)
    without changing backend endpoints or viewer interfaces.
    """

    @abstractmethod
    def diagnose(self, trace_dict: Dict[str, Any]) -> DiagnosisResult:
        """Inspects an observable trace dict and returns a structured diagnosis."""
        pass
