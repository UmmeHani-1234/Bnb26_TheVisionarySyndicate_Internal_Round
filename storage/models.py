"""SQLAlchemy models for Run and ExecutionStep trace storage."""

from typing import Any, Dict, Optional
from sqlalchemy import Column, String, Integer, Float, Text, JSON, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from .database import Base


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Run(Base):
    """Represents a single agent execution trace."""

    __tablename__ = "runs"

    run_id = Column(String(64), primary_key=True, index=True)
    started_at = Column(String(64), default=utc_now_iso, nullable=False)
    ended_at = Column(String(64), nullable=True)
    status = Column(String(32), default="running", nullable=False)  # "running", "success", "failed"
    user_request = Column(Text, nullable=False)
    final_output = Column(Text, nullable=True)

    steps = relationship(
        "ExecutionStep",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ExecutionStep.step_id",
    )

    def to_dict(self, include_steps: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "user_request": self.user_request,
            "final_output": self.final_output,
        }
        if include_steps:
            data["steps"] = [step.to_dict() for step in self.steps]
        return data


class ExecutionStep(Base):
    """Represents an ordered observable execution step belonging to a run."""

    __tablename__ = "execution_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    step_id = Column(Integer, nullable=False, index=True)
    run_id = Column(String(64), ForeignKey("runs.run_id"), nullable=False, index=True)
    event_id = Column(String(64), nullable=False, index=True)
    step_type = Column(String(64), nullable=False)
    tool_name = Column(String(64), nullable=True)
    input = Column(JSON, nullable=True)
    output = Column(JSON, nullable=True)
    state_before = Column(JSON, nullable=True)
    state_after = Column(JSON, nullable=True)
    timestamp = Column(String(64), default=utc_now_iso, nullable=False)
    latency = Column(Float, nullable=True)  # duration in seconds
    status = Column(String(32), default="success", nullable=False)  # "running", "success", "failed"

    run = relationship("Run", back_populates="steps")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "run_id": self.run_id,
            "event_id": self.event_id,
            "step_type": self.step_type,
            "tool_name": self.tool_name,
            "input": self.input,
            "output": self.output,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "timestamp": self.timestamp,
            "latency": self.latency,
            "status": self.status,
        }
