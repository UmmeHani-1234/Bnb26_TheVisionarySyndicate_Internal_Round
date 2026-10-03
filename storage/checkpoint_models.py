"""Checkpoint SQLAlchemy model for Stage 5: Debugging + Replay.

A checkpoint stores the complete observable application state of the agent
at a specific execution step, enabling replay from that exact point.
Only observable facts are stored — no hidden chain-of-thought.
"""

from typing import Any, Dict
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Checkpoint(Base):
    """Stores a restorable snapshot of observable agent state at a given step."""

    __tablename__ = "checkpoints"

    checkpoint_id = Column(String(64), primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("runs.run_id"), nullable=False, index=True)
    step_id = Column(Integer, nullable=False)
    created_at = Column(String(64), default=utc_now_iso, nullable=False)

    # "before_step" or "after_step" — helps the developer understand when to branch
    checkpoint_type = Column(String(32), default="before_step", nullable=False)

    # Full observable state JSON — what the agent knew/had at this moment.
    # Contains: user_request, budget, retrieved_products, selected_product,
    # tools_executed, failure_mode (if any).
    state = Column(JSON, nullable=False)

    run = relationship("Run", foreign_keys=[run_id])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "created_at": self.created_at,
            "checkpoint_type": self.checkpoint_type,
            "state": self.state,
        }
