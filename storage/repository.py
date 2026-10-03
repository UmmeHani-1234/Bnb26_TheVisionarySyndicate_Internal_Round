"""Repository layer for persisting and retrieving agent execution traces."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import select

from .models import Run, ExecutionStep, utc_now_iso
from .database import SessionLocal, init_db

logger = logging.getLogger(__name__)


class TraceRepository:
    """Manages CRUD operations for Runs and ExecutionSteps."""

    def __init__(self, db: Optional[Session] = None) -> None:
        init_db()
        self._db = db

    def _get_session(self) -> Session:
        return self._db if self._db is not None else SessionLocal()

    def create_run(
        self,
        run_id: str,
        user_request: str,
        started_at: Optional[str] = None,
        status: str = "running",
    ) -> Run:
        """Creates a new Run record in storage."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            # Check if run already exists
            existing = session.get(Run, run_id)
            if existing:
                existing.user_request = user_request
                existing.status = status
                session.commit()
                session.refresh(existing)
                return existing

            run = Run(
                run_id=run_id,
                user_request=user_request,
                started_at=started_at or utc_now_iso(),
                status=status,
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return run
        finally:
            if close_on_finish:
                session.close()

    def add_step(
        self,
        run_id: str,
        event_dict: Dict[str, Any],
    ) -> ExecutionStep:
        """Appends an execution step to a run, maintaining strict execution order."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            run = session.get(Run, run_id)
            if not run:
                raise ValueError(f"Run '{run_id}' not found.")

            # Calculate step_id to strictly preserve ordering
            step_id = event_dict.get("step_id")
            if step_id is None:
                step_id = event_dict.get("sequence")
            if step_id is None:
                current_count = len(run.steps)
                step_id = current_count + 1

            # Latency in seconds (duration_ms / 1000)
            latency = None
            metadata = event_dict.get("metadata") or {}
            if "duration_ms" in metadata and metadata["duration_ms"] is not None:
                try:
                    latency = round(float(metadata["duration_ms"]) / 1000.0, 3)
                except (ValueError, TypeError):
                    pass
            elif "latency" in event_dict and event_dict["latency"] is not None:
                latency = float(event_dict["latency"])

            status_str = event_dict.get("status", "success")
            if status_str in ("error", "failed"):
                status_str = "failed"
            elif status_str in ("started", "running"):
                status_str = "running"
            else:
                status_str = "success"

            step = ExecutionStep(
                step_id=step_id,
                run_id=run_id,
                event_id=event_dict.get("event_id") or f"evt-{step_id}",
                step_type=event_dict.get("step_type") or event_dict.get("event_type", "unknown"),
                tool_name=event_dict.get("tool_name"),
                input=event_dict.get("input"),
                output=event_dict.get("output"),
                state_before=event_dict.get("state_before"),
                state_after=event_dict.get("state_after"),
                timestamp=event_dict.get("timestamp") or utc_now_iso(),
                latency=latency,
                status=status_str,
            )
            session.add(step)
            session.commit()
            session.refresh(step)
            return step
        finally:
            if close_on_finish:
                session.close()

    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        final_output: Optional[str] = None,
        ended_at: Optional[str] = None,
    ) -> Optional[Run]:
        """Updates run status, final response, and completion time."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            run = session.get(Run, run_id)
            if not run:
                return None
            if status:
                run.status = status
            if final_output is not None:
                run.final_output = final_output
            if ended_at:
                run.ended_at = ended_at
            elif status in ("success", "failed") and not run.ended_at:
                run.ended_at = utc_now_iso()
            session.commit()
            session.refresh(run)
            return run
        finally:
            if close_on_finish:
                session.close()

    def get_run(self, run_id: str) -> Optional[Run]:
        """Retrieves a single run record by ID."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            return session.get(Run, run_id)
        finally:
            if close_on_finish:
                session.close()

    def get_run_trace(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete execution trace including all ordered steps."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            run = session.get(Run, run_id)
            if not run:
                return None
            return run.to_dict(include_steps=True)
        finally:
            if close_on_finish:
                session.close()

    def list_runs(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Returns list of runs ordered by started_at descending."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            runs = session.query(Run).order_by(Run.started_at.desc()).offset(offset).limit(limit).all()
            return [r.to_dict(include_steps=False) for r in runs]
        finally:
            if close_on_finish:
                session.close()
