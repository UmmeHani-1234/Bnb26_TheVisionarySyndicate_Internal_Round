"""Repository layer for persisting and retrieving agent execution traces."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from .models import Run, ExecutionStep, ConversationMessage, utc_now_iso
from .checkpoint_models import Checkpoint
from .database import SessionLocal, init_db

logger = logging.getLogger(__name__)


class TraceRepository:
    """Manages CRUD operations for Runs, ExecutionSteps, and Checkpoints."""

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
        failure_metadata: Optional[Dict[str, Any]] = None,
        parent_run_id: Optional[str] = None,
        replay_metadata: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """Creates a new Run record in storage."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            existing = session.get(Run, run_id)
            if existing:
                existing.user_request = user_request
                existing.status = status
                if failure_metadata is not None:
                    existing.failure_metadata = failure_metadata
                if parent_run_id is not None:
                    existing.parent_run_id = parent_run_id
                if replay_metadata is not None:
                    existing.replay_metadata = replay_metadata
                session.commit()
                session.refresh(existing)
                return existing

            run = Run(
                run_id=run_id,
                user_request=user_request,
                started_at=started_at or utc_now_iso(),
                status=status,
                failure_metadata=failure_metadata,
                parent_run_id=parent_run_id,
                replay_metadata=replay_metadata,
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

            step_id = event_dict.get("step_id")
            if step_id is None:
                step_id = event_dict.get("sequence")
            if step_id is None:
                current_count = len(run.steps)
                step_id = current_count + 1

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
        failure_metadata: Optional[Dict[str, Any]] = None,
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
            if failure_metadata is not None:
                run.failure_metadata = failure_metadata
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

    # ------------------------------------------------------------------ Checkpoint CRUD

    def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        """Persists a Checkpoint record to storage."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            existing = session.get(Checkpoint, checkpoint.checkpoint_id)
            if existing:
                existing.state = checkpoint.state
                existing.checkpoint_type = checkpoint.checkpoint_type
                session.commit()
                session.refresh(existing)
                return existing
            session.add(checkpoint)
            session.commit()
            session.refresh(checkpoint)
            return checkpoint
        finally:
            if close_on_finish:
                session.close()

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Retrieves a checkpoint by its ID."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            return session.get(Checkpoint, checkpoint_id)
        finally:
            if close_on_finish:
                session.close()

    def list_checkpoints_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Returns all checkpoints belonging to a run, ordered by step_id."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            cps = (
                session.query(Checkpoint)
                .filter(Checkpoint.run_id == run_id)
                .order_by(Checkpoint.step_id)
                .all()
            )
            return [cp.to_dict() for cp in cps]
        finally:
            if close_on_finish:
                session.close()

    def list_replays_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Returns all replay runs created from the given original run_id."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            replays = (
                session.query(Run)
                .filter(Run.parent_run_id == run_id)
                .order_by(Run.started_at.desc())
                .all()
            )
            return [r.to_dict(include_steps=False) for r in replays]
        finally:
            if close_on_finish:
                session.close()

    # ------------------------------------------------------------------ Conversation Message CRUD

    def save_conversation_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMessage:
        """Persists a conversation turn (system, user, or assistant) to storage."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            norm_role = role.lower()
            if norm_role in ("human", "user"):
                norm_role = "user"
            elif norm_role in ("ai", "assistant"):
                norm_role = "assistant"
            elif norm_role in ("system",):
                norm_role = "system"

            msg = ConversationMessage(
                conversation_id=conversation_id,
                role=norm_role,
                content=content,
                timestamp=utc_now_iso(),
                message_metadata=metadata,
            )
            session.add(msg)
            session.commit()
            session.refresh(msg)
            return msg
        finally:
            if close_on_finish:
                session.close()

    def get_conversation_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves ordered conversation history for a given conversation_id."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            query = (
                session.query(ConversationMessage)
                .filter(ConversationMessage.conversation_id == conversation_id)
                .order_by(ConversationMessage.id.asc())
            )
            if limit is not None and limit > 0:
                query = query.limit(limit)
            messages = query.all()
            return [m.to_dict() for m in messages]
        finally:
            if close_on_finish:
                session.close()

    def clear_conversation_messages(self, conversation_id: str) -> None:
        """Clears all stored messages for a specific conversation_id."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            session.query(ConversationMessage).filter(
                ConversationMessage.conversation_id == conversation_id
            ).delete()
            session.commit()
        finally:
            if close_on_finish:
                session.close()

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[str]:
        """Returns distinct conversation IDs ordered by recent activity."""
        session = self._get_session()
        close_on_finish = self._db is None
        try:
            from sqlalchemy import func
            subq = (
                session.query(
                    ConversationMessage.conversation_id,
                    func.max(ConversationMessage.id).label("max_id"),
                )
                .group_by(ConversationMessage.conversation_id)
                .order_by(func.max(ConversationMessage.id).desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [row[0] for row in subq]
        finally:
            if close_on_finish:
                session.close()
