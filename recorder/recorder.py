"""ExecutionRecorder sink connecting agent event pipeline directly to trace storage."""

import logging
import uuid
from typing import Any, Dict, Optional

from agent.events import Event, EventType, EventStatus
from storage.repository import TraceRepository

logger = logging.getLogger(__name__)


class ExecutionRecorder:
    """Listens to EventLog stream, automatically persisting Runs and ExecutionSteps."""

    def __init__(self, run_id: Optional[str] = None, repository: Optional[TraceRepository] = None) -> None:
        self.repository = repository or TraceRepository()
        self.run_id = run_id
        self._current_run_id = run_id
        self._user_request: str = ""

    def record(self, event: Event) -> None:
        """Sink method called synchronously on every agent event emission."""
        try:
            # 1. Initialize Run on AGENT_STARTED if not already set
            if event.event_type == EventType.AGENT_STARTED:
                if not self._current_run_id:
                    self._current_run_id = self.run_id or f"run-{uuid.uuid4().hex[:8]}"
                self.repository.create_run(
                    run_id=self._current_run_id,
                    user_request=self._user_request or "Initializing...",
                    started_at=event.timestamp,
                    status="running",
                )

            # 2. Capture and update user request
            elif event.event_type == EventType.USER_REQUEST_RECEIVED:
                if not self._current_run_id:
                    self._current_run_id = self.run_id or f"run-{uuid.uuid4().hex[:8]}"
                req = ""
                if isinstance(event.input, dict):
                    req = event.input.get("request", "")
                elif isinstance(event.input, str):
                    req = event.input
                self._user_request = req
                self.repository.create_run(
                    run_id=self._current_run_id,
                    user_request=self._user_request or "Empty request",
                    started_at=event.timestamp,
                    status="running",
                )

            # Ensure run exists before adding step
            if not self._current_run_id:
                self._current_run_id = self.run_id or f"run-{uuid.uuid4().hex[:8]}"
                self.repository.create_run(
                    run_id=self._current_run_id,
                    user_request=self._user_request or "Agent Execution",
                    status="running",
                )

            # 3. Always store the observable execution step
            self.repository.add_step(
                run_id=self._current_run_id,
                event_dict=event.to_dict(),
            )

            # 4. Final step handling & run completion
            if event.event_type == EventType.FINAL_RESPONSE_GENERATED:
                resp_text = ""
                if isinstance(event.output, dict):
                    resp_text = event.output.get("response", "")
                elif isinstance(event.output, str):
                    resp_text = event.output
                self.repository.update_run(
                    run_id=self._current_run_id,
                    status="success",
                    final_output=resp_text,
                    ended_at=event.timestamp,
                )

            elif event.event_type == EventType.AGENT_ERROR:
                err_msg = ""
                if isinstance(event.output, dict):
                    err_msg = event.output.get("message", "")
                elif isinstance(event.output, str):
                    err_msg = event.output
                if not err_msg:
                    err_msg = event.summary
                self.repository.update_run(
                    run_id=self._current_run_id,
                    status="failed",
                    final_output=err_msg,
                    ended_at=event.timestamp,
                )

        except Exception as exc:
            # Observability and recording must never crash the main agent
            logger.exception("ExecutionRecorder failed to store event: %s", exc)
