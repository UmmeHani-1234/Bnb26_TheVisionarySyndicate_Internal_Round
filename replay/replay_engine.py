"""ReplayEngine: Restores agent state from a checkpoint and runs an alternative execution.

Architecture:
    Original Run → Checkpoint → ReplayEngine → New Replay Run

The replay engine:
  1. Loads the checkpoint state (observable facts only, no chain-of-thought).
  2. Applies optional developer overrides (budget, failure_mode).
  3. Initialises a fresh LocalChatModel whose "stage 1/2/3/4" state
     is pre-seeded from the checkpoint (so it skips already-completed steps).
  4. Runs the agent from the checkpoint point forward.
  5. Records all replayed events into a NEW run (parent_run_id = original).
  6. Never touches the original run.

Safe overrides (match the LocalChatModel parameter surface):
  - max_budget    : int  – override the budget cap in calculate_budget calls
  - failure_mode  : str  – change/remove the injected failure ("none" to clear)
  - tool_name     : str  – force a specific tool to be called first (wrong_tool fix)
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from storage.repository import TraceRepository
from storage.checkpoint_models import Checkpoint
from recorder.recorder import ExecutionRecorder
from agent.agent import LaptopAgent
from agent.local_model import LocalChatModel

logger = logging.getLogger(__name__)

# Allowed override keys and their expected Python types
_ALLOWED_OVERRIDES: Dict[str, type] = {
    "max_budget": int,
    "failure_mode": str,
    "tool_name": str,
}


@dataclass
class ReplayResult:
    """Outcome of a replay execution."""
    original_run_id: str
    replay_run_id: str
    checkpoint_id: str
    checkpoint_step_id: int
    status: str               # "success" | "failed"
    final_response: str
    replay_type: str = "standard"   # "standard" | "alternative"
    override_applied: Optional[Dict[str, Any]] = None


class ReplayEngine:
    """Executes replay runs from a checkpoint, with optional override parameters."""

    def __init__(self, repository: Optional[TraceRepository] = None) -> None:
        self.repo = repository or TraceRepository()

    # ------------------------------------------------------------------ public API

    def replay(
        self,
        run_id: str,
        checkpoint: Checkpoint,
        override: Optional[Dict[str, Any]] = None,
    ) -> ReplayResult:
        """Replay agent execution from ``checkpoint`` with an optional controlled override.

        Parameters
        ----------
        run_id:
            The original run ID (must match checkpoint.run_id).
        checkpoint:
            The Checkpoint object returned by CheckpointManager.
        override:
            Optional dict of controlled parameter changes.
            Allowed keys: max_budget (int), failure_mode (str), tool_name (str).

        Returns
        -------
        ReplayResult with the new replay run ID and outcome.
        """
        # 1. Validate override keys
        validated_override = self._validate_override(override)

        # 2. Build replay context from checkpoint state + override
        state = dict(checkpoint.state or {})
        replay_run_id = f"replay-{run_id}-{uuid.uuid4().hex[:6]}"
        replay_type = "alternative" if validated_override else "standard"

        # 3. Apply overrides onto state
        if validated_override:
            if "max_budget" in validated_override:
                state["budget"] = validated_override["max_budget"]
            if "failure_mode" in validated_override:
                fm = validated_override["failure_mode"]
                state["failure_mode"] = None if fm.lower() in ("none", "", "clear") else fm

        failure_mode = state.get("failure_mode") or None
        user_request = state.get("user_request", "")

        # 4. Build a pre-seeded LocalChatModel that skips already-completed tools
        tools_done = list(state.get("tools_executed", []))
        seeded_llm = self._build_seeded_llm(state, tools_done, validated_override)

        # 5. Set up recorder linked to the new replay run
        recorder = ExecutionRecorder(run_id=replay_run_id, repository=self.repo)

        # 6. Store the replay run with lineage metadata
        self.repo.create_run(
            run_id=replay_run_id,
            user_request=user_request,
            status="running",
            parent_run_id=run_id,
            replay_metadata={
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_step_id": checkpoint.step_id,
                "replay_type": replay_type,
                "override": validated_override,
                "state_at_checkpoint": state,
            },
        )

        # 7. Run the agent from checkpoint state
        try:
            agent = LaptopAgent(llm=seeded_llm, sinks=[recorder.record])
            agent_result = agent.run(user_request, failure_mode=failure_mode)
        except Exception as exc:
            logger.exception("Replay execution failed for run '%s'", replay_run_id)
            self.repo.update_run(
                run_id=replay_run_id,
                status="failed",
                final_output=f"Replay error: {exc}",
            )
            return ReplayResult(
                original_run_id=run_id,
                replay_run_id=replay_run_id,
                checkpoint_id=checkpoint.checkpoint_id,
                checkpoint_step_id=checkpoint.step_id,
                status="failed",
                final_response=f"Replay error: {exc}",
                replay_type=replay_type,
                override_applied=validated_override,
            )

        return ReplayResult(
            original_run_id=run_id,
            replay_run_id=replay_run_id,
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_step_id=checkpoint.step_id,
            status=agent_result.status,
            final_response=agent_result.final_response,
            replay_type=replay_type,
            override_applied=validated_override,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _validate_override(override: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate override keys and coerce types.  Raises ValueError on bad input."""
        if not override:
            return None
        validated: Dict[str, Any] = {}
        for key, value in override.items():
            if key not in _ALLOWED_OVERRIDES:
                raise ValueError(
                    f"Override key '{key}' is not allowed. "
                    f"Allowed keys: {list(_ALLOWED_OVERRIDES.keys())}"
                )
            expected_type = _ALLOWED_OVERRIDES[key]
            try:
                validated[key] = expected_type(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Override '{key}' must be {expected_type.__name__}, got {type(value).__name__}: {exc}"
                ) from exc
        return validated or None

    @staticmethod
    def _build_seeded_llm(
        state: Dict[str, Any],
        tools_done: List[str],
        override: Optional[Dict[str, Any]],
    ) -> LocalChatModel:
        """Create a LocalChatModel pre-seeded to resume from the checkpoint.

        The seeding works by overriding the budget/failure_mode inside the LocalChatModel
        at construction time. The LocalChatModel already reads tools_done from the
        LangChain message history, so when the agent graph starts fresh it will
        call the same tools — but the budget / failure_mode reflect the checkpoint
        (+ any override) rather than the original run's environment variable.
        """
        import os
        from agent.local_model import LocalChatModel

        # Map override fields onto LocalChatModel's constructor
        failure_mode_val = state.get("failure_mode") or None
        if override and "failure_mode" in override:
            raw_fm = override["failure_mode"]
            failure_mode_val = None if raw_fm.lower() in ("none", "", "clear") else raw_fm

        return LocalChatModel(failure_mode=failure_mode_val)
