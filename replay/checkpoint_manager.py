"""CheckpointManager: Extracts and saves observable agent state at each execution step.

A checkpoint captures exactly what the agent knew at a given moment:
  - the user's original request
  - the budget constraint
  - which products were found by search_products
  - which product was selected / under consideration
  - which tools have been executed so far
  - the active failure_mode (if any)

No hidden chain-of-thought is stored.  Every field is derived from observable
tool inputs/outputs that already appear in the execution trace.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from storage.repository import TraceRepository
from storage.checkpoint_models import Checkpoint

logger = logging.getLogger(__name__)

# Step types that represent meaningful branch-points where replay is useful.
CHECKPOINT_STEP_TYPES = {
    "agent_started",
    "user_request_received",
    "tool_selected",
    "tool_completed",
    "tool_error",
    "final_response_generated",
    "agent_error",
}


def _extract_budget(text: str) -> Optional[int]:
    """Re-parse budget from user request text (mirrors agent/local_model.py)."""
    import re
    text_lower = text.lower()
    lakh = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|l)\b", text_lower)
    if lakh:
        return int(float(lakh.group(1)) * 100_000)
    k = re.search(r"(\d+)\s*k\b", text_lower)
    if k:
        return int(k.group(1)) * 1_000
    num = re.search(r"(?:₹|rs\.?|inr)?\s*(\d{1,3}(?:,\d{3})+|\d{4,6})", text)
    if num:
        val = int(num.group(1).replace(",", ""))
        if val > 1_000:
            return val
    return None


def _build_observable_state(
    user_request: str,
    steps_so_far: List[Dict[str, Any]],
    failure_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Reconstruct observable application state from executed steps up to this point."""
    budget = _extract_budget(user_request)
    retrieved_products: List[Dict[str, Any]] = []
    selected_product: Optional[Dict[str, Any]] = None
    tools_executed: List[str] = []

    for step in steps_so_far:
        tool_name = step.get("tool_name")
        output = step.get("output") or {}
        step_type = step.get("step_type", "")

        if step_type == "tool_completed" and tool_name:
            tools_executed.append(tool_name)

            if tool_name == "search_products" and isinstance(output, dict):
                retrieved_products = output.get("products", [])

            elif tool_name == "check_specifications" and isinstance(output, dict):
                # The product being checked is the "selected" candidate
                prod_name = output.get("product")
                if prod_name and retrieved_products:
                    match = next((p for p in retrieved_products if p.get("name") == prod_name), None)
                    if match:
                        selected_product = match

            elif tool_name == "calculate_budget" and isinstance(output, dict):
                prod_name = output.get("product")
                if prod_name and not selected_product and retrieved_products:
                    match = next((p for p in retrieved_products if p.get("name") == prod_name), None)
                    if match:
                        selected_product = match

    return {
        "user_request": user_request,
        "budget": budget,
        "retrieved_products": retrieved_products,
        "selected_product": selected_product,
        "tools_executed": tools_executed,
        "failure_mode": failure_mode or None,
    }


class CheckpointManager:
    """Creates and retrieves agent execution checkpoints.

    Checkpoints are automatically created after every observable tool_completed /
    tool_error step, and also at agent_started and user_request_received.
    Each checkpoint contains enough state to resume the LocalChatModel agent.
    """

    def __init__(self, repository: Optional[TraceRepository] = None) -> None:
        self.repo = repository or TraceRepository()

    # ------------------------------------------------------------------ public API

    def create_checkpoints_for_run(
        self,
        run_id: str,
        user_request: str,
        steps: List[Dict[str, Any]],
        failure_mode: Optional[str] = None,
    ) -> List[Checkpoint]:
        """Creates and persists one checkpoint per observable execution step.

        Call this after a run has been fully recorded (all steps present).
        Returns the list of Checkpoint objects that were saved.
        """
        checkpoints: List[Checkpoint] = []
        for i, step in enumerate(steps):
            step_type = step.get("step_type", "")
            if step_type not in CHECKPOINT_STEP_TYPES:
                continue

            step_id = step.get("step_id", i + 1)
            steps_before = steps[:i]  # state *before* this step executes
            state = _build_observable_state(user_request, steps_before, failure_mode)

            cp_id = f"cp-{run_id}-{step_id:04d}"
            cp = Checkpoint(
                checkpoint_id=cp_id,
                run_id=run_id,
                step_id=step_id,
                checkpoint_type="before_step",
                state=state,
            )
            saved = self.repo.save_checkpoint(cp)
            checkpoints.append(saved)

        return checkpoints

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Retrieves a checkpoint by ID."""
        return self.repo.get_checkpoint(checkpoint_id)

    def list_checkpoints_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Returns all checkpoints for a run, ordered by step_id."""
        return self.repo.list_checkpoints_for_run(run_id)

    def validate_checkpoint_belongs_to_run(
        self, checkpoint_id: str, run_id: str
    ) -> Checkpoint:
        """Returns the checkpoint only if it belongs to run_id; raises ValueError otherwise."""
        cp = self.repo.get_checkpoint(checkpoint_id)
        if cp is None:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found.")
        if cp.run_id != run_id:
            raise ValueError(
                f"Checkpoint '{checkpoint_id}' belongs to run '{cp.run_id}', not '{run_id}'."
            )
        return cp
