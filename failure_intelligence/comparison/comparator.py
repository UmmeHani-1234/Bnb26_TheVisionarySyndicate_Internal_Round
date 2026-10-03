"""Trace Comparator and Recovery Verifier for Black Box.

Performs 3-way alignment and comparative analysis across:
1. Original failed execution
2. Alternative/replay execution
3. Verified successful reference execution

Evaluates step-level divergences, downstream propagation, and runs the
IndependentVerifier to confirm recovery status.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from failure_intelligence.evaluation.verifier import IndependentVerifier, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class AlignedStepComparison:
    step_id: int
    step_type: str
    original_tool: Optional[str]
    original_status: str
    original_summary: str
    alternative_tool: Optional[str]
    alternative_status: str
    alternative_summary: str
    reference_tool: Optional[str]
    reference_status: str
    reference_summary: str
    is_changed: bool
    change_description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceComparisonResult:
    original_run_id: str
    alternative_run_id: str
    reference_run_id: Optional[str]
    steps: List[AlignedStepComparison]
    changed_steps: List[int]
    downstream_effects: List[str]
    original_verifier: Dict[str, Any]
    alternative_verifier: Dict[str, Any]
    recovered: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_run_id": self.original_run_id,
            "alternative_run_id": self.alternative_run_id,
            "reference_run_id": self.reference_run_id,
            "steps": [s.to_dict() for s in self.steps],
            "changed_steps": self.changed_steps,
            "downstream_effects": self.downstream_effects,
            "original_verifier": self.original_verifier,
            "alternative_verifier": self.alternative_verifier,
            "recovered": self.recovered,
        }


def _step_summary(step: Optional[Dict[str, Any]]) -> str:
    """Produces a clean 1-line observable summary of a step."""
    if not step:
        return "Not present"
    tname = step.get("tool_name")
    stype = step.get("step_type", "")
    out = step.get("output") or {}
    status = step.get("status", "unknown")

    if isinstance(out, dict):
        if out.get("error_type"):
            return f"Error: {out.get('error_type')}"
        if out.get("product"):
            return f"{out.get('product')} (₹{out.get('price', 'N/A')})"
        if out.get("count") is not None:
            return f"{out.get('count')} products returned"
        if out.get("response"):
            resp = out["response"]
            return (resp[:60] + "...") if len(resp) > 60 else resp

    if tname:
        return f"{tname} ({status})"
    return f"{stype} ({status})"


class TraceComparator:
    """Compares original, alternative, and reference runs side-by-side."""

    def __init__(self, verifier: Optional[IndependentVerifier] = None) -> None:
        self.verifier = verifier or IndependentVerifier()

    def compare_traces(
        self,
        original_trace: Dict[str, Any],
        alternative_trace: Dict[str, Any],
        reference_trace: Optional[Dict[str, Any]] = None,
    ) -> TraceComparisonResult:
        orig_id = original_trace.get("run_id", "original")
        alt_id = alternative_trace.get("run_id", "alternative")
        ref_id = reference_trace.get("run_id") if reference_trace else None

        orig_steps = original_trace.get("steps", [])
        alt_steps = alternative_trace.get("steps", [])
        ref_steps = reference_trace.get("steps", []) if reference_trace else []

        orig_by_id = {s.get("step_id"): s for s in orig_steps if s.get("step_id") is not None}
        alt_by_id = {s.get("step_id"): s for s in alt_steps if s.get("step_id") is not None}
        ref_by_id = {s.get("step_id"): s for s in ref_steps if s.get("step_id") is not None}

        # Collect all unique step_ids preserving logical order
        all_step_ids = sorted(list(set(orig_by_id.keys()) | set(alt_by_id.keys()) | set(ref_by_id.keys())))

        aligned_steps: List[AlignedStepComparison] = []
        changed_step_ids: List[int] = []
        downstream_effects: List[str] = []

        first_change_seen = False

        for sid in all_step_ids:
            o = orig_by_id.get(sid)
            a = alt_by_id.get(sid)
            r = ref_by_id.get(sid)

            stype = (o or a or r or {}).get("step_type", "unknown")

            o_tool = o.get("tool_name") if o else None
            a_tool = a.get("tool_name") if a else None
            r_tool = r.get("tool_name") if r else None

            o_status = o.get("status", "none") if o else "none"
            a_status = a.get("status", "none") if a else "none"
            r_status = r.get("status", "none") if r else "none"

            o_sum = _step_summary(o)
            a_sum = _step_summary(a)
            r_sum = _step_summary(r)

            # Check if alternative diverged or was modified from original
            is_changed = False
            change_notes = []

            if o_tool != a_tool:
                is_changed = True
                change_notes.append(f"Tool: {o_tool or 'none'} -> {a_tool or 'none'}")

            if o_status != a_status:
                is_changed = True
                change_notes.append(f"Status: {o_status} -> {a_status}")

            if o_sum != a_sum and not is_changed:
                is_changed = True
                change_notes.append("Output updated")

            change_desc = "; ".join(change_notes) if change_notes else "Identical"

            if is_changed:
                changed_step_ids.append(sid)
                if not first_change_seen:
                    downstream_effects.append(f"Intervention initiated at Step {sid} ({change_desc}).")
                    first_change_seen = True
                else:
                    downstream_effects.append(f"Step {sid} adapted downstream: {change_desc}.")
            elif first_change_seen:
                if a_status == "success" and o_status in ("failed", "error"):
                    downstream_effects.append(f"Step {sid} recovered to success in alternative execution.")

            aligned_steps.append(
                AlignedStepComparison(
                    step_id=sid,
                    step_type=stype,
                    original_tool=o_tool,
                    original_status=o_status,
                    original_summary=o_sum,
                    alternative_tool=a_tool,
                    alternative_status=a_status,
                    alternative_summary=a_sum,
                    reference_tool=r_tool,
                    reference_status=r_status,
                    reference_summary=r_sum,
                    is_changed=is_changed,
                    change_description=change_desc,
                )
            )

        # 4. Independent Verifier checks on both runs
        v_orig = self.verifier.verify(original_trace)
        v_alt = self.verifier.verify(alternative_trace)

        # Recovery is achieved if the alternative passed while the original failed
        recovered = (not v_orig.passed) and v_alt.passed

        if recovered:
            downstream_effects.append("Independent Verifier confirms complete outcome recovery in alternative run.")
        elif not v_alt.passed:
            downstream_effects.append(f"Alternative execution failed verification: {v_alt.reason}")

        return TraceComparisonResult(
            original_run_id=orig_id,
            alternative_run_id=alt_id,
            reference_run_id=ref_id,
            steps=aligned_steps,
            changed_steps=changed_step_ids,
            downstream_effects=downstream_effects,
            original_verifier=v_orig.to_dict(),
            alternative_verifier=v_alt.to_dict(),
            recovered=recovered,
        )
