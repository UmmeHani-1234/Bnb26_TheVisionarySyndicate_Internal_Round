"""Feature extractor for observable agent execution traces.

Calculates the six structured signals for every candidate step by comparing
a target run against an aligned verified successful reference run.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .signals import (
    compute_downstream_impact,
    compute_error_status,
    compute_latency_anomaly,
    compute_output_divergence,
    compute_state_divergence,
    compute_tool_mismatch,
)

FEATURE_NAMES = [
    "tool_mismatch",
    "output_divergence",
    "state_divergence",
    "downstream_impact",
    "latency_anomaly",
    "error_status",
]


@dataclass
class StepFeatureRecord:
    run_id: str
    step_id: int
    step_type: str
    tool_name: Optional[str]
    features: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_id": self.step_id,
            "step_type": self.step_type,
            "tool_name": self.tool_name,
            "features": {k: round(v, 4) for k, v in self.features.items()},
        }

    def to_vector(self) -> List[float]:
        return [self.features.get(name, 0.0) for name in FEATURE_NAMES]


class FeatureExtractor:
    """Extracts structured, inspectable feature records for all candidate steps in a trace."""

    def extract_features(
        self,
        target_trace: Dict[str, Any],
        reference_trace: Optional[Dict[str, Any]] = None,
    ) -> List[StepFeatureRecord]:
        run_id = target_trace.get("run_id", "unknown_run")
        target_steps = target_trace.get("steps", [])

        ref_steps = reference_trace.get("steps", []) if reference_trace else []
        # Build lookup for reference steps by step_type or step_id
        ref_by_id = {s.get("step_id"): s for s in ref_steps if s.get("step_id") is not None}
        ref_by_type = {}
        for s in ref_steps:
            stype = s.get("step_type")
            if stype and stype not in ref_by_type:
                ref_by_type[stype] = s

        records: List[StepFeatureRecord] = []

        for idx, target_step in enumerate(target_steps):
            step_id = target_step.get("step_id", idx + 1)
            step_type = target_step.get("step_type", "unknown_step")
            tool_name = target_step.get("tool_name")

            # Find matching reference step: prefer same step_id, then same step_type, then same index
            ref_step = ref_by_id.get(step_id)
            if not ref_step:
                ref_step = ref_by_type.get(step_type)
            if not ref_step and idx < len(ref_steps):
                ref_step = ref_steps[idx]

            # Calculate the 6 signals
            f_tool_mismatch = compute_tool_mismatch(target_step, ref_step)
            f_output_div = compute_output_divergence(target_step, ref_step)
            f_state_div = compute_state_divergence(target_step, ref_step)
            f_downstream = compute_downstream_impact(idx, target_steps)
            f_latency = compute_latency_anomaly(target_step, ref_step)
            f_error = compute_error_status(target_step)

            feature_dict = {
                "tool_mismatch": f_tool_mismatch,
                "output_divergence": f_output_div,
                "state_divergence": f_state_div,
                "downstream_impact": f_downstream,
                "latency_anomaly": f_latency,
                "error_status": f_error,
            }

            records.append(
                StepFeatureRecord(
                    run_id=run_id,
                    step_id=step_id,
                    step_type=step_type,
                    tool_name=tool_name,
                    features=feature_dict,
                )
            )

        return records
