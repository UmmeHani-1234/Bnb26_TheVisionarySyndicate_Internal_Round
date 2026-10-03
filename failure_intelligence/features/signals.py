"""Calculators for the six primary failure intelligence signals.

Signal 1: tool_mismatch
Signal 2: output_divergence
Signal 3: state_divergence
Signal 4: downstream_impact
Signal 5: latency_anomaly
Signal 6: error_status

All signals produce normalized numerical values in the range [0.0, 1.0].
"""

import json
from typing import Any, Dict, List, Optional


def compute_tool_mismatch(target_step: Dict[str, Any], ref_step: Optional[Dict[str, Any]]) -> float:
    """Computes whether target step called a different or inappropriate tool."""
    target_tool = target_step.get("tool_name")
    if not target_tool:
        return 0.0

    if not ref_step:
        return 0.0

    ref_tool = ref_step.get("tool_name")
    if not ref_tool:
        # Reference had no tool at this step, but target does
        return 1.0

    return 0.0 if target_tool == ref_tool else 1.0


def compute_output_divergence(target_step: Dict[str, Any], ref_step: Optional[Dict[str, Any]]) -> float:
    """Computes deterministic/structured divergence between step outputs."""
    t_out = target_step.get("output")
    if not t_out:
        return 0.0

    # If target has explicit error output
    if isinstance(t_out, dict):
        if t_out.get("status") == "error" or t_out.get("error_type") or t_out.get("malformed"):
            return 1.0

    if not ref_step or not ref_step.get("output"):
        return 0.3 if t_out else 0.0

    r_out = ref_step.get("output")

    # Both are dicts: compare keys, status, and payload values
    if isinstance(t_out, dict) and isinstance(r_out, dict):
        if t_out.get("status") != r_out.get("status"):
            return 0.85

        # Key symmetric difference
        t_keys = set(t_out.keys())
        r_keys = set(r_out.keys())
        key_union = t_keys | r_keys
        if not key_union:
            return 0.0
        key_diff_ratio = len(t_keys ^ r_keys) / len(key_union)

        # Check product count if search
        t_count = t_out.get("count")
        r_count = r_out.get("count")
        count_div = 0.0
        if t_count is not None and r_count is not None:
            max_c = max(t_count, r_count, 1)
            count_div = abs(t_count - r_count) / max_c

        # Check price differences if budget/specs
        t_price = t_out.get("price")
        r_price = r_out.get("price")
        price_div = 0.0
        if t_price and r_price:
            price_div = min(1.0, abs(float(t_price) - float(r_price)) / max(float(r_price), 1.0))

        divergence = 0.4 * key_diff_ratio + 0.3 * count_div + 0.3 * price_div
        return min(1.0, round(divergence, 3))

    # Compare string serialization
    t_str = str(t_out).strip()
    r_str = str(r_out).strip()
    if t_str == r_str:
        return 0.0

    # Jaccard word distance
    w_t = set(t_str.split())
    w_r = set(r_str.split())
    union = w_t | w_r
    if not union:
        return 0.0
    return min(1.0, round(1.0 - (len(w_t & w_r) / len(union)), 3))


def compute_state_divergence(
    target_step: Dict[str, Any],
    ref_step: Optional[Dict[str, Any]],
) -> float:
    """Measures divergence in inputs and parameters passed into the step."""
    t_in = target_step.get("input")
    if not t_in:
        return 0.0

    if not ref_step or not ref_step.get("input"):
        return 0.2 if t_in else 0.0

    r_in = ref_step.get("input")
    if isinstance(t_in, dict) and isinstance(r_in, dict):
        t_keys = set(t_in.keys())
        r_keys = set(r_in.keys())
        union = t_keys | r_keys
        if not union:
            return 0.0
        key_div = len(t_keys ^ r_keys) / len(union)

        # Value mismatches on shared keys
        shared = t_keys & r_keys
        mismatches = sum(1 for k in shared if str(t_in[k]) != str(r_in[k]))
        val_div = mismatches / max(len(shared), 1)

        return min(1.0, round(0.5 * key_div + 0.5 * val_div, 3))

    return 0.0 if str(t_in) == str(r_in) else 0.5


def compute_downstream_impact(step_index: int, all_steps: List[Dict[str, Any]]) -> float:
    """Measures whether subsequent steps experienced failures, errors, or anomalies."""
    if step_index >= len(all_steps) - 1:
        # Last step has no subsequent steps
        # If it's a final response and failed, give it 1.0 impact
        last_step = all_steps[step_index]
        return 1.0 if last_step.get("status") in ("failed", "error") else 0.0

    subsequent = all_steps[step_index + 1 :]
    abnormal_count = 0
    for s in subsequent:
        st = s.get("status", "")
        out = s.get("output") or {}
        stype = s.get("step_type", "")
        if st in ("failed", "error") or stype in ("tool_error", "agent_error"):
            abnormal_count += 1
        elif isinstance(out, dict) and (out.get("status") == "error" or out.get("error_type")):
            abnormal_count += 1

    return min(1.0, round(abnormal_count / len(subsequent), 3))


def compute_latency_anomaly(target_step: Dict[str, Any], ref_step: Optional[Dict[str, Any]]) -> float:
    """Calculates normalized latency anomaly relative to reference execution."""
    t_lat = target_step.get("latency")
    if t_lat is None:
        return 0.0

    if not ref_step or ref_step.get("latency") is None:
        # Default baseline threshold: anything over 5.0 seconds is suspicious
        return min(1.0, max(0.0, float(t_lat) / 10.0))

    r_lat = float(ref_step["latency"])
    t_lat = float(t_lat)

    # Relative difference
    diff = abs(t_lat - r_lat)
    denom = max(r_lat, 0.2)
    relative_change = diff / denom

    # Normalize to [0, 1] using soft saturation
    # e.g. 5x latency difference maps to near 1.0
    return min(1.0, round(relative_change / 5.0, 3))


def compute_error_status(target_step: Dict[str, Any]) -> float:
    """Observable execution status representation."""
    status = (target_step.get("status") or "").lower()
    stype = (target_step.get("step_type") or "").lower()
    out = target_step.get("output") or {}

    if status in ("failed", "error", "timeout") or stype in ("tool_error", "agent_error"):
        return 1.0

    if isinstance(out, dict):
        if out.get("status") == "error" or out.get("error_type") == "timeout":
            return 1.0
        if out.get("malformed") is True or out.get("price") is None and "price" in out:
            return 0.75

    return 0.0
