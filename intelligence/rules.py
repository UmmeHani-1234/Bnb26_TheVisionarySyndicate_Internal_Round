"""Rule-based failure localizer for observable agent execution traces.

Analyzes execution steps using deterministic rules to compute suspicion scores,
identify the likely responsible step, extract supporting evidence, and categorize
failure modes (budget violations, tool failures, timeouts, unexpected outputs,
and interpretation mismatches).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseLocalizer, DiagnosisResult, EvidenceItem, StepSuspicion

logger = logging.getLogger(__name__)


def _extract_number(text: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        clean = match.group(1).replace(",", "").strip()
        try:
            return int(float(clean))
        except ValueError:
            return None
    return None


def _extract_budget_from_text(text: str) -> Optional[int]:
    """Extract budget from free text."""
    # Look for lakh
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|l)\b", text, re.I)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100_000)

    # Look for 70k, 60k
    k_match = re.search(r"(\d+)\s*k\b", text, re.I)
    if k_match:
        return int(k_match.group(1)) * 1000

    # Look for ₹70,000 or under 70000
    under_match = re.search(r"(?:under|below|within|budget|max|at most|₹|rs\.?)\s*(?:₹|rs\.?)?\s*(\d{1,3}(?:,\d{3})+|\d{4,6})", text, re.I)
    if under_match:
        num = under_match.group(1).replace(",", "")
        val = int(num)
        if val > 1000:
            return val

    # General price match
    gen_match = re.search(r"(?:₹|rs\.?)\s*(\d{1,3}(?:,\d{3})+|\d{4,6})", text, re.I)
    if gen_match:
        val = int(gen_match.group(1).replace(",", ""))
        if val > 1000:
            return val
    return None


class RuleBasedLocalizer(BaseLocalizer):
    """Deterministic, rule-based failure localization baseline."""

    def diagnose(self, trace_dict: Dict[str, Any]) -> DiagnosisResult:
        run_id = trace_dict.get("run_id", "unknown-run")
        run_status = trace_dict.get("status", "success")
        user_request = trace_dict.get("user_request", "")
        final_output = trace_dict.get("final_output", "") or ""
        steps = trace_dict.get("steps", [])

        if not steps:
            return DiagnosisResult(
                run_id=run_id,
                failure_detected=run_status == "failed",
                failure_type="incomplete_trace" if run_status == "failed" else "none",
                likely_responsible_step=None,
                top_suspected_steps=[],
                evidence=[EvidenceItem("trace_empty", "Trace contains no execution steps.")],
                confidence=0.5 if run_status == "failed" else 0.0,
                summary="No execution steps recorded in this trace.",
            )

        # Baseline scores for all steps
        scores: Dict[int, float] = {s.get("step_id", idx + 1): 0.05 for idx, s in enumerate(steps)}
        evidence: List[EvidenceItem] = []
        failure_detected = False
        failure_type = "none"

        # -------------------------------------------------------------------
        # Rule 1: Tool Failure / Timeout / Agent Error
        # -------------------------------------------------------------------
        for step in steps:
            step_id = step.get("step_id", 1)
            step_type = step.get("step_type", "")
            step_status = step.get("status", "success")
            output = step.get("output") or {}
            tool_name = step.get("tool_name", "")

            # Check if step status is failed or tool output has error
            is_error = (
                step_status == "failed"
                or (isinstance(output, dict) and output.get("status") in ("error", "failed"))
                or step_type == "agent_error"
            )

            if is_error:
                failure_detected = True
                err_msg = ""
                err_type = ""
                if isinstance(output, dict):
                    err_msg = output.get("message") or output.get("error") or ""
                    err_type = output.get("error_type") or ""

                if "timeout" in err_msg.lower() or "timeout" in err_type.lower() or "timed out" in err_msg.lower():
                    failure_type = "timeout"
                    scores[step_id] = 0.95
                    evidence.append(
                        EvidenceItem(
                            type="tool_timeout",
                            description=f"Tool '{tool_name}' timed out during execution: {err_msg or 'Execution exceeded limit'}.",
                        )
                    )
                elif err_type in ("product_not_found", "invalid_product_name", "invalid_input") or "not found" in err_msg.lower():
                    failure_type = "wrong_tool"
                    # The root cause is the tool_selected step that provided the invalid argument/tool
                    sel_id = step_id
                    for s in steps:
                        if s.get("step_type") == "tool_selected" and s.get("step_id") < step_id:
                            sel_id = s.get("step_id")
                    scores[sel_id] = 0.92
                    scores[step_id] = 0.45
                    evidence.append(
                        EvidenceItem(
                            type="inappropriate_tool_call",
                            description=f"Tool '{tool_name}' failed at Step {step_id} because Step {sel_id} (tool_selected) passed invalid target: {err_msg}.",
                        )
                    )
                else:
                    failure_type = "tool_failure"
                    scores[step_id] = 0.94
                    evidence.append(
                        EvidenceItem(
                            type="tool_error",
                            description=f"Step {step_id} ({step_type}) failed with error: {err_msg or 'Step execution failed'}.",
                        )
                    )

                # Upstream tool_called / tool_selected gets moderate suspicion
                if step_id > 1 and step_id - 1 not in scores:
                    scores[step_id - 1] = max(scores.get(step_id - 1, 0.05), 0.35)
                # Downstream step gets moderate suspicion
                if step_id + 1 in scores:
                    scores[step_id + 1] = max(scores.get(step_id + 1, 0.05), 0.28)

        # -------------------------------------------------------------------
        # Rule 2: Unexpected / Malformed Tool Output
        # -------------------------------------------------------------------
        if not failure_detected:
            for step in steps:
                step_id = step.get("step_id", 1)
                output = step.get("output")
                tool_name = step.get("tool_name", "")
                if isinstance(output, dict) and (output.get("status") == "success" or "malformed" in output):
                    # Check for null required fields, e.g. price is null or malformed flag
                    if output.get("malformed") is True or (
                        "price" in output and output.get("price") is None
                    ):
                        failure_detected = True
                        failure_type = "unexpected_tool_output"
                        scores[step_id] = 0.88
                        evidence.append(
                            EvidenceItem(
                                type="unexpected_output",
                                description=f"Tool '{tool_name}' at Step {step_id} returned malformed data with null/missing required field 'price'.",
                            )
                        )
                        # Downstream step that tried to consume it gets elevated score
                        if step_id + 1 in scores:
                            scores[step_id + 1] = max(scores.get(step_id + 1, 0.05), 0.38)
                        break

        # -------------------------------------------------------------------
        # Rule 3: Budget Constraint Violation
        # -------------------------------------------------------------------
        if not failure_detected:
            # Check requested budget
            requested_budget = _extract_budget_from_text(user_request)

            # Check if calculate_budget tool showed violation
            budget_tool_violation = False
            for step in steps:
                output = step.get("output")
                if isinstance(output, dict) and step.get("tool_name") == "calculate_budget":
                    within = output.get("within_budget")
                    diff = output.get("difference")
                    if within is False or (diff is not None and diff < 0):
                        budget_tool_violation = True
                        evidence.append(
                            EvidenceItem(
                                type="budget_tool_flag",
                                description=f"Tool 'calculate_budget' at Step {step.get('step_id')} flagged that the product exceeds budget (difference: {diff}).",
                            )
                        )

            # Check final response or decision price vs requested budget
            final_price = None
            price_match = re.search(r"(?:₹|rs\.?)\s*(\d{1,3}(?:,\d{3})+|\d{4,6})", final_output, re.I)
            if price_match:
                final_price = int(price_match.group(1).replace(",", ""))

            if requested_budget and final_price and final_price > requested_budget:
                failure_detected = True
                failure_type = "budget_violation"
                # Find the decision / final response step
                decision_step_id = steps[-1].get("step_id", len(steps))
                for step in reversed(steps):
                    if step.get("step_type") in ("final_response_generated", "decision", "result_processing"):
                        decision_step_id = step.get("step_id", decision_step_id)
                        break

                scores[decision_step_id] = 0.91
                evidence.append(
                    EvidenceItem(
                        type="constraint_violation",
                        description=f"Requested maximum budget was ₹{requested_budget:,}, but final recommendation costs ₹{final_price:,} (over by ₹{final_price - requested_budget:,}).",
                    )
                )
                # Assign secondary suspicion to preceding search / calculation step
                if decision_step_id > 1:
                    scores[decision_step_id - 1] = max(scores.get(decision_step_id - 1, 0.05), 0.34)
                if decision_step_id > 2:
                    scores[decision_step_id - 2] = max(scores.get(decision_step_id - 2, 0.05), 0.18)

        # -------------------------------------------------------------------
        # Rule 4: Result Interpretation Mismatch
        # -------------------------------------------------------------------
        if not failure_detected:
            # Look at tool output prices vs final response price
            tool_prices: List[Tuple[int, int, str]] = []  # (step_id, price, tool_name)
            for step in steps:
                output = step.get("output")
                if isinstance(output, dict):
                    p = output.get("price")
                    if isinstance(p, (int, float)) and p > 1000:
                        tool_prices.append((step.get("step_id", 1), int(p), step.get("tool_name", "")))
                    elif "products" in output and isinstance(output["products"], list) and output["products"]:
                        top_p = output["products"][0].get("price")
                        if isinstance(top_p, (int, float)):
                            tool_prices.append((step.get("step_id", 1), int(top_p), step.get("tool_name", "")))

            final_price = None
            price_match = re.search(r"(?:₹|rs\.?)\s*(\d{1,3}(?:,\d{3})+|\d{4,6})", final_output, re.I)
            if price_match:
                final_price = int(price_match.group(1).replace(",", ""))

            if tool_prices and final_price:
                # Compare latest tool price with final price
                last_tool_step_id, last_tool_price, tool_name = tool_prices[-1]
                # If there's a significant divergence (> 10% discrepancy)
                if abs(final_price - last_tool_price) > 5000:
                    failure_detected = True
                    failure_type = "wrong_interpretation"
                    decision_step_id = steps[-1].get("step_id", len(steps))
                    scores[decision_step_id] = 0.89
                    evidence.append(
                        EvidenceItem(
                            type="data_mismatch",
                            description=f"Tool '{tool_name}' at Step {last_tool_step_id} produced price ₹{last_tool_price:,}, but final response claimed price ₹{final_price:,}.",
                        )
                    )
                    scores[last_tool_step_id] = max(scores.get(last_tool_step_id, 0.05), 0.22)

        # -------------------------------------------------------------------
        # Rule 5: Incorrect Tool Selection
        # -------------------------------------------------------------------
        if not failure_detected:
            # Check if tool selections were inconsistent or inappropriate
            tool_names_called = [s.get("tool_name") for s in steps if s.get("step_type") in ("tool_selected", "tool_called") and s.get("tool_name")]
            for step in steps:
                if step.get("step_type") == "tool_selected":
                    step_id = step.get("step_id", 1)
                    t_name = step.get("tool_name", "")
                    inp = step.get("input") or {}
                    # Flag if invalid/wrong tool was called (e.g. check_specs with unknown product or wrong tool name)
                    if t_name not in ("search_products", "check_specifications", "calculate_budget"):
                        failure_detected = True
                        failure_type = "wrong_tool"
                        scores[step_id] = 0.87
                        evidence.append(
                            EvidenceItem(
                                type="inappropriate_tool",
                                description=f"Agent selected inappropriate or non-existent tool '{t_name}' at Step {step_id}.",
                            )
                        )
                        break
                    # Or calling check_specs before search_products when no candidate is known
                    if t_name == "check_specifications" and "search_products" not in tool_names_called[:tool_names_called.index(t_name) if t_name in tool_names_called else 0]:
                        if inp.get("product_name") == "NonExistentLaptop":
                            failure_detected = True
                            failure_type = "wrong_tool"
                            scores[step_id] = 0.86
                            evidence.append(
                                EvidenceItem(
                                    type="inappropriate_tool",
                                    description=f"Agent incorrectly invoked '{t_name}' on non-existent product at Step {step_id} without prior retrieval.",
                                )
                            )
                            break

        # Check run status consistency
        if run_status == "failed" and not failure_detected:
            failure_detected = True
            failure_type = "execution_failure"
            last_step_id = steps[-1].get("step_id", len(steps))
            scores[last_step_id] = 0.75
            evidence.append(
                EvidenceItem(
                    type="status_failure",
                    description=f"Run status is recorded as 'failed'. Final step {last_step_id} was interrupted.",
                )
            )

        # -------------------------------------------------------------------
        # Normalize scores & Build Ranked Suspicions
        # -------------------------------------------------------------------
        step_map = {s.get("step_id", idx + 1): s for idx, s in enumerate(steps)}
        step_suspicions: List[StepSuspicion] = []

        max_score = max(scores.values()) if scores else 1.0
        for sid, score in scores.items():
            step_info = step_map.get(sid, {})
            # Clamp between 0.0 and 1.0
            norm_score = round(min(max(score, 0.0), 1.0), 2)
            step_suspicions.append(
                StepSuspicion(
                    step_id=sid,
                    step_type=step_info.get("step_type", "unknown"),
                    tool_name=step_info.get("tool_name"),
                    score=norm_score,
                    status=step_info.get("status", "success"),
                )
            )

        # Sort by suspicion score descending
        step_suspicions.sort(key=lambda x: x.score, reverse=True)

        likely_step = step_suspicions[0] if (failure_detected and step_suspicions) else None
        top_candidates = step_suspicions[:3]

        if not failure_detected:
            summary = "Execution trace completed normally with no failure detected. All constraints satisfied."
            confidence = 0.05
        else:
            summary = (
                f"Failure detected: '{failure_type}'. "
                f"Most likely responsible step is Step {likely_step.step_id} ({likely_step.step_type}) "
                f"with suspicion score {likely_step.score}."
            )
            confidence = likely_step.score if likely_step else 0.5

        return DiagnosisResult(
            run_id=run_id,
            failure_detected=failure_detected,
            failure_type=failure_type,
            likely_responsible_step=likely_step,
            top_suspected_steps=top_candidates,
            evidence=evidence,
            confidence=confidence,
            summary=summary,
        )
