"""Independent Verifier for agent execution traces.

Performs deterministic, independent checks on whether an agent run truly satisfied
its requirements, without relying on self-assessment by the agent itself.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    passed: bool
    reason: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "details": self.details or {},
        }


def _extract_budget(text: str) -> Optional[int]:
    """Extract budget from free text."""
    text_lower = text.lower()
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|l)\b", text_lower)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100_000)

    k_match = re.search(r"(\d+)\s*k\b", text_lower)
    if k_match:
        return int(k_match.group(1)) * 1000

    num_match = re.search(r"(?:₹|rs\.?|inr)?\s*(\d{1,3}(?:,\d{3})+|\d{4,6})", text)
    if num_match:
        val = int(num_match.group(1).replace(",", ""))
        if val > 1000:
            return val
    return None


def _extract_price_from_response(text: str) -> Optional[int]:
    """Find the recommended product price in final response."""
    match = re.search(r"(?:price|₹|rs\.?)\s*:?\s*(?:₹|rs\.?)?\s*(\d{1,3}(?:,\d{3})+|\d{4,6})", text, re.I)
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


class IndependentVerifier:
    """Verifies whether an agent execution succeeded according to observable criteria."""

    def verify(self, trace: Dict[str, Any]) -> VerificationResult:
        """Inspects trace dictionary and deterministically evaluates outcome."""
        if not trace:
            return VerificationResult(passed=False, reason="Empty or missing trace.")

        user_request = trace.get("user_request") or ""
        budget = _extract_budget(user_request)
        steps = trace.get("steps", [])

        if not steps:
            return VerificationResult(passed=False, reason="No steps recorded in execution trace.")

        # 1. Check for tool errors, timeouts, or agent errors
        for step in steps:
            stype = step.get("step_type", "")
            status = step.get("status", "")
            out = step.get("output") or {}

            if status in ("failed", "error") or stype in ("tool_error", "agent_error"):
                err_msg = out.get("message") if isinstance(out, dict) else str(out)
                return VerificationResult(
                    passed=False,
                    reason=f"Execution error encountered at step {step.get('step_id')}: {err_msg or 'Unspecified error'}",
                    details={"step_id": step.get("step_id"), "step_type": stype},
                )

            # Check for timeout in output payload
            if isinstance(out, dict) and out.get("error_type") == "timeout":
                return VerificationResult(
                    passed=False,
                    reason=f"Tool timeout detected at step {step.get('step_id')}.",
                    details={"step_id": step.get("step_id")},
                )

            # Check for malformed / corrupted tool output
            if isinstance(out, dict) and out.get("malformed") is True:
                return VerificationResult(
                    passed=False,
                    reason=f"Malformed or corrupted tool output detected at step {step.get('step_id')}.",
                    details={"step_id": step.get("step_id")},
                )

        # 2. Check tool selection flow: check_specifications must not be called before search_products
        tool_names_called = []
        for step in steps:
            stype = step.get("step_type", "")
            tname = step.get("tool_name")
            if stype in ("tool_called", "tool_completed", "tool_selected") and tname:
                tool_names_called.append(tname)

        if "check_specifications" in tool_names_called and "search_products" in tool_names_called:
            first_specs = tool_names_called.index("check_specifications")
            first_search = tool_names_called.index("search_products")
            if first_specs < first_search:
                return VerificationResult(
                    passed=False,
                    reason="Invalid workflow: check_specifications called before search_products.",
                    details={"tool_order": tool_names_called},
                )

        # 3. Check final response existence
        final_steps = [s for s in steps if s.get("step_type") in ("final_response", "final_response_generated")]
        final_text = ""
        output_price = None
        if final_steps:
            final_out = final_steps[-1].get("output") or {}
            if isinstance(final_out, dict):
                final_text = final_out.get("response") or ""
                if final_out.get("price") is not None:
                    try:
                        output_price = int(final_out["price"])
                    except (ValueError, TypeError):
                        pass
            elif isinstance(final_out, str):
                final_text = final_out
        elif trace.get("final_output"):
            final_text = trace["final_output"]

        if (not final_text or not final_text.strip()) and output_price is not None:
            final_text = f"Recommended product for ₹{output_price}"

        if not final_text or not final_text.strip():
            return VerificationResult(passed=False, reason="No final response generated by agent.")

        # 4. Check budget constraints against final answer and tool calculations
        stated_price = _extract_price_from_response(final_text) or output_price

        # Look for calculate_budget outputs
        calc_steps = [s for s in steps if s.get("tool_name") == "calculate_budget" and s.get("step_type") in ("tool_completed", "tool_execution")]
        tool_price = None
        for cs in calc_steps:
            cout = cs.get("output") or {}
            if isinstance(cout, dict):
                if cout.get("price") is not None:
                    tool_price = int(cout["price"])
                if cout.get("within_budget") is False:
                    return VerificationResult(
                        passed=False,
                        reason=f"Budget violation: Product price (₹{cout.get('price', 'N/A')}) exceeds user budget (₹{budget or cout.get('budget', 'N/A')}).",
                        details={"product": cout.get("product"), "price": cout.get("price"), "budget": cout.get("budget")},
                    )

        # 5. Check if stated price exceeds budget
        if budget and stated_price and stated_price > budget:
            return VerificationResult(
                passed=False,
                reason=f"Budget constraint violated: Recommended price ₹{stated_price:,} exceeds budget ₹{budget:,}.",
                details={"stated_price": stated_price, "budget": budget},
            )

        # 6. Check for interpretation mismatch (stated price differs substantially from tool price)
        if tool_price and stated_price and abs(tool_price - stated_price) > 500:
            return VerificationResult(
                passed=False,
                reason=f"Data interpretation mismatch: Tool output price ₹{tool_price:,} does not match final response stated price ₹{stated_price:,}.",
                details={"tool_price": tool_price, "stated_price": stated_price},
            )

        return VerificationResult(
            passed=True,
            reason="All deterministic success criteria met.",
            details={"stated_price": stated_price, "budget": budget},
        )
