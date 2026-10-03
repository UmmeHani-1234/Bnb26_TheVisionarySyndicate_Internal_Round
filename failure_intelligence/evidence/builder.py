"""Structured evidence builder for failure localization.

Extracts concrete, observable trace facts corresponding to suspicious candidate steps.
Strictly relies on verifiable execution data rather than LLM assertions.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from failure_intelligence.localization.rule_based import RankedCandidateStep


@dataclass
class StepEvidencePacket:
    run_id: str
    rank: int
    step_id: int
    step_type: str
    tool_name: Optional[str]
    suspicion_score: float
    signals: Dict[str, float]
    trace_facts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rank": self.rank,
            "step_id": self.step_id,
            "step_type": self.step_type,
            "tool_name": self.tool_name,
            "suspicion_score": round(self.suspicion_score, 4),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "trace_facts": self.trace_facts,
        }


class EvidenceBuilder:
    """Builds structured evidence packets mapping observable trace facts to candidate steps."""

    def build_evidence(
        self,
        run_id: str,
        ranked_steps: List[RankedCandidateStep],
        target_trace: Dict[str, Any],
        reference_trace: Optional[Dict[str, Any]] = None,
        max_candidates: int = 3,
    ) -> List[StepEvidencePacket]:
        steps = target_trace.get("steps", [])
        steps_by_id = {s.get("step_id"): s for s in steps}
        user_request = target_trace.get("user_request", "")

        packets: List[StepEvidencePacket] = []

        for idx, candidate in enumerate(ranked_steps[:max_candidates], start=1):
            step = steps_by_id.get(candidate.step_id, {})
            facts: List[str] = []

            # 1. Fact on tool and execution status
            tname = candidate.tool_name or step.get("tool_name")
            stype = candidate.step_type or step.get("step_type")
            status = step.get("status", "unknown")

            if tname:
                facts.append(f"Step executed tool '{tname}' with status '{status}'.")
            else:
                facts.append(f"Step performed '{stype}' with status '{status}'.")

            # 2. Fact on errors if present
            out = step.get("output") or {}
            if isinstance(out, dict):
                if out.get("error_type"):
                    facts.append(f"Output recorded error '{out.get('error_type')}': {out.get('message', 'No message')}.")
                if out.get("malformed"):
                    facts.append("Output payload contained malformed or corrupted values (null price).")
                if out.get("within_budget") is False:
                    facts.append(f"Budget check failed: product price exceeds budget by ₹{abs(out.get('difference', 0)):,}.")

            # 3. Fact on inputs
            inp = step.get("input") or {}
            if isinstance(inp, dict) and inp:
                facts.append(f"Parameters passed: {', '.join(f'{k}={v}' for k, v in inp.items())}.")

            # 4. Fact on latency if anomalous
            lat = step.get("latency")
            if lat is not None and candidate.signals.get("latency_anomaly", 0.0) > 0.4:
                facts.append(f"Latency was unusually high ({lat}s), yielding anomaly score {candidate.signals['latency_anomaly']:.2f}.")

            # 5. Fact on downstream impact
            impact = candidate.signals.get("downstream_impact", 0.0)
            if impact > 0.0:
                facts.append(f"Followed by abnormal downstream behavior (downstream impact score: {impact:.2f}).")

            # 6. Tool mismatch against reference
            if candidate.signals.get("tool_mismatch", 0.0) > 0.5:
                ref_tool = "None"
                if reference_trace:
                    for rs in reference_trace.get("steps", []):
                        if rs.get("step_id") == candidate.step_id:
                            ref_tool = rs.get("tool_name") or "None"
                            break
                facts.append(f"Tool mismatch detected: target used '{tname}', whereas successful reference used '{ref_tool}'.")

            if not facts:
                facts.append(f"Observable execution properties at step {candidate.step_id} resulted in suspicion score {candidate.suspicion_score:.2f}.")

            packets.append(
                StepEvidencePacket(
                    run_id=run_id,
                    rank=idx,
                    step_id=candidate.step_id,
                    step_type=stype,
                    tool_name=tname,
                    suspicion_score=candidate.suspicion_score,
                    signals=candidate.signals,
                    trace_facts=facts,
                )
            )

        return packets
