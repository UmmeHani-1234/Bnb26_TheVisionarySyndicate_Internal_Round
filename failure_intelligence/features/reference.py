"""Reference trace selection for Black Box failure intelligence.

Retrieves verified successful execution traces to serve as comparative baselines
for feature extraction, trace divergence measurement, and failure localization.
Enforces strict split isolation so test runs are never selected as references.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from storage.repository import TraceRepository
from failure_intelligence.evaluation.verifier import IndependentVerifier

logger = logging.getLogger(__name__)


@dataclass
class ReferenceMatch:
    reference_run_id: str
    match_mode: str  # "same_input", "compatible_intent", "compatible_pool"
    similarity_score: float
    reference_trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_run_id": self.reference_run_id,
            "match_mode": self.match_mode,
            "similarity_score": round(self.similarity_score, 3),
        }


class ReferenceSelector:
    """Selects valid, verified successful reference runs for comparative analysis."""

    def __init__(
        self,
        repository: Optional[TraceRepository] = None,
        verifier: Optional[IndependentVerifier] = None,
        allowed_run_ids: Optional[Set[str]] = None,
    ) -> None:
        self.repo = repository
        self.verifier = verifier or IndependentVerifier()
        # Enforces train/test split isolation when set
        self.allowed_run_ids = set(allowed_run_ids) if allowed_run_ids is not None else None

    def select_reference(
        self,
        target_trace: Dict[str, Any],
        candidate_traces: Optional[List[Dict[str, Any]]] = None,
        allowed_run_ids: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Selects a verified reference trace from candidates or repository."""
        effective_allowed = set(allowed_run_ids) if allowed_run_ids is not None else self.allowed_run_ids

        # If candidate_traces provided, search within them directly
        if candidate_traces is not None:
            target_query = (target_trace.get("user_request") or "").strip().lower()
            target_run_id = target_trace.get("run_id", "")

            candidates = [
                t for t in candidate_traces
                if t.get("run_id") != target_run_id
                and t.get("status") == "success"
                and (effective_allowed is None or t.get("run_id") in effective_allowed)
                and self.verifier.verify(t).passed
            ]
            if not candidates:
                return None

            # 1. Same input match
            for c in candidates:
                c_query = (c.get("user_request") or "").strip().lower()
                if target_query and c_query == target_query:
                    return c

            # 2. Compatible intent
            best_c = None
            best_sim = -1.0
            for c in candidates:
                c_query = (c.get("user_request") or "").strip().lower()
                sim = self._compute_query_similarity(target_query, c_query)
                if sim > best_sim:
                    best_sim = sim
                    best_c = c

            return best_c if best_c is not None else candidates[0]

        # Otherwise fallback to repository lookup
        if self.repo is not None:
            match = self.find_reference_trace(target_trace)
            return match.reference_trace if match else None
        return None

    def find_reference_trace(
        self,
        target_trace: Dict[str, Any],
        mode: str = "same_input",
    ) -> Optional[ReferenceMatch]:
        """Finds the most suitable verified successful reference trace for target_trace.
        
        Args:
            target_trace: The trace being analyzed (may be failed or alternative).
            mode: "same_input" tries exact query match first, then falls back to compatible pool.
        """
        if not self.repo:
            return None

        target_run_id = target_trace.get("run_id", "")
        target_query = (target_trace.get("user_request") or "").strip().lower()

        # 1. Fetch runs from repository
        runs = self.repo.list_runs(limit=100)

        # 2. Filter candidates:
        # - Must not be the target run itself
        # - Must be marked 'success'
        # - Must belong to allowed_run_ids (if split-restricted)
        candidates = []
        for r in runs:
            rid = r.get("run_id")
            if rid == target_run_id:
                continue
            if r.get("status") != "success":
                continue
            if self.allowed_run_ids is not None and rid not in self.allowed_run_ids:
                continue
            candidates.append(r)

        if not candidates:
            logger.warning("No eligible candidate reference runs found in repository.")
            return None

        # 3. Check for exact same_input match
        best_exact: Optional[ReferenceMatch] = None
        compatible_matches: List[ReferenceMatch] = []

        for c in candidates:
            rid = c.get("run_id")
            trace = self.repo.get_run_trace(rid)
            if not trace:
                continue

            # Must pass the independent success verifier
            v_res = self.verifier.verify(trace)
            if not v_res.passed:
                continue

            c_query = (trace.get("user_request") or "").strip().lower()

            if target_query and c_query == target_query:
                best_exact = ReferenceMatch(
                    reference_run_id=rid,
                    match_mode="same_input",
                    similarity_score=1.0,
                    reference_trace=trace,
                )
                break

            # Compute intent/token compatibility
            sim = self._compute_query_similarity(target_query, c_query)
            if sim > 0.3:
                compatible_matches.append(
                    ReferenceMatch(
                        reference_run_id=rid,
                        match_mode="compatible_intent",
                        similarity_score=sim,
                        reference_trace=trace,
                    )
                )

        if best_exact:
            return best_exact

        # If same_input mode requested but no exact match found, fall back to best compatible
        if compatible_matches:
            compatible_matches.sort(key=lambda m: m.similarity_score, reverse=True)
            return compatible_matches[0]

        # Final fallback: any verified successful run from allowed pool
        for c in candidates:
            rid = c.get("run_id")
            trace = self.repo.get_run_trace(rid)
            if trace and self.verifier.verify(trace).passed:
                return ReferenceMatch(
                    reference_run_id=rid,
                    match_mode="compatible_pool",
                    similarity_score=0.2,
                    reference_trace=trace,
                )

        return None

    @staticmethod
    def _compute_query_similarity(query_a: str, query_b: str) -> float:
        """Computes Jaccard word similarity between two requests."""
        if not query_a or not query_b:
            return 0.0
        words_a = set(query_a.split())
        words_b = set(query_b.split())
        union = words_a | words_b
        if not union:
            return 0.0
        return len(words_a & words_b) / len(union)
