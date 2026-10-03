"""Improved Rule-Based Failure Localizer for Black Box.

Combines the six structured signals into normalized suspicion scores using
validation-tunable weights. Ranks candidate steps and provides Top-1 / Top-3
candidates for investigation, with earliest-step tie-breaking.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from failure_intelligence.features.extractor import FEATURE_NAMES, FeatureExtractor, StepFeatureRecord

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "error_status": 0.30,
    "tool_mismatch": 0.20,
    "output_divergence": 0.20,
    "downstream_impact": 0.15,
    "state_divergence": 0.10,
    "latency_anomaly": 0.05,
}


@dataclass
class RankedCandidateStep:
    step_id: int
    step_type: str
    tool_name: Optional[str]
    suspicion_score: float
    signals: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "tool_name": self.tool_name,
            "suspicion_score": round(self.suspicion_score, 4),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
        }


@dataclass
class LocalizationResult:
    run_id: str
    method: str
    ranked_steps: List[RankedCandidateStep]
    weights_used: Dict[str, float]

    @property
    def top_1(self) -> Optional[RankedCandidateStep]:
        return self.ranked_steps[0] if self.ranked_steps else None

    @property
    def top_3(self) -> List[RankedCandidateStep]:
        return self.ranked_steps[:3]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "method": self.method,
            "ranked_steps": [s.to_dict() for s in self.ranked_steps],
            "weights_used": self.weights_used,
        }


class WeightedRuleLocalizer:
    """Ranks candidate execution steps using a weighted linear combination of the six signals."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self._normalize_weights()
        self.extractor = feature_extractor or FeatureExtractor()

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def localize(
        self,
        target_trace: Dict[str, Any],
        reference_trace: Optional[Dict[str, Any]] = None,
        feature_records: Optional[List[StepFeatureRecord]] = None,
    ) -> LocalizationResult:
        run_id = target_trace.get("run_id", "unknown_run")

        if feature_records is None:
            feature_records = self.extractor.extract_features(target_trace, reference_trace)

        ranked: List[RankedCandidateStep] = []

        for rec in feature_records:
            score = 0.0
            for sig_name, weight in self.weights.items():
                val = rec.features.get(sig_name, 0.0)
                score += weight * val

            # Normalize to [0.0, 1.0]
            score = max(0.0, min(1.0, score))

            ranked.append(
                RankedCandidateStep(
                    step_id=rec.step_id,
                    step_type=rec.step_type,
                    tool_name=rec.tool_name,
                    suspicion_score=score,
                    signals=rec.features,
                )
            )

        # Sort by suspicion_score descending; tie-breaker: earliest step (step_id ascending)
        ranked.sort(key=lambda s: (-s.suspicion_score, s.step_id))

        return LocalizationResult(
            run_id=run_id,
            method="rule_based",
            ranked_steps=ranked,
            weights_used={k: round(v, 4) for k, v in self.weights.items()},
        )

    def tune_weights_on_validation(
        self,
        validation_dataset: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], int]],
    ) -> Dict[str, float]:
        """Tunes signal weights using the validation dataset only. Never use test dataset!
        
        Args:
            validation_dataset: List of (target_trace, ref_trace, ground_truth_step_id)
        """
        if not validation_dataset:
            logger.warning("Empty validation dataset; keeping default weights.")
            return self.weights

        best_weights = copy.deepcopy(self.weights)
        best_top1 = -1.0

        # Grid-like perturbation over key signal weight combinations
        weight_candidates = [
            {"error_status": 0.35, "tool_mismatch": 0.25, "output_divergence": 0.20, "downstream_impact": 0.10, "state_divergence": 0.05, "latency_anomaly": 0.05},
            {"error_status": 0.25, "tool_mismatch": 0.25, "output_divergence": 0.25, "downstream_impact": 0.15, "state_divergence": 0.05, "latency_anomaly": 0.05},
            {"error_status": 0.40, "tool_mismatch": 0.20, "output_divergence": 0.20, "downstream_impact": 0.10, "state_divergence": 0.05, "latency_anomaly": 0.05},
            {"error_status": 0.20, "tool_mismatch": 0.20, "output_divergence": 0.30, "downstream_impact": 0.15, "state_divergence": 0.10, "latency_anomaly": 0.05},
            {"error_status": 0.30, "tool_mismatch": 0.20, "output_divergence": 0.20, "downstream_impact": 0.15, "state_divergence": 0.10, "latency_anomaly": 0.05},
        ]

        for cand_weights in weight_candidates:
            self.weights = cand_weights
            self._normalize_weights()

            correct_top1 = 0
            for target_trace, ref_trace, truth_step in validation_dataset:
                res = self.localize(target_trace, ref_trace)
                if res.top_1 and res.top_1.step_id == truth_step:
                    correct_top1 += 1

            acc = correct_top1 / len(validation_dataset)
            if acc > best_top1:
                best_top1 = acc
                best_weights = copy.deepcopy(self.weights)

        self.weights = best_weights
        self._normalize_weights()
        logger.info("Tuned weights on validation set: %s (Top-1 acc: %.2f)", self.weights, best_top1)
        return self.weights
