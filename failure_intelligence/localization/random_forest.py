"""Random Forest Failure Localizer for Black Box.

Trains a scikit-learn RandomForestClassifier on the six structured observable signals
to predict `is_responsible_step` (0 or 1). Ranks candidate steps by predicted suspicion
scores. Strictly enforces no-leakage: ground-truth labels are never used as features.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from failure_intelligence.features.extractor import FEATURE_NAMES, FeatureExtractor, StepFeatureRecord
from .rule_based import LocalizationResult, RankedCandidateStep

logger = logging.getLogger(__name__)


class RandomForestLocalizer:
    """Supervised ML model for failure step localization using the 6 observable signals."""

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        random_state: int = 42,
    ) -> None:
        self.extractor = feature_extractor or FeatureExtractor()
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
        )
        self.is_trained: bool = False

    def train_on_examples(
        self,
        train_examples: List[Any],  # List[LabeledExecutionExample]
        reference_traces_by_run: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Trains the Random Forest model using training examples only. Never use test set!
        
        Args:
            train_examples: Labeled examples from the training split.
            reference_traces_by_run: Optional mapping from run_id to reference trace.
        """
        X: List[List[float]] = []
        y: List[int] = []

        total_steps = 0
        positive_steps = 0

        for ex in train_examples:
            # Only train on failed runs with an identified ground-truth responsible step
            if not getattr(ex, "is_failure", False) or getattr(ex, "responsible_step_id", None) is None:
                continue

            ref_trace = None
            if reference_traces_by_run:
                ref_trace = reference_traces_by_run.get(ex.run_id)

            records = self.extractor.extract_features(ex.trace, ref_trace)
            resp_id = ex.responsible_step_id

            for rec in records:
                X.append(rec.to_vector())
                is_responsible = 1 if rec.step_id == resp_id else 0
                y.append(is_responsible)
                total_steps += 1
                if is_responsible == 1:
                    positive_steps += 1

        if len(X) < 10 or positive_steps == 0:
            logger.warning("Insufficient training data for Random Forest (rows=%d, positives=%d).", len(X), positive_steps)
            self.is_trained = False
            return {
                "status": "insufficient_data",
                "total_rows": len(X),
                "positive_steps": positive_steps,
            }

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=int)

        self.clf.fit(X_arr, y_arr)
        self.is_trained = True

        feature_importances = dict(zip(FEATURE_NAMES, [round(float(imp), 4) for imp in self.clf.feature_importances_]))
        logger.info("Random Forest trained on %d step records. Feature importances: %s", len(X), feature_importances)

        return {
            "status": "trained",
            "total_rows": len(X),
            "positive_steps": positive_steps,
            "feature_importances": feature_importances,
        }

    def localize(
        self,
        target_trace: Dict[str, Any],
        reference_trace: Optional[Dict[str, Any]] = None,
        feature_records: Optional[List[StepFeatureRecord]] = None,
    ) -> LocalizationResult:
        """Ranks candidate steps in target_trace by predicted suspicion score."""
        run_id = target_trace.get("run_id", "unknown_run")

        if feature_records is None:
            feature_records = self.extractor.extract_features(target_trace, reference_trace)

        if not self.is_trained:
            logger.warning("RandomForestLocalizer is not trained. Assigning zero suspicion scores.")
            ranked = [
                RankedCandidateStep(
                    step_id=r.step_id,
                    step_type=r.step_type,
                    tool_name=r.tool_name,
                    suspicion_score=0.0,
                    signals=r.features,
                )
                for r in feature_records
            ]
            return LocalizationResult(
                run_id=run_id,
                method="random_forest",
                ranked_steps=ranked,
                weights_used={"status": "untrained"},
            )

        X = np.array([r.to_vector() for r in feature_records], dtype=float)
        # Predict probability of being responsible step (class 1)
        probs = self.clf.predict_proba(X)
        class_1_idx = 1 if self.clf.classes_[1] == 1 else 0

        ranked = []
        for idx, rec in enumerate(feature_records):
            score = float(probs[idx][class_1_idx])
            ranked.append(
                RankedCandidateStep(
                    step_id=rec.step_id,
                    step_type=rec.step_type,
                    tool_name=rec.tool_name,
                    suspicion_score=score,
                    signals=rec.features,
                )
            )

        # Sort by suspicion score descending; tie-breaker: earliest step (step_id ascending)
        ranked.sort(key=lambda s: (-s.suspicion_score, s.step_id))

        importances = dict(zip(FEATURE_NAMES, [round(float(imp), 4) for imp in self.clf.feature_importances_]))

        return LocalizationResult(
            run_id=run_id,
            method="random_forest",
            ranked_steps=ranked,
            weights_used=importances,
        )
