"""Benchmark Evaluator for Black Box Stage 6.

Evaluates Failure Localization models (Rule-Based, Random Forest, Random Baseline)
on leakage-safe test splits, reporting Top-1, Top-3, MRR, Known vs Held-out breakdowns,
per-category performance, and Replay Recovery verification.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from failure_intelligence.features.extractor import FeatureExtractor
from failure_intelligence.features.reference import ReferenceSelector
from failure_intelligence.localization.random_forest import RandomForestLocalizer
from failure_intelligence.localization.rule_based import WeightedRuleLocalizer
from failure_intelligence.evaluation.dataset import DatasetSplits, LabeledExecutionExample
from failure_intelligence.evaluation.metrics import (
    CategoryEvaluationSummary,
    MethodEvaluationSummary,
    compute_random_baseline,
)

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluates Black Box failure intelligence models across test splits."""

    def __init__(
        self,
        rule_localizer: Optional[WeightedRuleLocalizer] = None,
        rf_localizer: Optional[RandomForestLocalizer] = None,
        reference_selector: Optional[ReferenceSelector] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        self.extractor = feature_extractor or FeatureExtractor()
        self.rule_localizer = rule_localizer or WeightedRuleLocalizer(feature_extractor=self.extractor)
        self.rf_localizer = rf_localizer or RandomForestLocalizer(feature_extractor=self.extractor)
        self.ref_selector = reference_selector or ReferenceSelector()

    def run_full_evaluation(
        self,
        splits: DatasetSplits,
        replay_comparisons: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Runs complete benchmark evaluation following strict leakage-safety rules.
        
        1. Trains RF model on train split only.
        2. Tunes rule weights on validation split only.
        3. Evaluates on test split (Known categories) and held_out_test split (Held-out category).
        4. Reference traces are selected strictly from train/reference split.
        """
        # Validate test splits
        failed_test_runs = [e for e in splits.test if e.is_failure and e.responsible_step_id is not None]
        failed_held_out = [e for e in splits.held_out_test if e.is_failure and e.responsible_step_id is not None]

        if not failed_test_runs:
            return {
                "status": "evaluation_not_ready",
                "reason": "Insufficient test failure runs in test split for evaluation.",
                "total_test_runs": 0,
            }

        # 1. Train Random Forest model on training split only
        train_ref_traces = self._build_training_reference_map(splits.train)
        self.rf_localizer.train_on_examples(splits.train, train_ref_traces)

        # 2. Tune Rule-Based weights on validation split only
        val_dataset = []
        for val_ex in splits.validation:
            if val_ex.is_failure and val_ex.responsible_step_id is not None:
                # Reference from train split only
                ref_trace = self.ref_selector.select_reference(
                    val_ex.trace,
                    candidate_traces=[t.trace for t in splits.train if not t.is_failure],
                    allowed_run_ids=splits.train_run_ids,
                )
                val_dataset.append((val_ex.trace, ref_trace, val_ex.responsible_step_id))

        if val_dataset:
            self.rule_localizer.tune_weights_on_validation(val_dataset)

        # Reference pool for testing must come ONLY from train split
        train_success_traces = [t.trace for t in splits.train if not t.is_failure]

        # 3. Evaluate Known Categories on Test Split
        known_rule_metrics, known_rf_metrics, known_candidates = self._evaluate_examples(
            failed_test_runs, train_success_traces, splits.train_run_ids
        )

        # 4. Evaluate Held-out Category on Held-out Test Split
        held_out_rule_metrics, held_out_rf_metrics, _ = self._evaluate_examples(
            failed_held_out, train_success_traces, splits.train_run_ids
        )

        # 5. Combined summary across all tested failure runs (Known + Held Out)
        all_failed_tests = failed_test_runs + failed_held_out
        comb_rule, comb_rf, total_cand_counts = self._evaluate_examples(
            all_failed_tests, train_success_traces, splits.train_run_ids
        )

        avg_candidates = sum(total_cand_counts) / len(total_cand_counts) if total_cand_counts else 8.0
        random_baseline = compute_random_baseline(avg_candidates)

        # 6. Per-Category Breakdown
        category_summaries = self._compute_category_breakdown(
            all_failed_tests, train_success_traces, splits.train_run_ids, splits.held_out_category
        )

        # 7. Replay Recovery Evaluation
        replay_summary = self._compute_replay_summary(replay_comparisons)

        return {
            "status": "completed",
            "summary": {
                "total_test_runs": len(all_failed_tests),
                "random_baseline": random_baseline.to_dict(),
                "rule_based": comb_rule.to_dict(),
                "random_forest": comb_rf.to_dict(),
            },
            "known_vs_held_out": {
                "known_categories": {
                    "total_runs": len(failed_test_runs),
                    "rule_based": {
                        "top1": round(known_rule_metrics.top1_accuracy, 4),
                        "top3": round(known_rule_metrics.top3_accuracy, 4),
                        "mrr": round(known_rule_metrics.mrr, 4),
                    },
                    "random_forest": {
                        "top1": round(known_rf_metrics.top1_accuracy, 4),
                        "top3": round(known_rf_metrics.top3_accuracy, 4),
                        "mrr": round(known_rf_metrics.mrr, 4),
                    },
                },
                "held_out_category": {
                    "category": splits.held_out_category,
                    "total_runs": len(failed_held_out),
                    "rule_based": {
                        "top1": round(held_out_rule_metrics.top1_accuracy, 4),
                        "top3": round(held_out_rule_metrics.top3_accuracy, 4),
                        "mrr": round(held_out_rule_metrics.mrr, 4),
                    },
                    "random_forest": {
                        "top1": round(held_out_rf_metrics.top1_accuracy, 4),
                        "top3": round(held_out_rf_metrics.top3_accuracy, 4),
                        "mrr": round(held_out_rf_metrics.mrr, 4),
                    },
                },
            },
            "by_category": [c.to_dict() for c in category_summaries],
            "replay_recovery": replay_summary,
        }

    def _evaluate_examples(
        self,
        examples: List[LabeledExecutionExample],
        reference_traces: List[Dict[str, Any]],
        allowed_ref_ids: Set[str],
    ) -> Tuple[MethodEvaluationSummary, MethodEvaluationSummary, List[int]]:
        """Evaluates rule-based and random forest models on a list of examples."""
        total = len(examples)
        if total == 0:
            zero_summary = lambda name: MethodEvaluationSummary(
                method=name,
                total_test_runs=0,
                top1_correct=0,
                top3_correct=0,
                top1_accuracy=0.0,
                top3_accuracy=0.0,
                mrr=0.0,
            )
            return zero_summary("rule_based"), zero_summary("random_forest"), []

        rule_top1 = 0
        rule_top3 = 0
        rule_rr_sum = 0.0

        rf_top1 = 0
        rf_top3 = 0
        rf_rr_sum = 0.0

        candidate_counts: List[int] = []

        for ex in examples:
            ref_trace = self.ref_selector.select_reference(
                ex.trace,
                candidate_traces=reference_traces,
                allowed_run_ids=allowed_ref_ids,
            )
            resp_step = ex.responsible_step_id

            # Rule-based prediction
            rule_res = self.rule_localizer.localize(ex.trace, ref_trace)
            cand_count = len(rule_res.ranked_steps)
            candidate_counts.append(cand_count if cand_count > 0 else 8)

            # Check rule top1
            if rule_res.top_1 and rule_res.top_1.step_id == resp_step:
                rule_top1 += 1
            # Check rule top3
            top3_ids = [s.step_id for s in rule_res.top_3]
            if resp_step in top3_ids:
                rule_top3 += 1
            # Rule RR
            rank_rule = next((idx + 1 for idx, s in enumerate(rule_res.ranked_steps) if s.step_id == resp_step), None)
            if rank_rule:
                rule_rr_sum += 1.0 / rank_rule

            # Random Forest prediction
            rf_res = self.rf_localizer.localize(ex.trace, ref_trace)
            if rf_res.top_1 and rf_res.top_1.step_id == resp_step:
                rf_top1 += 1
            rf_top3_ids = [s.step_id for s in rf_res.top_3]
            if resp_step in rf_top3_ids:
                rf_top3 += 1
            rank_rf = next((idx + 1 for idx, s in enumerate(rf_res.ranked_steps) if s.step_id == resp_step), None)
            if rank_rf:
                rf_rr_sum += 1.0 / rank_rf

        rule_summary = MethodEvaluationSummary(
            method="rule_based",
            total_test_runs=total,
            top1_correct=rule_top1,
            top3_correct=rule_top3,
            top1_accuracy=rule_top1 / total,
            top3_accuracy=rule_top3 / total,
            mrr=rule_rr_sum / total,
        )

        rf_summary = MethodEvaluationSummary(
            method="random_forest",
            total_test_runs=total,
            top1_correct=rf_top1,
            top3_correct=rf_top3,
            top1_accuracy=rf_top1 / total,
            top3_accuracy=rf_top3 / total,
            mrr=rf_rr_sum / total,
        )

        return rule_summary, rf_summary, candidate_counts

    def _compute_category_breakdown(
        self,
        examples: List[LabeledExecutionExample],
        reference_traces: List[Dict[str, Any]],
        allowed_ref_ids: Set[str],
        held_out_category: str,
    ) -> List[CategoryEvaluationSummary]:
        """Calculates per-category evaluation metrics with small sample size warnings."""
        from collections import defaultdict

        by_cat: Dict[str, List[LabeledExecutionExample]] = defaultdict(list)
        for ex in examples:
            by_cat[ex.failure_type].append(ex)

        summaries: List[CategoryEvaluationSummary] = []
        for cat, cat_examples in by_cat.items():
            rule_sum, rf_sum, _ = self._evaluate_examples(cat_examples, reference_traces, allowed_ref_ids)
            total = len(cat_examples)
            summaries.append(
                CategoryEvaluationSummary(
                    category=cat,
                    is_held_out=(cat == held_out_category),
                    total_test_runs=total,
                    small_sample_warning=(total < 5),
                    rule_based_top1=rule_sum.top1_accuracy,
                    rule_based_top3=rule_sum.top3_accuracy,
                    random_forest_top1=rf_sum.top1_accuracy,
                    random_forest_top3=rf_sum.top3_accuracy,
                )
            )

        summaries.sort(key=lambda c: (c.is_held_out, c.category))
        return summaries

    def _compute_replay_summary(
        self,
        replay_comparisons: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Summarizes recovery verification across alternative runs."""
        if not replay_comparisons:
            return {
                "branches_attempted": 0,
                "recovered": 0,
                "not_recovered": 0,
                "recovery_rate": 0.0,
            }

        attempted = len(replay_comparisons)
        recovered = sum(1 for c in replay_comparisons if c.get("recovered") is True)
        not_recovered = attempted - recovered
        rate = recovered / attempted if attempted > 0 else 0.0

        return {
            "branches_attempted": attempted,
            "recovered": recovered,
            "not_recovered": not_recovered,
            "recovery_rate": round(rate, 4),
        }

    def _build_training_reference_map(
        self, train_examples: List[LabeledExecutionExample]
    ) -> Dict[str, Dict[str, Any]]:
        """Maps training failed runs to a compatible training success run."""
        success_traces = [e.trace for e in train_examples if not e.is_failure]
        ref_map: Dict[str, Dict[str, Any]] = {}
        for ex in train_examples:
            if ex.is_failure:
                ref = self.ref_selector.select_reference(ex.trace, success_traces)
                if ref:
                    ref_map[ex.run_id] = ref
        return ref_map
