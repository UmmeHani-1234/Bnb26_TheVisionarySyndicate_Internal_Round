"""Service layer coordinating Failure Intelligence, Trace Comparison, and Evaluation."""

import logging
from typing import Any, Dict, List, Optional

from storage.repository import TraceRepository
from failure_intelligence.comparison.comparator import TraceComparator, TraceComparisonResult
from failure_intelligence.evidence.builder import EvidenceBuilder
from failure_intelligence.features.extractor import FeatureExtractor
from failure_intelligence.features.reference import ReferenceSelector
from failure_intelligence.localization.random_forest import RandomForestLocalizer
from failure_intelligence.localization.rule_based import WeightedRuleLocalizer
from failure_intelligence.evaluation.dataset import DatasetManager, DatasetSplits
from failure_intelligence.evaluation.evaluator import BenchmarkEvaluator
from failure_intelligence.evaluation.verifier import IndependentVerifier

logger = logging.getLogger(__name__)


class FailureIntelligenceService:
    """Coordinates failure intelligence, reference selection, comparison, and evaluation."""

    _cached_evaluation: Optional[Dict[str, Any]] = None

    def __init__(self, repository: TraceRepository) -> None:
        self.repo = repository
        self.verifier = IndependentVerifier()
        self.ref_selector = ReferenceSelector()
        self.extractor = FeatureExtractor()
        self.rule_localizer = WeightedRuleLocalizer(feature_extractor=self.extractor)
        self.rf_localizer = RandomForestLocalizer(feature_extractor=self.extractor)
        self.evidence_builder = EvidenceBuilder(
            rule_localizer=self.rule_localizer,
            rf_localizer=self.rf_localizer,
            ref_selector=self.ref_selector,
        )
        self.comparator = TraceComparator(verifier=self.verifier)
        self.dataset_manager = DatasetManager(repository=self.repo, verifier=self.verifier)
        self.evaluator = BenchmarkEvaluator(
            rule_localizer=self.rule_localizer,
            rf_localizer=self.rf_localizer,
            reference_selector=self.ref_selector,
            feature_extractor=self.extractor,
        )

    def compare_traces(
        self,
        original_run_id: str,
        alternative_run_id: str,
        reference_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compares original failed run, alternative run, and successful reference run."""
        orig_trace = self.repo.get_run_trace(original_run_id)
        if not orig_trace:
            raise ValueError(f"Original run '{original_run_id}' not found.")

        alt_trace = self.repo.get_run_trace(alternative_run_id)
        if not alt_trace:
            raise ValueError(f"Alternative run '{alternative_run_id}' not found.")

        ref_trace = None
        if reference_run_id:
            ref_trace = self.repo.get_run_trace(reference_run_id)
        else:
            # Auto-select best reference from successful runs in repository
            all_runs = self.repo.list_runs(limit=100)
            candidate_traces = []
            for r in all_runs:
                if r.get("status") == "success" and r.get("run_id") not in (original_run_id, alternative_run_id):
                    t = self.repo.get_run_trace(r["run_id"])
                    if t:
                        candidate_traces.append(t)
            ref_trace = self.ref_selector.select_reference(orig_trace, candidate_traces)

        res = self.comparator.compare_traces(orig_trace, alt_trace, ref_trace)
        return res.to_dict()

    def diagnose_run(self, run_id: str) -> Dict[str, Any]:
        """Produces evidence-backed failure diagnosis for a run."""
        trace = self.repo.get_run_trace(run_id)
        if not trace:
            raise ValueError(f"Run '{run_id}' not found.")

        # Candidate reference traces from previous successful runs
        all_runs = self.repo.list_runs(limit=50)
        cand_refs = [
            self.repo.get_run_trace(r["run_id"])
            for r in all_runs
            if r.get("status") == "success" and r.get("run_id") != run_id
        ]
        cand_refs = [t for t in cand_refs if t]

        ref_trace = self.ref_selector.select_reference(trace, cand_refs)
        packet = self.evidence_builder.build_evidence(trace, ref_trace)
        return packet.to_dict()

    def get_or_run_benchmark(
        self,
        force_regenerate: bool = False,
        target_success_count: int = 40,
        target_failure_count: int = 80,
    ) -> Dict[str, Any]:
        """Returns cached benchmark evaluation or executes benchmark generation and evaluation."""
        if not force_regenerate and FailureIntelligenceService._cached_evaluation is not None:
            return FailureIntelligenceService._cached_evaluation

        # Collect replay comparisons to evaluate recovery
        replay_comparisons = self._gather_replay_comparisons()

        logger.info("Generating evaluation dataset: %d success, %d failure...", target_success_count, target_failure_count)
        dataset = self.dataset_manager.generate_benchmark_dataset(
            target_success_count=target_success_count,
            target_failure_count=target_failure_count,
        )

        splits = self.dataset_manager.split_dataset(dataset)
        eval_result = self.evaluator.run_full_evaluation(splits, replay_comparisons=replay_comparisons)

        FailureIntelligenceService._cached_evaluation = eval_result
        return eval_result

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """GET /evaluation/summary"""
        if FailureIntelligenceService._cached_evaluation is None:
            return {
                "status": "evaluation_not_ready",
                "reason": "Evaluation benchmark has not been run yet. Please invoke /evaluation/run-benchmark or run via UI.",
            }
        eval_data = FailureIntelligenceService._cached_evaluation
        return {
            "status": eval_data.get("status"),
            "summary": eval_data.get("summary", {}),
            "known_vs_held_out": eval_data.get("known_vs_held_out", {}),
        }

    def get_evaluation_localization(self) -> Dict[str, Any]:
        """GET /evaluation/localization"""
        if FailureIntelligenceService._cached_evaluation is None:
            return {
                "status": "evaluation_not_ready",
                "reason": "Evaluation benchmark has not been run yet.",
            }
        eval_data = FailureIntelligenceService._cached_evaluation
        return {
            "status": eval_data.get("status"),
            "summary": eval_data.get("summary", {}),
            "known_vs_held_out": eval_data.get("known_vs_held_out", {}),
        }

    def get_evaluation_by_category(self) -> Dict[str, Any]:
        """GET /evaluation/by-category"""
        if FailureIntelligenceService._cached_evaluation is None:
            return {
                "status": "evaluation_not_ready",
                "reason": "Evaluation benchmark has not been run yet.",
                "categories": [],
            }
        eval_data = FailureIntelligenceService._cached_evaluation
        return {
            "status": eval_data.get("status"),
            "categories": eval_data.get("by_category", []),
        }

    def get_evaluation_replay(self) -> Dict[str, Any]:
        """GET /evaluation/replay"""
        if FailureIntelligenceService._cached_evaluation is None:
            # Attempt to gather live replay recovery data from existing db replays
            replay_comparisons = self._gather_replay_comparisons()
            replay_summary = self.evaluator._compute_replay_summary(replay_comparisons)
            return {
                "status": "partial",
                "replay_recovery": replay_summary,
            }
        eval_data = FailureIntelligenceService._cached_evaluation
        return {
            "status": eval_data.get("status"),
            "replay_recovery": eval_data.get("replay_recovery", {}),
        }

    def _gather_replay_comparisons(self) -> List[Dict[str, Any]]:
        """Gathers trace comparisons for all existing replay pairs in the repository."""
        comparisons = []
        try:
            all_runs = self.repo.list_runs(limit=100)
            for r in all_runs:
                run_id = r.get("run_id")
                replays = self.repo.list_replays_for_run(run_id)
                for rep in replays:
                    alt_id = rep.get("replay_run_id")
                    if alt_id:
                        try:
                            comp = self.compare_traces(original_run_id=run_id, alternative_run_id=alt_id)
                            comparisons.append(comp)
                        except Exception:
                            pass
        except Exception as exc:
            logger.warning("Error gathering replay comparisons: %s", exc)
        return comparisons
