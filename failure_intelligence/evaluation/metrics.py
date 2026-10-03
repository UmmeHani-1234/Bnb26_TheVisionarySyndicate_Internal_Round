"""Evaluation metrics and statistical baselines for failure localization."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MethodEvaluationSummary:
    method: str
    total_test_runs: int
    top1_correct: int
    top3_correct: int
    top1_accuracy: float
    top3_accuracy: float
    mrr: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "total_test_runs": self.total_test_runs,
            "top1_correct": self.top1_correct,
            "top3_correct": self.top3_correct,
            "top1_accuracy": round(self.top1_accuracy, 4),
            "top3_accuracy": round(self.top3_accuracy, 4),
            "mrr": round(self.mrr, 4),
        }


@dataclass
class CategoryEvaluationSummary:
    category: str
    is_held_out: bool
    total_test_runs: int
    small_sample_warning: bool
    rule_based_top1: float
    rule_based_top3: float
    random_forest_top1: float
    random_forest_top3: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "is_held_out": self.is_held_out,
            "total_test_runs": self.total_test_runs,
            "small_sample_warning": self.small_sample_warning,
            "rule_based_top1": round(self.rule_based_top1, 4),
            "rule_based_top3": round(self.rule_based_top3, 4),
            "random_forest_top1": round(self.random_forest_top1, 4),
            "random_forest_top3": round(self.random_forest_top3, 4),
        }


def compute_random_baseline(avg_candidates: float) -> MethodEvaluationSummary:
    """Calculates theoretical random guessing baseline for N candidate steps."""
    n = max(avg_candidates, 1.0)
    top1 = min(1.0, 1.0 / n)
    top3 = min(1.0, 3.0 / n)
    harmonic_sum = sum(1.0 / i for i in range(1, int(n) + 1))
    mrr = harmonic_sum / n

    return MethodEvaluationSummary(
        method="random_baseline",
        total_test_runs=0,
        top1_correct=0,
        top3_correct=0,
        top1_accuracy=top1,
        top3_accuracy=top3,
        mrr=mrr,
    )
