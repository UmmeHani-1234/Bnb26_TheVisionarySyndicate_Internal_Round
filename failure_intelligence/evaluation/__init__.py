"""Evaluation module for Black Box failure intelligence."""

from .dataset import DatasetManager, DatasetSplits, LabeledExecutionExample
from .evaluator import BenchmarkEvaluator
from .metrics import (
    CategoryEvaluationSummary,
    MethodEvaluationSummary,
    compute_random_baseline,
)
from .verifier import IndependentVerifier, VerificationResult

__all__ = [
    "IndependentVerifier",
    "VerificationResult",
    "DatasetManager",
    "DatasetSplits",
    "LabeledExecutionExample",
    "BenchmarkEvaluator",
    "MethodEvaluationSummary",
    "CategoryEvaluationSummary",
    "compute_random_baseline",
]
