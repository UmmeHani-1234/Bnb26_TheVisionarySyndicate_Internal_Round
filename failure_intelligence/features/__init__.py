"""Feature extraction package for Black Box failure intelligence."""

from .extractor import FEATURE_NAMES, FeatureExtractor, StepFeatureRecord
from .reference import ReferenceMatch, ReferenceSelector
from .signals import (
    compute_downstream_impact,
    compute_error_status,
    compute_latency_anomaly,
    compute_output_divergence,
    compute_state_divergence,
    compute_tool_mismatch,
)

__all__ = [
    "FEATURE_NAMES",
    "FeatureExtractor",
    "StepFeatureRecord",
    "ReferenceSelector",
    "ReferenceMatch",
    "compute_tool_mismatch",
    "compute_output_divergence",
    "compute_state_divergence",
    "compute_downstream_impact",
    "compute_latency_anomaly",
    "compute_error_status",
]
