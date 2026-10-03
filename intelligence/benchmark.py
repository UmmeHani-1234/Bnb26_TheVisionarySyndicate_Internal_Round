"""Evaluation suite and benchmark dataset for Black Box Failure Intelligence."""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from agent.agent import LaptopAgent
from recorder.recorder import ExecutionRecorder
from storage.repository import TraceRepository
from .rules import RuleBasedLocalizer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkScenario:
    scenario_id: str
    description: str
    request: str
    failure_mode: str
    expected_failure: bool
    expected_failure_type: str
    # Ground truth for evaluation only (never passed to localizer)
    ground_truth_step_type: str
    ground_truth_step_name: Optional[str] = None


BENCHMARK_SCENARIOS: List[BenchmarkScenario] = [
    BenchmarkScenario(
        scenario_id="bench-001",
        description="Happy path: Student budget under ₹40,000",
        request="Find a budget laptop for a student under ₹40,000",
        failure_mode="none",
        expected_failure=False,
        expected_failure_type="none",
        ground_truth_step_type="none",
    ),
    BenchmarkScenario(
        scenario_id="bench-002",
        description="Happy path: Programming & gaming laptop under ₹70,000",
        request="Find a laptop under ₹70,000 suitable for programming and gaming with 16GB RAM",
        failure_mode="none",
        expected_failure=False,
        expected_failure_type="none",
        ground_truth_step_type="none",
    ),
    BenchmarkScenario(
        scenario_id="bench-003",
        description="Budget constraint violation: Agent selects ₹84,990 laptop for ₹70,000 budget",
        request="Find a laptop strictly under ₹70,000 for web development",
        failure_mode="budget_violation",
        expected_failure=True,
        expected_failure_type="budget_violation",
        ground_truth_step_type="final_response_generated",
    ),
    BenchmarkScenario(
        scenario_id="bench-004",
        description="Result interpretation mismatch: Tool output ₹64,990 stated as ₹85,000",
        request="Recommend a gaming laptop under ₹1,00,000",
        failure_mode="wrong_interpretation",
        expected_failure=True,
        expected_failure_type="wrong_interpretation",
        ground_truth_step_type="final_response_generated",
    ),
    BenchmarkScenario(
        scenario_id="bench-005",
        description="Unexpected tool output: Tool returns malformed data with null price",
        request="Find an ultrabook under ₹60,000",
        failure_mode="unexpected_output",
        expected_failure=True,
        expected_failure_type="unexpected_tool_output",
        ground_truth_step_type="tool_completed",
    ),
    BenchmarkScenario(
        scenario_id="bench-006",
        description="Tool execution timeout / failure: Tool fails during execution",
        request="Find a laptop with 16GB RAM",
        failure_mode="timeout",
        expected_failure=True,
        expected_failure_type="timeout",
        ground_truth_step_type="tool_error",
    ),
    BenchmarkScenario(
        scenario_id="bench-007",
        description="Incorrect tool selection: Agent calls specs directly without retrieval",
        request="Find a laptop under ₹50,000",
        failure_mode="wrong_tool",
        expected_failure=True,
        expected_failure_type="wrong_tool",
        ground_truth_step_type="tool_selected",
    ),
]


@dataclass
class EvaluationMetrics:
    total_runs: int
    failure_runs_count: int
    success_runs_count: int
    top1_correct: int
    top3_correct: int
    top1_accuracy: float
    top3_accuracy: float
    details: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """Runs controlled benchmark scenarios and calculates objective localization accuracy."""

    def __init__(self, repository: Optional[TraceRepository] = None) -> None:
        self.repository = repository or TraceRepository()
        self.localizer = RuleBasedLocalizer()

    def run_benchmark(self, scenarios: Optional[List[BenchmarkScenario]] = None) -> EvaluationMetrics:
        target_scenarios = scenarios or BENCHMARK_SCENARIOS
        details: List[Dict[str, Any]] = []

        top1_correct = 0
        top3_correct = 0
        failure_count = 0

        import uuid
        for sc in target_scenarios:
            run_id = f"bench-{sc.scenario_id}-{uuid.uuid4().hex[:6]}"
            recorder = ExecutionRecorder(run_id=run_id, repository=self.repository)
            agent = LaptopAgent(sinks=[recorder.record])

            # Run agent with controlled failure injection
            result = agent.run(sc.request, failure_mode=sc.failure_mode)

            # Retrieve complete recorded trace from storage
            trace = self.repository.get_run_trace(run_id)
            if not trace:
                continue

            # Store ground truth metadata in run for evaluation record
            self.repository.update_run(
                run_id=run_id,
                failure_metadata={
                    "scenario_id": sc.scenario_id,
                    "expected_failure": sc.expected_failure,
                    "expected_failure_type": sc.expected_failure_type,
                    "ground_truth_step_type": sc.ground_truth_step_type,
                },
            )

            # Localizer independently diagnoses trace (NO ground truth passed!)
            diagnosis = self.localizer.diagnose(trace)

            is_top1 = False
            is_top3 = False

            if sc.expected_failure:
                failure_count += 1
                # Check top-1 match
                if diagnosis.likely_responsible_step:
                    if diagnosis.likely_responsible_step.step_type == sc.ground_truth_step_type:
                        is_top1 = True

                # Check top-3 match
                top3_types = [s.step_type for s in diagnosis.top_suspected_steps[:3]]
                if sc.ground_truth_step_type in top3_types:
                    is_top3 = True

                if is_top1:
                    top1_correct += 1
                if is_top3:
                    top3_correct += 1
            else:
                # Normal run: correct if no failure detected
                if not diagnosis.failure_detected:
                    is_top1 = True
                    is_top3 = True

            details.append({
                "scenario_id": sc.scenario_id,
                "description": sc.description,
                "run_id": run_id,
                "expected_failure_type": sc.expected_failure_type,
                "predicted_failure_type": diagnosis.failure_type,
                "ground_truth_step_type": sc.ground_truth_step_type,
                "predicted_step_type": diagnosis.likely_responsible_step.step_type if diagnosis.likely_responsible_step else None,
                "is_top1_match": is_top1,
                "is_top3_match": is_top3,
                "confidence": diagnosis.confidence,
                "evidence_count": len(diagnosis.evidence),
            })

        eval_denominator = failure_count if failure_count > 0 else len(target_scenarios)
        top1_acc = round((top1_correct / eval_denominator) * 100, 1)
        top3_acc = round((top3_correct / eval_denominator) * 100, 1)

        return EvaluationMetrics(
            total_runs=len(target_scenarios),
            failure_runs_count=failure_count,
            success_runs_count=len(target_scenarios) - failure_count,
            top1_correct=top1_correct,
            top3_correct=top3_correct,
            top1_accuracy=top1_acc,
            top3_accuracy=top3_acc,
            details=details,
        )
