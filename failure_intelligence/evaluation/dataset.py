"""Benchmark dataset generator and leakage-safe dataset splitting for Black Box.

Generates labeled execution runs using controlled fault injection, verifies failures
using the IndependentVerifier, and partitions runs into Train, Validation, and Test
splits with a held-out failure category. Never leaks ground-truth labels into features.
"""

import copy
import logging
import random
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.agent import LaptopAgent
from recorder.recorder import ExecutionRecorder
from storage.repository import TraceRepository
from failure_intelligence.evaluation.verifier import IndependentVerifier, VerificationResult

logger = logging.getLogger(__name__)

# Configurable failure categories
KNOWN_FAILURE_CATEGORIES = [
    "budget_violation",
    "wrong_tool",
    "wrong_interpretation",
    "timeout",
]

DEFAULT_HELD_OUT_CATEGORY = "unexpected_output"


@dataclass
class LabeledExecutionExample:
    """Represents a labeled agent execution run for evaluation.
    
    Labels (responsible_step, failure_type, verifier_result) are strictly
    segregated from observable trace features to prevent target leakage.
    """
    run_id: str
    scenario_id: str
    user_request: str
    failure_type: str  # "none", "budget_violation", "wrong_tool", etc.
    injection_id: Optional[str]
    # Ground truth label for evaluation only (never used as feature)
    responsible_step_id: Optional[int]
    responsible_step_type: Optional[str]
    is_failure: bool
    verifier_passed: bool
    verifier_reason: str
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "user_request": self.user_request,
            "failure_type": self.failure_type,
            "injection_id": self.injection_id,
            "responsible_step_id": self.responsible_step_id,
            "responsible_step_type": self.responsible_step_type,
            "is_failure": self.is_failure,
            "verifier_passed": self.verifier_passed,
            "verifier_reason": self.verifier_reason,
        }


@dataclass
class DatasetSplits:
    train: List[LabeledExecutionExample]
    validation: List[LabeledExecutionExample]
    test: List[LabeledExecutionExample]
    held_out_test: List[LabeledExecutionExample]
    held_out_category: str

    @property
    def train_run_ids(self) -> Set[str]:
        return {e.run_id for e in self.train}

    @property
    def validation_run_ids(self) -> Set[str]:
        return {e.run_id for e in self.validation}

    @property
    def test_run_ids(self) -> Set[str]:
        return {e.run_id for e in self.test}


class DatasetManager:
    """Manages creation, loading, and leakage-safe splitting of evaluation datasets."""

    def __init__(
        self,
        repository: TraceRepository,
        verifier: Optional[IndependentVerifier] = None,
        held_out_category: str = DEFAULT_HELD_OUT_CATEGORY,
    ) -> None:
        self.repo = repository
        self.verifier = verifier or IndependentVerifier()
        self.held_out_category = held_out_category

    def generate_benchmark_dataset(
        self,
        target_success_count: int = 50,
        target_failure_count: int = 100,
        random_seed: int = 42,
    ) -> List[LabeledExecutionExample]:
        """Generates diverse controlled runs at practical prototype scale (120-180 runs).
        
        Uses variations of queries and failure injection modes, validating each run
        against the independent verifier.
        """
        random.seed(random_seed)

        query_templates = [
            ("Find a laptop under ₹{b} for student coursework", [35000, 40000, 45000, 50000]),
            ("Recommend a laptop within ₹{b} suitable for programming and gaming", [65000, 70000, 75000, 80000]),
            ("Need a lightweight ultrabook under ₹{b} with 16GB RAM", [60000, 65000, 75000, 80000]),
            ("Looking for high-performance laptop within ₹{b}", [70000, 85000, 90000, 95000]),
            ("Best laptop under ₹{b} for college assignments", [38000, 42000, 48000, 50000]),
        ]

        examples: List[LabeledExecutionExample] = []

        # 1. Generate Successful runs
        logger.info("Generating %d successful runs...", target_success_count)
        success_created = 0
        while success_created < target_success_count:
            tmpl, budgets = random.choice(query_templates)
            budget = random.choice(budgets)
            request = tmpl.format(b=budget)

            run_id = f"bench-succ-{uuid.uuid4().hex[:6]}"
            recorder = ExecutionRecorder(run_id=run_id, repository=self.repo)
            agent = LaptopAgent(sinks=[recorder.record])

            agent_result = agent.run(request, failure_mode=None)
            trace = self.repo.get_run_trace(run_id)

            if trace:
                v_res = self.verifier.verify(trace)
                if v_res.passed:
                    examples.append(
                        LabeledExecutionExample(
                            run_id=run_id,
                            scenario_id=f"scen-succ-{success_created + 1}",
                            user_request=request,
                            failure_type="none",
                            injection_id=None,
                            responsible_step_id=None,
                            responsible_step_type=None,
                            is_failure=False,
                            verifier_passed=True,
                            verifier_reason=v_res.reason,
                            trace=trace,
                        )
                    )
                    success_created += 1

        # 2. Generate Failed runs across all failure modes (including held-out)
        all_failure_modes = KNOWN_FAILURE_CATEGORIES + [self.held_out_category]
        logger.info("Generating %d failed runs across %s...", target_failure_count, all_failure_modes)

        failed_created = 0
        while failed_created < target_failure_count:
            f_mode = all_failure_modes[failed_created % len(all_failure_modes)]
            tmpl, budgets = random.choice(query_templates)
            budget = random.choice(budgets)
            request = tmpl.format(b=budget)

            run_id = f"bench-fail-{uuid.uuid4().hex[:6]}"
            recorder = ExecutionRecorder(run_id=run_id, repository=self.repo)
            agent = LaptopAgent(sinks=[recorder.record])

            agent_result = agent.run(request, failure_mode=f_mode)
            trace = self.repo.get_run_trace(run_id)

            if trace:
                v_res = self.verifier.verify(trace)
                # Only keep as valid failure case if independent verifier confirms failure
                if not v_res.passed:
                    resp_id, resp_type = self._determine_ground_truth_step(trace, f_mode)
                    examples.append(
                        LabeledExecutionExample(
                            run_id=run_id,
                            scenario_id=f"scen-fail-{failed_created + 1}",
                            user_request=request,
                            failure_type=f_mode,
                            injection_id=f"inj-{uuid.uuid4().hex[:4]}",
                            responsible_step_id=resp_id,
                            responsible_step_type=resp_type,
                            is_failure=True,
                            verifier_passed=False,
                            verifier_reason=v_res.reason,
                            trace=trace,
                        )
                    )
                    failed_created += 1

        logger.info("Generated %d total evaluation runs (%d success, %d failed).", len(examples), success_created, failed_created)
        return examples

    @staticmethod
    def _determine_ground_truth_step(trace: Dict[str, Any], failure_type: str) -> Tuple[Optional[int], Optional[str]]:
        """Identifies ground truth responsible step based on controlled injection specification.
        
        This is metadata for evaluation scoring only, never given to localizers.
        """
        steps = trace.get("steps", [])
        if not steps:
            return None, None

        if failure_type == "wrong_tool":
            # Responsible step is the premature tool selection
            for s in steps:
                if s.get("step_type") in ("tool_selected", "tool_called"):
                    return s.get("step_id"), s.get("step_type")
        elif failure_type in ("timeout", "tool_error"):
            for s in steps:
                if s.get("step_type") == "tool_error" or s.get("status") in ("failed", "error"):
                    return s.get("step_id"), s.get("step_type")
        elif failure_type == "unexpected_output":
            for s in steps:
                out = s.get("output") or {}
                if isinstance(out, dict) and out.get("malformed"):
                    return s.get("step_id"), s.get("step_type")
        elif failure_type in ("budget_violation", "wrong_interpretation"):
            # The decision / final response generation step where violation occurred
            final_steps = [s for s in steps if s.get("step_type") in ("final_response_generated", "final_response")]
            if final_steps:
                return final_steps[-1].get("step_id"), final_steps[-1].get("step_type")

        # Default fallback to last step
        last = steps[-1]
        return last.get("step_id"), last.get("step_type")

    def split_dataset(
        self,
        examples: List[LabeledExecutionExample],
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        random_seed: int = 42,
    ) -> DatasetSplits:
        """Splits runs into Train, Validation, Test, and Held-Out Test sets.
        
        Strictly prevents data leakage by:
        1. Separating the held-out failure category into held_out_test only.
        2. Splitting known-category examples by run scenario groups.
        3. Enforcing that test examples never appear in train or validation sets.
        """
        random.seed(random_seed)

        # 1. Separate held-out category
        held_out_examples = [e for e in examples if e.failure_type == self.held_out_category]
        known_examples = [e for e in examples if e.failure_type != self.held_out_category]

        # 2. Shuffle known examples deterministically
        shuffled = copy.copy(known_examples)
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_split = shuffled[:n_train]
        val_split = shuffled[n_train : n_train + n_val]
        test_split = shuffled[n_train + n_val :]

        logger.info(
            "Dataset split complete: Train=%d, Val=%d, Test=%d, HeldOut(%s)=%d",
            len(train_split),
            len(val_split),
            len(test_split),
            self.held_out_category,
            len(held_out_examples),
        )

        return DatasetSplits(
            train=train_split,
            validation=val_split,
            test=test_split,
            held_out_test=held_out_examples,
            held_out_category=self.held_out_category,
        )
