"""Comprehensive Stage 6 test suite for Black Box.

Validates:
1. Reference Trace Selection & Isolation
2. 6 Structured Observable Signals & Feature Extraction
3. Improved Weighted Rule-Based Localizer (Ranking, Tie-Breaking, Validation Tuning)
4. Random Forest Classifier (Training, Ranking, Untrained Handling, No Leakage)
5. Dataset Generation & Leakage-Safe Splitting (Held-Out Category Separation)
6. Independent Verifier (Deterministic Outcome Validation)
7. Trace Comparator & 3-Way Alignment (Recovery Verification)
8. Evaluation Metrics & Baselines (Top-1, Top-3, MRR, Random Baseline, Category Breakdown)
"""

import pytest
from failure_intelligence.comparison.comparator import TraceComparator
from failure_intelligence.evaluation.dataset import DatasetManager, DatasetSplits, LabeledExecutionExample
from failure_intelligence.evaluation.evaluator import BenchmarkEvaluator
from failure_intelligence.evaluation.metrics import (
    CategoryEvaluationSummary,
    MethodEvaluationSummary,
    compute_random_baseline,
)
from failure_intelligence.evaluation.verifier import IndependentVerifier
from failure_intelligence.features.extractor import FeatureExtractor, StepFeatureRecord
from failure_intelligence.features.reference import ReferenceSelector
from failure_intelligence.features.signals import (
    compute_downstream_impact,
    compute_error_status,
    compute_latency_anomaly,
    compute_output_divergence,
    compute_state_divergence,
    compute_tool_mismatch,
)
from failure_intelligence.localization.random_forest import RandomForestLocalizer
from failure_intelligence.localization.rule_based import WeightedRuleLocalizer
from storage.database import SessionLocal, init_db
from storage.repository import TraceRepository


# --------------------------------------------------------------------------- Fixtures
@pytest.fixture
def repo():
    init_db()
    db = SessionLocal()
    r = TraceRepository(db=db)
    yield r
    db.close()


@pytest.fixture
def verifier():
    return IndependentVerifier()


@pytest.fixture
def mock_success_trace():
    return {
        "run_id": "succ-001",
        "user_request": "Laptop under 50000 for college",
        "status": "success",
        "steps": [
            {
                "step_id": 1,
                "step_type": "request_understanding",
                "tool_name": None,
                "status": "success",
                "latency": 0.5,
                "state_before": {},
                "state_after": {"budget": 50000},
                "output": {"intent": "find_laptop", "budget": 50000},
            },
            {
                "step_id": 2,
                "step_type": "tool_selection",
                "tool_name": "search_products",
                "status": "success",
                "latency": 0.4,
                "state_before": {"budget": 50000},
                "state_after": {"selected_tool": "search_products"},
                "output": {"tool": "search_products"},
            },
            {
                "step_id": 3,
                "step_type": "tool_execution",
                "tool_name": "search_products",
                "status": "success",
                "latency": 0.8,
                "state_before": {"selected_tool": "search_products"},
                "state_after": {"search_completed": True},
                "output": {"count": 3, "products": [{"name": "Ideapad", "price": 42000}]},
            },
            {
                "step_id": 4,
                "step_type": "final_response_generated",
                "tool_name": None,
                "status": "success",
                "latency": 0.6,
                "state_before": {"search_completed": True},
                "state_after": {"done": True},
                "output": {"response": "I recommend Ideapad for ₹42,000.", "price": 42000},
            },
        ],
    }


@pytest.fixture
def mock_failed_trace():
    return {
        "run_id": "fail-001",
        "user_request": "Laptop under 50000 for college",
        "status": "failed",
        "steps": [
            {
                "step_id": 1,
                "step_type": "request_understanding",
                "tool_name": None,
                "status": "success",
                "latency": 0.5,
                "state_before": {},
                "state_after": {"budget": 50000},
                "output": {"intent": "find_laptop", "budget": 50000},
            },
            {
                "step_id": 2,
                "step_type": "tool_selection",
                "tool_name": "calculate_budget",  # Mismatch!
                "status": "success",
                "latency": 0.4,
                "state_before": {"budget": 50000},
                "state_after": {"selected_tool": "calculate_budget"},
                "output": {"tool": "calculate_budget"},
            },
            {
                "step_id": 3,
                "step_type": "tool_execution",
                "tool_name": "calculate_budget",
                "status": "failed",  # Error status!
                "latency": 5.2,  # Latency anomaly!
                "state_before": {"selected_tool": "calculate_budget"},
                "state_after": {"error": "Missing parameter"},
                "output": {"error_type": "MissingParameterError"},
            },
            {
                "step_id": 4,
                "step_type": "final_response_generated",
                "tool_name": None,
                "status": "failed",
                "latency": 0.4,
                "state_before": {"error": "Missing parameter"},
                "state_after": {"done": False},
                "output": {"response": "Error occurred during budget calculation."},
            },
        ],
    }


# --------------------------------------------------------------------------- 1. Reference Selection Tests
def test_reference_selector_same_input(mock_success_trace, mock_failed_trace):
    selector = ReferenceSelector()
    candidates = [
        mock_success_trace,
        {
            "run_id": "succ-002",
            "user_request": "Completely different query about phones",
            "status": "success",
            "steps": [],
        },
    ]

    selected = selector.select_reference(mock_failed_trace, candidates)
    assert selected is not None
    assert selected["run_id"] == "succ-001"


def test_reference_selector_strict_split_isolation(mock_success_trace, mock_failed_trace):
    selector = ReferenceSelector()
    # If allowed_run_ids excludes succ-001, it must NOT be selected
    selected = selector.select_reference(
        mock_failed_trace,
        candidate_traces=[mock_success_trace],
        allowed_run_ids={"succ-999"},
    )
    assert selected is None


# --------------------------------------------------------------------------- 2. Feature Extraction & Signals Tests
def test_signal_computations():
    # 1. Tool mismatch
    assert compute_tool_mismatch({"tool_name": "search_products"}, {"tool_name": "search_products"}) == 0.0
    assert compute_tool_mismatch({"tool_name": "calculate_budget"}, {"tool_name": "search_products"}) == 1.0
    assert compute_tool_mismatch({"tool_name": None}, {"tool_name": None}) == 0.0

    # 2. Output divergence
    assert compute_output_divergence({"output": {"price": 42000}}, {"output": {"price": 42000}}) == 0.0
    assert compute_output_divergence({"output": {"price": 85000}}, {"output": {"price": 42000}}) > 0.0

    # 3. State divergence
    assert compute_state_divergence({"input": {"b": 50}}, {"input": {"b": 50}}) == 0.0
    assert compute_state_divergence({"input": {"b": 50}}, {"input": {"b": 80}}) > 0.0

    # 4. Latency anomaly
    assert compute_latency_anomaly({"latency": 0.8}, {"latency": 0.8}) == 0.0
    assert compute_latency_anomaly({"latency": 5.0}, {"latency": 0.8}) > 0.5

    # 5. Error status
    assert compute_error_status({"status": "success"}) == 0.0
    assert compute_error_status({"status": "failed"}) == 1.0
    assert compute_error_status({"status": "error"}) == 1.0

    # 6. Downstream impact
    steps = [
        {"step_id": 1, "status": "success"},
        {"step_id": 2, "status": "failed"},
        {"step_id": 3, "status": "failed"},
    ]
    impact = compute_downstream_impact(2, steps)
    assert impact == 1.0


def test_feature_extractor_produces_structured_inspectable_records(mock_failed_trace, mock_success_trace):
    extractor = FeatureExtractor()
    records = extractor.extract_features(mock_failed_trace, mock_success_trace)

    assert len(records) == 4
    for r in records:
        assert isinstance(r, StepFeatureRecord)
        assert len(r.to_vector()) == 6
        assert 0.0 <= r.features["tool_mismatch"] <= 1.0
        assert 0.0 <= r.features["output_divergence"] <= 1.0
        assert 0.0 <= r.features["state_divergence"] <= 1.0
        assert 0.0 <= r.features["downstream_impact"] <= 1.0
        assert 0.0 <= r.features["latency_anomaly"] <= 1.0
        assert 0.0 <= r.features["error_status"] <= 1.0


# --------------------------------------------------------------------------- 3. Improved Rule-Based Localizer Tests
def test_rule_localizer_ranking_and_tie_breaking(mock_failed_trace, mock_success_trace):
    localizer = WeightedRuleLocalizer()
    res = localizer.localize(mock_failed_trace, mock_success_trace)

    assert res.method == "rule_based"
    assert len(res.ranked_steps) == 4
    # Top-1 exists and is ranked highest
    assert res.top_1 is not None
    assert len(res.top_3) == 3

    # Ranking must be in non-increasing suspicion score order
    scores = [s.suspicion_score for s in res.ranked_steps]
    assert scores == sorted(scores, reverse=True)


def test_rule_localizer_validation_tuning(mock_failed_trace, mock_success_trace):
    localizer = WeightedRuleLocalizer()
    val_dataset = [(mock_failed_trace, mock_success_trace, 3)]
    initial_weights = dict(localizer.weights)
    tuned = localizer.tune_weights_on_validation(val_dataset)

    assert sum(tuned.values()) == pytest.approx(1.0, rel=1e-3)
    assert isinstance(tuned, dict)


# --------------------------------------------------------------------------- 4. Random Forest Localizer Tests
def test_random_forest_localizer_untrained_handling(mock_failed_trace):
    rf = RandomForestLocalizer()
    res = rf.localize(mock_failed_trace)
    assert res.method == "random_forest"
    # When untrained, should not crash, returns 0.0 suspicion scores
    assert all(s.suspicion_score == 0.0 for s in res.ranked_steps)


def test_random_forest_localizer_training_and_ranking(mock_failed_trace, mock_success_trace):
    rf = RandomForestLocalizer(n_estimators=10)

    # Synthetic training examples without leakage
    train_examples = [
        LabeledExecutionExample(
            run_id=f"run-{i}",
            scenario_id="scen-1",
            user_request="req",
            failure_type="wrong_tool",
            injection_id="inj-1",
            responsible_step_id=3,
            responsible_step_type="tool_execution",
            is_failure=True,
            verifier_passed=False,
            verifier_reason="error",
            trace=mock_failed_trace,
        )
        for i in range(12)
    ]

    res_train = rf.train_on_examples(train_examples, {f"run-{i}": mock_success_trace for i in range(12)})
    assert res_train["status"] == "trained"
    assert rf.is_trained is True

    # Prediction
    pred = rf.localize(mock_failed_trace, mock_success_trace)
    assert pred.method == "random_forest"
    assert len(pred.ranked_steps) == 4
    assert pred.top_1 is not None
    assert pred.top_1.suspicion_score >= pred.ranked_steps[-1].suspicion_score


# --------------------------------------------------------------------------- 5. Dataset Manager & Splitting Tests
def test_dataset_splitting_and_held_out_isolation():
    # Construct synthetic labeled examples
    examples = []
    # 20 known failure runs
    for i in range(20):
        examples.append(
            LabeledExecutionExample(
                run_id=f"k-{i}",
                scenario_id=f"s-{i}",
                user_request="req",
                failure_type="budget_violation",
                injection_id=f"inj-{i}",
                responsible_step_id=4,
                responsible_step_type="final_response",
                is_failure=True,
                verifier_passed=False,
                verifier_reason="over budget",
                trace={"run_id": f"k-{i}", "steps": []},
            )
        )
    # 10 held-out failure runs
    for i in range(10):
        examples.append(
            LabeledExecutionExample(
                run_id=f"h-{i}",
                scenario_id=f"sh-{i}",
                user_request="req",
                failure_type="unexpected_output",
                injection_id=f"inj-h-{i}",
                responsible_step_id=3,
                responsible_step_type="tool_execution",
                is_failure=True,
                verifier_passed=False,
                verifier_reason="corrupted format",
                trace={"run_id": f"h-{i}", "steps": []},
            )
        )

    splits = DatasetManager(repository=None, held_out_category="unexpected_output").split_dataset(examples)

    # 1. Held-out category must ONLY be present in held_out_test
    assert len(splits.held_out_test) == 10
    assert all(e.failure_type == "unexpected_output" for e in splits.held_out_test)
    assert not any(e.failure_type == "unexpected_output" for e in splits.train)
    assert not any(e.failure_type == "unexpected_output" for e in splits.validation)
    assert not any(e.failure_type == "unexpected_output" for e in splits.test)

    # 2. No run ID leakage between splits
    train_ids = set(splits.train_run_ids)
    val_ids = set(splits.validation_run_ids)
    test_ids = set(splits.test_run_ids)
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


# --------------------------------------------------------------------------- 6. Independent Verifier Tests
def test_independent_verifier_validates_budget_and_status(mock_success_trace, mock_failed_trace, verifier):
    v_succ = verifier.verify(mock_success_trace)
    assert v_succ.passed is True

    v_fail = verifier.verify(mock_failed_trace)
    assert v_fail.passed is False
    assert "error" in v_fail.reason.lower()


def test_independent_verifier_detects_over_budget(verifier):
    over_budget_trace = {
        "run_id": "ob-1",
        "user_request": "Laptop under 40000",
        "steps": [
            {
                "step_id": 1,
                "step_type": "final_response_generated",
                "status": "success",
                "output": {"price": 75000, "product": "Predator Helios"},
            }
        ],
    }
    res = verifier.verify(over_budget_trace)
    assert res.passed is False
    assert "budget" in res.reason.lower() or "exceeds" in res.reason.lower()


# --------------------------------------------------------------------------- 7. Trace Comparator Tests
def test_trace_comparator_3_way_alignment_and_recovery(mock_failed_trace, mock_success_trace, verifier):
    comparator = TraceComparator(verifier=verifier)

    # Alternative run where step 3 was corrected and succeeded
    alt_trace = {
        "run_id": "alt-001",
        "user_request": "Laptop under 50000 for college",
        "status": "success",
        "steps": [
            {
                "step_id": 1,
                "step_type": "request_understanding",
                "status": "success",
                "output": {"intent": "find_laptop", "budget": 50000},
            },
            {
                "step_id": 2,
                "step_type": "tool_selection",
                "tool_name": "search_products",
                "status": "success",
                "output": {"tool": "search_products"},
            },
            {
                "step_id": 3,
                "step_type": "tool_execution",
                "tool_name": "search_products",
                "status": "success",
                "output": {"count": 2},
            },
            {
                "step_id": 4,
                "step_type": "final_response_generated",
                "status": "success",
                "output": {"response": "Found laptop for ₹45000", "price": 45000},
            },
        ],
    }

    comp = comparator.compare_traces(mock_failed_trace, alt_trace, mock_success_trace)

    assert comp.original_run_id == "fail-001"
    assert comp.alternative_run_id == "alt-001"
    assert comp.reference_run_id == "succ-001"

    # Step changes must be detected
    assert len(comp.changed_steps) > 0
    # Recovery must be confirmed because original failed and alternative passed
    assert comp.recovered is True
    assert comp.original_verifier["passed"] is False
    assert comp.alternative_verifier["passed"] is True


# --------------------------------------------------------------------------- 8. Evaluation Metrics & Baselines Tests
def test_random_baseline_calculation():
    base = compute_random_baseline(8.0)
    assert base.method == "random_baseline"
    assert base.top1_accuracy == pytest.approx(0.125, rel=1e-3)
    assert base.top3_accuracy == pytest.approx(0.375, rel=1e-3)
    assert 0.0 < base.mrr < 1.0


def test_evaluator_insufficient_data_handling():
    evaluator = BenchmarkEvaluator()
    empty_splits = DatasetSplits(
        train=[],
        validation=[],
        test=[],
        held_out_test=[],
        held_out_category="unexpected_output",
    )
    res = evaluator.run_full_evaluation(empty_splits)
    assert res["status"] == "evaluation_not_ready"
    assert "insufficient" in res["reason"].lower()
