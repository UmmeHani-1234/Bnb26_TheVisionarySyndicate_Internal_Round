"""Comprehensive unit and integration tests for Stage 4: Failure Intelligence."""

import os
import pytest
from fastapi.testclient import TestClient

from agent.agent import LaptopAgent
from intelligence.base import DiagnosisResult, EvidenceItem, StepSuspicion
from intelligence.benchmark import BenchmarkRunner, BENCHMARK_SCENARIOS
from intelligence.rules import RuleBasedLocalizer
from recorder.recorder import ExecutionRecorder
from server import app
from storage.database import SessionLocal, init_db
from storage.repository import TraceRepository

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_db()


@pytest.fixture
def repo():
    return TraceRepository()


@pytest.fixture
def localizer():
    return RuleBasedLocalizer()


# --------------------------------------------------------------------------- 1. Rule-Based Localizer Unit Tests


def test_normal_successful_trace_has_no_failure(localizer):
    trace = {
        "run_id": "test-success-01",
        "status": "success",
        "user_request": "Find a laptop under 70000",
        "final_output": "I recommend Lenovo IdeaPad Slim 5 at ₹58,990.",
        "steps": [
            {"step_id": 1, "step_type": "agent_started", "status": "success"},
            {"step_id": 2, "step_type": "user_request_received", "status": "success", "input": {"request": "Find a laptop under 70000"}},
            {"step_id": 3, "step_type": "tool_selected", "tool_name": "search_products", "status": "success"},
            {"step_id": 4, "step_type": "tool_called", "tool_name": "search_products", "status": "running"},
            {"step_id": 5, "step_type": "tool_completed", "tool_name": "search_products", "status": "success", "output": {"products": [{"name": "Lenovo IdeaPad Slim 5", "price": 58990}]}},
            {"step_id": 6, "step_type": "final_response_generated", "status": "success", "output": {"response": "I recommend Lenovo IdeaPad Slim 5 at ₹58,990."}},
        ]
    }
    diag = localizer.diagnose(trace)
    assert not diag.failure_detected
    assert diag.failure_type == "none"
    assert diag.likely_responsible_step is None


def test_budget_violation_localization(localizer):
    trace = {
        "run_id": "test-budget-violation-01",
        "status": "success",
        "user_request": "Find a laptop strictly under ₹70,000",
        "final_output": "I recommend Apple MacBook Air M2 for ₹84,990.",
        "steps": [
            {"step_id": 1, "step_type": "agent_started", "status": "success"},
            {"step_id": 2, "step_type": "user_request_received", "status": "success"},
            {"step_id": 3, "step_type": "tool_completed", "tool_name": "search_products", "output": {"products": [{"name": "Apple MacBook Air M2", "price": 84990}]}},
            {"step_id": 4, "step_type": "tool_completed", "tool_name": "calculate_budget", "output": {"within_budget": False, "difference": -14990}},
            {"step_id": 5, "step_type": "final_response_generated", "status": "success", "output": {"response": "I recommend Apple MacBook Air M2 for ₹84,990."}},
        ]
    }
    diag = localizer.diagnose(trace)
    assert diag.failure_detected
    assert diag.failure_type == "budget_violation"
    assert diag.likely_responsible_step is not None
    assert diag.likely_responsible_step.step_id == 5
    assert diag.likely_responsible_step.score >= 0.8
    # Evidence must contain concrete values
    descriptions = " ".join(e.description for e in diag.evidence)
    assert "70,000" in descriptions
    assert "84,990" in descriptions


def test_tool_timeout_localization(localizer):
    trace = {
        "run_id": "test-timeout-01",
        "status": "failed",
        "user_request": "Find laptop under 60k",
        "final_output": "Error: Timeout",
        "steps": [
            {"step_id": 1, "step_type": "agent_started", "status": "success"},
            {"step_id": 2, "step_type": "tool_selected", "tool_name": "search_products", "status": "success"},
            {"step_id": 3, "step_type": "tool_called", "tool_name": "search_products", "status": "running"},
            {"step_id": 4, "step_type": "tool_error", "tool_name": "search_products", "status": "failed", "output": {"status": "error", "error_type": "timeout", "message": "Tool execution timed out after 30000ms"}},
            {"step_id": 5, "step_type": "agent_error", "status": "failed", "output": {"message": "Timeout"}},
        ]
    }
    diag = localizer.diagnose(trace)
    assert diag.failure_detected
    assert diag.failure_type == "timeout"
    assert diag.likely_responsible_step.step_id == 4
    assert diag.likely_responsible_step.score >= 0.9


def test_unexpected_tool_output_localization(localizer):
    trace = {
        "run_id": "test-unexpected-01",
        "status": "success",
        "user_request": "Find ultrabook under 60000",
        "final_output": "I found a laptop.",
        "steps": [
            {"step_id": 1, "step_type": "agent_started", "status": "success"},
            {"step_id": 2, "step_type": "tool_completed", "tool_name": "search_products", "status": "success", "output": {"malformed": True, "price": None}},
            {"step_id": 3, "step_type": "final_response_generated", "status": "success"},
        ]
    }
    diag = localizer.diagnose(trace)
    assert diag.failure_detected
    assert diag.failure_type == "unexpected_tool_output"
    assert diag.likely_responsible_step.step_id == 2
    assert "null" in diag.evidence[0].description.lower() or "price" in diag.evidence[0].description.lower()


def test_result_interpretation_mismatch(localizer):
    trace = {
        "run_id": "test-mismatch-01",
        "status": "success",
        "user_request": "Find gaming laptop under 100000",
        "final_output": "I recommend HP Victus 15 at ₹85,000.",
        "steps": [
            {"step_id": 1, "step_type": "agent_started", "status": "success"},
            {"step_id": 2, "step_type": "tool_completed", "tool_name": "search_products", "status": "success", "output": {"price": 64990, "product": "HP Victus 15"}},
            {"step_id": 3, "step_type": "final_response_generated", "status": "success", "output": {"response": "I recommend HP Victus 15 at ₹85,000."}},
        ]
    }
    diag = localizer.diagnose(trace)
    assert diag.failure_detected
    assert diag.failure_type == "wrong_interpretation"
    assert diag.likely_responsible_step.step_id == 3
    assert "64,990" in diag.evidence[0].description
    assert "85,000" in diag.evidence[0].description


def test_suspicion_scores_normalized_and_ranked(localizer):
    trace = {
        "run_id": "test-ranking-01",
        "status": "failed",
        "user_request": "Find laptop under 50000",
        "final_output": "",
        "steps": [
            {"step_id": 1, "step_type": "agent_started"},
            {"step_id": 2, "step_type": "tool_selected", "tool_name": "search_products"},
            {"step_id": 3, "step_type": "tool_error", "tool_name": "search_products", "status": "failed", "output": {"status": "error", "error_type": "timeout", "message": "timeout"}},
        ]
    }
    diag = localizer.diagnose(trace)
    for s in diag.top_suspected_steps:
        assert 0.0 <= s.score <= 1.0

    # Ensure ranking is sorted descending
    scores = [s.score for s in diag.top_suspected_steps]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- 2. API Integration Tests


def test_api_diagnosis_endpoint(repo):
    # Store a test trace
    run_id = "test-api-diag-run"
    repo.create_run(run_id=run_id, user_request="Find laptop under ₹70,000", status="success")
    repo.add_step(run_id, {"step_id": 1, "step_type": "agent_started", "status": "success"})
    repo.add_step(run_id, {"step_id": 2, "step_type": "final_response_generated", "status": "success", "output": {"response": "Recommending ₹84,990 laptop."}})
    repo.update_run(run_id, final_output="Recommending ₹84,990 laptop.")

    res = client.get(f"/runs/{run_id}/diagnosis")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == run_id
    assert data["failure_detected"] is True
    assert data["failure_type"] == "budget_violation"
    assert len(data["evidence"]) > 0
    assert len(data["top_suspected_steps"]) > 0


def test_api_diagnosis_not_found():
    res = client.get("/runs/non-existent-run-9999/diagnosis")
    assert res.status_code == 404


def test_api_evaluate_endpoint():
    res = client.post("/evaluate/diagnosis")
    assert res.status_code == 200
    data = res.json()
    assert data["total_runs"] >= 7
    assert data["top1_accuracy"] >= 80.0
    assert data["top3_accuracy"] >= 80.0
    assert len(data["details"]) >= 7


# --------------------------------------------------------------------------- 3. Benchmark Accuracy Test


def test_benchmark_runner_accuracy(repo):
    runner = BenchmarkRunner(repository=repo)
    metrics = runner.run_benchmark()
    assert metrics.top1_accuracy >= 80.0
    assert metrics.top3_accuracy >= 80.0
    assert metrics.total_runs == len(BENCHMARK_SCENARIOS)
