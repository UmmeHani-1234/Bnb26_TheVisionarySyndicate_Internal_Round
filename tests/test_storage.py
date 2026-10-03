"""Phase 3 Tests: Trace Storage, Trace Retrieval, and API Endpoints.

Covers:
1. Creating a run
2. Recording an event
3. Recording multiple ordered events
4. Retrieving a run
5. Retrieving a complete trace
6. Handling an invalid run ID
7. Handling a failed/incomplete execution
8. End-to-end agent execution with automatic trace storage (two runs test)
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from storage.database import Base, SessionLocal, engine, init_db
from storage.models import Run, ExecutionStep
from storage.repository import TraceRepository
from recorder.recorder import ExecutionRecorder
from server import app
from agent.agent import LaptopAgent
from agent.events import EventType
from tests.test_agent import ScriptedChatModel, happy_path_script, tool_call
from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def setup_test_db():
    """Recreate tables before each test for total isolation."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


@pytest.fixture
def repo():
    return TraceRepository()


@pytest.fixture
def client():
    return TestClient(app)


# --------------------------------------------------------------------------- 1. Unit Tests


def test_1_create_run(repo: TraceRepository):
    """Test 1: Creating a run."""
    run = repo.create_run(
        run_id="run-001",
        user_request="Find a laptop under ₹60,000",
        status="running",
    )
    assert run.run_id == "run-001"
    assert run.user_request == "Find a laptop under ₹60,000"
    assert run.status == "running"
    assert run.started_at is not None
    assert run.ended_at is None


def test_2_record_an_event(repo: TraceRepository):
    """Test 2: Recording an event."""
    repo.create_run(run_id="run-002", user_request="Testing single step")
    step = repo.add_step(
        run_id="run-002",
        event_dict={
            "step_id": 1,
            "event_id": "evt-001",
            "step_type": "request_understanding",
            "input": {"query": "sample"},
            "output": {"parsed": True},
            "status": "success",
            "latency": 0.05,
        },
    )
    assert step.step_id == 1
    assert step.run_id == "run-002"
    assert step.tool_name is None
    assert step.status == "success"
    assert step.latency == 0.05


def test_3_record_multiple_ordered_events(repo: TraceRepository):
    """Test 3: Recording multiple ordered events."""
    repo.create_run(run_id="run-003", user_request="Multi-step ordering")

    steps_data = [
        {"step_id": 1, "step_type": "request_understanding"},
        {"step_id": 2, "step_type": "planning"},
        {"step_id": 3, "step_type": "tool_execution", "tool_name": "search_products"},
        {"step_id": 4, "step_type": "tool_execution", "tool_name": "check_specifications"},
        {"step_id": 5, "step_type": "tool_execution", "tool_name": "calculate_budget"},
        {"step_id": 6, "step_type": "decision"},
        {"step_id": 7, "step_type": "final_response"},
    ]

    for s in steps_data:
        repo.add_step("run-003", s)

    trace = repo.get_run_trace("run-003")
    assert trace is not None
    assert len(trace["steps"]) == 7

    # Verify strict execution ordering
    retrieved_step_ids = [step["step_id"] for step in trace["steps"]]
    assert retrieved_step_ids == [1, 2, 3, 4, 5, 6, 7]

    types = [step["step_type"] for step in trace["steps"]]
    assert types == [
        "request_understanding",
        "planning",
        "tool_execution",
        "tool_execution",
        "tool_execution",
        "decision",
        "final_response",
    ]

    tools = [step["tool_name"] for step in trace["steps"] if step["tool_name"]]
    assert tools == ["search_products", "check_specifications", "calculate_budget"]


def test_4_retrieve_a_run(repo: TraceRepository):
    """Test 4: Retrieving a run."""
    repo.create_run(run_id="run-004", user_request="Retrieve test", status="success")
    run = repo.get_run("run-004")
    assert run is not None
    assert run.run_id == "run-004"
    assert run.status == "success"


def test_5_retrieve_complete_trace(repo: TraceRepository):
    """Test 5: Retrieving a complete trace with all details."""
    repo.create_run(run_id="run-005", user_request="Complete trace query")
    repo.add_step(
        "run-005",
        {
            "step_id": 1,
            "step_type": "tool_execution",
            "tool_name": "search_products",
            "input": {"max_price": 70000},
            "output": {"count": 4},
            "status": "success",
            "metadata": {"duration_ms": 1240},
        },
    )
    repo.update_run("run-005", status="success", final_output="Here is the recommendation.")

    trace = repo.get_run_trace("run-005")
    assert trace is not None
    assert trace["run_id"] == "run-005"
    assert trace["status"] == "success"
    assert trace["user_request"] == "Complete trace query"
    assert trace["final_output"] == "Here is the recommendation."
    assert len(trace["steps"]) == 1
    assert trace["steps"][0]["tool_name"] == "search_products"
    assert trace["steps"][0]["latency"] == 1.24  # 1240 ms -> 1.24s


def test_6_handle_invalid_run_id(repo: TraceRepository, client: TestClient):
    """Test 6: Handling an invalid run ID."""
    # Repository level
    assert repo.get_run("non-existent-run") is None
    assert repo.get_run_trace("non-existent-run") is None

    with pytest.raises(ValueError, match="not found"):
        repo.add_step("non-existent-run", {"step_id": 1, "step_type": "test"})

    # API level
    response = client.get("/runs/non-existent-run")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_7_handle_failed_incomplete_execution(repo: TraceRepository):
    """Test 7: Handling a failed/incomplete execution (partial trace preservation)."""
    recorder = ExecutionRecorder(run_id="run-failed-001", repository=repo)

    # Agent runs into error mid-execution
    script = [
        tool_call("calculate_budget", {"product_name": "Imaginary Laptop", "budget": 50000}, "c1"),
        AIMessage(content="Laptop not found"),
    ]
    agent = LaptopAgent(llm=ScriptedChatModel(responses=script), sinks=[recorder.record])
    result = agent.run("Find Imaginary Laptop")

    trace = repo.get_run_trace("run-failed-001")
    assert trace is not None
    assert len(trace["steps"]) >= 4  # Agent started, request received, tool selected, tool error

    # Verify tool error step was captured and preserved
    error_steps = [s for s in trace["steps"] if s["status"] == "failed" or s["step_type"] == "tool_error"]
    assert len(error_steps) >= 1
    assert error_steps[0]["output"]["error_type"] == "product_not_found"


# --------------------------------------------------------------------------- 2. API Endpoints Tests


def test_api_runs_crud(client: TestClient):
    """Verify POST /runs, POST /runs/{id}/events, GET /runs, GET /runs/{id}."""
    # 1. POST /runs
    res = client.post("/runs", json={"run_id": "api-run-1", "user_request": "API test query"})
    assert res.status_code == 201
    assert res.json()["run_id"] == "api-run-1"

    # 2. POST /runs/{id}/events
    res_ev = client.post(
        "/runs/api-run-1/events",
        json={
            "step_type": "tool_execution",
            "tool_name": "search_products",
            "input": {"max_price": 65000},
            "output": {"count": 2},
            "status": "success",
            "latency": 0.42,
        },
    )
    assert res_ev.status_code == 201
    assert res_ev.json()["tool_name"] == "search_products"

    # 3. GET /runs
    res_list = client.get("/runs")
    assert res_list.status_code == 200
    runs = res_list.json()
    assert any(r["run_id"] == "api-run-1" for r in runs)

    # 4. GET /runs/{id}
    res_trace = client.get("/runs/api-run-1")
    assert res_trace.status_code == 200
    data = res_trace.json()
    assert data["run_id"] == "api-run-1"
    assert len(data["steps"]) == 1
    assert data["steps"][0]["tool_name"] == "search_products"


# --------------------------------------------------------------------------- 3. End-to-End Success Condition


def test_agent_two_separate_runs_stored_completely(repo: TraceRepository):
    """Success Condition: Run agent twice with two different requests and confirm

    that both executions are stored as separate runs with complete ordered execution steps.
    """
    # Run 1: Gaming request
    recorder_1 = ExecutionRecorder(run_id="run-gaming-101", repository=repo)
    agent_1 = LaptopAgent(llm=ScriptedChatModel(responses=happy_path_script()), sinks=[recorder_1.record])
    res_1 = agent_1.run("Find a laptop under ₹70,000 for gaming.")
    assert res_1.status == "success"

    # Run 2: Budget request
    recorder_2 = ExecutionRecorder(run_id="run-budget-102", repository=repo)
    agent_2 = LaptopAgent(llm=ScriptedChatModel(responses=happy_path_script()), sinks=[recorder_2.record])
    res_2 = agent_2.run("Find a budget laptop for college around ₹50,000.")
    assert res_2.status == "success"

    # Verify Run 1 in storage
    trace_1 = repo.get_run_trace("run-gaming-101")
    assert trace_1 is not None
    assert trace_1["run_id"] == "run-gaming-101"
    assert trace_1["status"] == "success"
    assert "gaming" in trace_1["user_request"].lower()
    assert len(trace_1["steps"]) >= 10
    step_ids_1 = [s["step_id"] for s in trace_1["steps"]]
    assert step_ids_1 == list(range(1, len(trace_1["steps"]) + 1))

    # Verify Run 2 in storage
    trace_2 = repo.get_run_trace("run-budget-102")
    assert trace_2 is not None
    assert trace_2["run_id"] == "run-budget-102"
    assert trace_2["status"] == "success"
    assert "college" in trace_2["user_request"].lower()
    assert len(trace_2["steps"]) >= 10
    step_ids_2 = [s["step_id"] for s in trace_2["steps"]]
    assert step_ids_2 == list(range(1, len(trace_2["steps"]) + 1))

    # Confirm both runs coexist separately in storage
    all_runs = repo.list_runs()
    run_ids = [r["run_id"] for r in all_runs]
    assert "run-gaming-101" in run_ids
    assert "run-budget-102" in run_ids


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
