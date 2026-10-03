"""Stage 5 Tests: Checkpoint Manager, Replay Engine, Alternative Execution.

Covers:
1.  Checkpoint creation from a run
2.  Checkpoint persistence to DB
3.  Checkpoint retrieval by ID
4.  Checkpoint linked to correct run
5.  Checkpoint contains required observable state
6.  Valid checkpoint replay creates a new run
7.  Invalid checkpoint ID returns error
8.  Checkpoint belonging to another run is rejected
9.  Original run remains unchanged after replay
10. Replay events are recorded in the new run
11. Replay state is restored correctly
12. Valid override (max_budget) creates alternative run
13. Valid override (failure_mode=none) removes failure
14. Invalid override key is rejected
15. Alternative run is stored separately
16. Budget violation recovery via replay
17. Wrong tool recovery via replay
18. API: GET /runs/{id}/checkpoints
19. API: POST /runs/{id}/replay
20. API: replay with invalid checkpoint
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from storage.database import Base, SessionLocal, engine, init_db
from storage.models import Run, ExecutionStep
from storage.checkpoint_models import Checkpoint
from storage.repository import TraceRepository
from replay.checkpoint_manager import CheckpointManager, _build_observable_state
from replay.replay_engine import ReplayEngine
from recorder.recorder import ExecutionRecorder
from server import app
from agent.agent import LaptopAgent
from tests.test_agent import ScriptedChatModel, happy_path_script, tool_call
from langchain_core.messages import AIMessage


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def setup_test_db():
    """Recreate all tables (including checkpoints) before each test."""
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


@pytest.fixture
def cm(repo):
    return CheckpointManager(repository=repo)


@pytest.fixture
def engine_obj(repo):
    return ReplayEngine(repository=repo)


def _run_happy_agent(repo: TraceRepository, run_id: str = "run-test-001") -> dict:
    """Helper: run the happy-path agent, store results, return trace."""
    recorder = ExecutionRecorder(run_id=run_id, repository=repo)
    agent = LaptopAgent(llm=ScriptedChatModel(responses=happy_path_script()), sinks=[recorder.record])
    agent.run("Find a laptop under ₹70,000 suitable for programming and gaming.")
    return repo.get_run_trace(run_id)


def _run_failure_agent(
    repo: TraceRepository,
    failure_mode: str = "budget_violation",
    run_id: str = "run-fail-001",
) -> dict:
    """Helper: run an agent with a controlled failure injected."""
    recorder = ExecutionRecorder(run_id=run_id, repository=repo)
    agent = LaptopAgent(sinks=[recorder.record])
    agent.run("Find a laptop under ₹70,000", failure_mode=failure_mode)
    return repo.get_run_trace(run_id)


# --------------------------------------------------------------------------- 1-5: Checkpoint creation


def test_checkpoint_creation_returns_list(repo, cm):
    """Test that create_checkpoints_for_run returns a non-empty list."""
    trace = _run_happy_agent(repo)
    cps = cm.create_checkpoints_for_run(
        run_id=trace["run_id"],
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    assert isinstance(cps, list)
    assert len(cps) > 0, "Should create at least one checkpoint"


def test_checkpoint_persistence(repo, cm):
    """Test that checkpoints are actually stored in the DB and retrievable."""
    trace = _run_happy_agent(repo)
    cps = cm.create_checkpoints_for_run(
        run_id=trace["run_id"],
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    for cp in cps:
        loaded = repo.get_checkpoint(cp.checkpoint_id)
        assert loaded is not None, f"Checkpoint {cp.checkpoint_id} not found in DB"
        assert loaded.checkpoint_id == cp.checkpoint_id


def test_checkpoint_retrieval_by_id(repo, cm):
    """Test get_checkpoint returns correct checkpoint."""
    trace = _run_happy_agent(repo)
    cps = cm.create_checkpoints_for_run(
        run_id=trace["run_id"],
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    first_cp_id = cps[0].checkpoint_id
    loaded = cm.get_checkpoint(first_cp_id)
    assert loaded is not None
    assert loaded.checkpoint_id == first_cp_id


def test_checkpoint_linked_to_correct_run(repo, cm):
    """Test that all checkpoints reference the correct run_id."""
    trace = _run_happy_agent(repo)
    run_id = trace["run_id"]
    cps = cm.create_checkpoints_for_run(
        run_id=run_id,
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    for cp in cps:
        assert cp.run_id == run_id, f"Checkpoint {cp.checkpoint_id} has wrong run_id"


def test_checkpoint_contains_required_observable_state(repo, cm):
    """Test checkpoint state has required observable fields."""
    trace = _run_happy_agent(repo)
    cps = cm.create_checkpoints_for_run(
        run_id=trace["run_id"],
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    for cp in cps:
        state = cp.state
        assert "user_request" in state, f"Missing user_request in checkpoint {cp.checkpoint_id}"
        assert "budget" in state
        assert "retrieved_products" in state
        assert "selected_product" in state
        assert "tools_executed" in state


# --------------------------------------------------------------------------- 6-11: Replay execution


def test_replay_creates_separate_run(repo, cm, engine_obj):
    """Test that replay creates a new run distinct from the original."""
    trace = _run_happy_agent(repo, run_id="orig-001")
    cps = cm.create_checkpoints_for_run(
        run_id="orig-001",
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    assert cps, "Need at least one checkpoint"
    result = engine_obj.replay(run_id="orig-001", checkpoint=cps[0])
    assert result.replay_run_id != "orig-001"
    replay_run = repo.get_run(result.replay_run_id)
    assert replay_run is not None, "Replay run should exist in DB"


def test_original_run_unchanged_after_replay(repo, cm, engine_obj):
    """Test the original run is never modified by a replay."""
    trace = _run_happy_agent(repo, run_id="orig-002")
    original_steps_count = len(trace["steps"])
    original_status = trace["status"]

    cps = cm.create_checkpoints_for_run(
        run_id="orig-002",
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    engine_obj.replay(run_id="orig-002", checkpoint=cps[0])

    reloaded_trace = repo.get_run_trace("orig-002")
    assert reloaded_trace["status"] == original_status
    assert len(reloaded_trace["steps"]) == original_steps_count


def test_replay_events_are_recorded(repo, cm, engine_obj):
    """Test that replay produces recorded events in the new run."""
    trace = _run_happy_agent(repo, run_id="orig-003")
    cps = cm.create_checkpoints_for_run(
        run_id="orig-003",
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    result = engine_obj.replay(run_id="orig-003", checkpoint=cps[0])
    replay_trace = repo.get_run_trace(result.replay_run_id)
    assert replay_trace is not None
    assert len(replay_trace.get("steps", [])) > 0, "Replay must record execution steps"


def test_replay_parent_run_id_is_set(repo, cm, engine_obj):
    """Test that replay run has parent_run_id pointing to original."""
    trace = _run_happy_agent(repo, run_id="orig-004")
    cps = cm.create_checkpoints_for_run(
        run_id="orig-004",
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    result = engine_obj.replay(run_id="orig-004", checkpoint=cps[0])
    replay_run = repo.get_run(result.replay_run_id)
    assert replay_run.parent_run_id == "orig-004"


def test_replay_state_restored_from_checkpoint(repo, cm, engine_obj):
    """Test that the user_request is correctly passed through from checkpoint state."""
    trace = _run_happy_agent(repo, run_id="orig-005")
    cps = cm.create_checkpoints_for_run(
        run_id="orig-005",
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    result = engine_obj.replay(run_id="orig-005", checkpoint=cps[0])
    replay_run = repo.get_run(result.replay_run_id)
    # The replay's user_request should match the checkpoint's user_request
    assert replay_run.user_request == trace["user_request"]


def test_replay_invalid_checkpoint_raises():
    """Test that CheckpointManager raises ValueError for unknown checkpoint."""
    repo = TraceRepository()
    cm = CheckpointManager(repository=repo)
    with pytest.raises(ValueError, match="not found"):
        cm.validate_checkpoint_belongs_to_run("cp-nonexistent-0001", "run-any")


def test_replay_checkpoint_wrong_run_raises(repo, cm):
    """Test that replaying with a checkpoint from a different run is rejected."""
    trace1 = _run_happy_agent(repo, run_id="run-a")
    trace2 = _run_happy_agent(repo, run_id="run-b")
    cps1 = cm.create_checkpoints_for_run(
        run_id="run-a",
        user_request=trace1["user_request"],
        steps=trace1["steps"],
    )
    # Try to use run-a's checkpoint for run-b
    with pytest.raises(ValueError, match="belongs to run"):
        cm.validate_checkpoint_belongs_to_run(cps1[0].checkpoint_id, "run-b")


# --------------------------------------------------------------------------- 12-15: Alternative execution


def test_valid_override_max_budget_accepted(engine_obj):
    """Test that max_budget override is accepted and coerced to int."""
    validated = engine_obj._validate_override({"max_budget": "80000"})
    assert validated is not None
    assert validated["max_budget"] == 80000
    assert isinstance(validated["max_budget"], int)


def test_valid_override_failure_mode_accepted(engine_obj):
    """Test that failure_mode override is accepted."""
    validated = engine_obj._validate_override({"failure_mode": "none"})
    assert validated is not None
    assert validated["failure_mode"] == "none"


def test_invalid_override_key_raises(engine_obj):
    """Test that unknown override keys raise ValueError."""
    with pytest.raises(ValueError, match="not allowed"):
        engine_obj._validate_override({"secret_mode": "hack"})


def test_alternative_run_stored_separately(repo, cm, engine_obj):
    """Test that alternative execution is stored as a new run with override metadata."""
    trace = _run_happy_agent(repo, run_id="orig-alt")
    cps = cm.create_checkpoints_for_run(
        run_id="orig-alt",
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    result = engine_obj.replay(
        run_id="orig-alt",
        checkpoint=cps[0],
        override={"failure_mode": "none"},
    )
    assert result.replay_type == "alternative"
    assert result.override_applied is not None
    replay_run = repo.get_run(result.replay_run_id)
    assert replay_run is not None
    assert replay_run.replay_metadata is not None
    assert replay_run.replay_metadata.get("replay_type") == "alternative"


def test_original_and_replay_both_viewable(repo, cm, engine_obj):
    """Test that both original and replay traces can be independently retrieved."""
    trace = _run_happy_agent(repo, run_id="orig-view")
    cps = cm.create_checkpoints_for_run(
        run_id="orig-view",
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    result = engine_obj.replay(run_id="orig-view", checkpoint=cps[0])

    orig_trace = repo.get_run_trace("orig-view")
    replay_trace = repo.get_run_trace(result.replay_run_id)
    assert orig_trace is not None
    assert replay_trace is not None
    assert orig_trace["run_id"] != replay_trace["run_id"]


# --------------------------------------------------------------------------- 16-17: Failure recovery


def test_budget_violation_replay_with_override(repo, cm, engine_obj):
    """Scenario 1: budget_violation failure, replay with failure_mode=none removes violation."""
    trace = _run_failure_agent(repo, failure_mode="budget_violation", run_id="fail-bv")
    cps = cm.create_checkpoints_for_run(
        run_id="fail-bv",
        user_request=trace["user_request"],
        steps=trace["steps"],
        failure_mode="budget_violation",
    )
    assert cps, "Need checkpoints for failure run"

    # Replay with failure_mode cleared → should no longer inject budget violation
    result = engine_obj.replay(
        run_id="fail-bv",
        checkpoint=cps[0],
        override={"failure_mode": "none"},
    )
    assert result.replay_run_id != "fail-bv"
    # Original run is still accessible and unchanged
    orig = repo.get_run_trace("fail-bv")
    assert orig is not None


def test_wrong_tool_replay_with_override(repo, cm, engine_obj):
    """Scenario 2: wrong_tool failure, replay with failure_mode=none removes wrong tool."""
    trace = _run_failure_agent(repo, failure_mode="wrong_tool", run_id="fail-wt")
    cps = cm.create_checkpoints_for_run(
        run_id="fail-wt",
        user_request=trace["user_request"],
        steps=trace["steps"],
        failure_mode="wrong_tool",
    )
    result = engine_obj.replay(
        run_id="fail-wt",
        checkpoint=cps[0],
        override={"failure_mode": "none"},
    )
    assert result.replay_run_id is not None
    orig = repo.get_run_trace("fail-wt")
    assert orig is not None


# --------------------------------------------------------------------------- 18-20: API endpoints


def test_api_get_checkpoints_returns_list(client, repo):
    """API: GET /runs/{id}/checkpoints returns list of checkpoints."""
    trace = _run_happy_agent(repo, run_id="api-cp-001")
    cm = CheckpointManager(repository=repo)
    cm.create_checkpoints_for_run(
        run_id="api-cp-001",
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    resp = client.get("/runs/api-cp-001/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "api-cp-001"
    assert data["count"] > 0
    assert isinstance(data["checkpoints"], list)


def test_api_get_checkpoints_unknown_run(client):
    """API: GET /runs/{id}/checkpoints with unknown run returns 404."""
    resp = client.get("/runs/no-such-run/checkpoints")
    assert resp.status_code == 404


def test_api_replay_success(client, repo):
    """API: POST /runs/{id}/replay succeeds for a valid checkpoint."""
    trace = _run_happy_agent(repo, run_id="api-replay-001")
    cm = CheckpointManager(repository=repo)
    cps = cm.create_checkpoints_for_run(
        run_id="api-replay-001",
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    assert cps
    resp = client.post(
        "/runs/api-replay-001/replay",
        json={"checkpoint_id": cps[0].checkpoint_id},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["original_run_id"] == "api-replay-001"
    assert data["replay_run_id"] != "api-replay-001"
    assert data["checkpoint_id"] == cps[0].checkpoint_id


def test_api_replay_invalid_checkpoint(client, repo):
    """API: POST /runs/{id}/replay with unknown checkpoint_id returns 400."""
    _run_happy_agent(repo, run_id="api-replay-002")
    resp = client.post(
        "/runs/api-replay-002/replay",
        json={"checkpoint_id": "cp-does-not-exist-0001"},
    )
    assert resp.status_code == 400


def test_api_replay_wrong_run(client, repo):
    """API: POST /runs/{id}/replay with checkpoint from different run returns 400."""
    trace1 = _run_happy_agent(repo, run_id="api-rw-001")
    _run_happy_agent(repo, run_id="api-rw-002")
    cm = CheckpointManager(repository=repo)
    cps = cm.create_checkpoints_for_run(
        run_id="api-rw-001",
        user_request=trace1["user_request"],
        steps=trace1["steps"],
    )
    resp = client.post(
        "/runs/api-rw-002/replay",
        json={"checkpoint_id": cps[0].checkpoint_id},
    )
    assert resp.status_code == 400


def test_api_replay_invalid_override_rejected(client, repo):
    """API: POST /runs/{id}/replay with bad override key returns 422."""
    trace = _run_happy_agent(repo, run_id="api-ov-001")
    cm = CheckpointManager(repository=repo)
    cps = cm.create_checkpoints_for_run(
        run_id="api-ov-001",
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    resp = client.post(
        "/runs/api-ov-001/replay",
        json={"checkpoint_id": cps[0].checkpoint_id, "override": {"hack_mode": True}},
    )
    assert resp.status_code == 422


def test_api_get_replays_for_run(client, repo):
    """API: GET /runs/{id}/replays lists all replay children."""
    trace = _run_happy_agent(repo, run_id="api-rep-list")
    cm = CheckpointManager(repository=repo)
    cps = cm.create_checkpoints_for_run(
        run_id="api-rep-list",
        user_request=trace["user_request"],
        steps=trace["steps"],
    )
    # Perform two replays
    client.post("/runs/api-rep-list/replay", json={"checkpoint_id": cps[0].checkpoint_id})
    client.post("/runs/api-rep-list/replay", json={"checkpoint_id": cps[0].checkpoint_id})

    resp = client.get("/runs/api-rep-list/replays")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
