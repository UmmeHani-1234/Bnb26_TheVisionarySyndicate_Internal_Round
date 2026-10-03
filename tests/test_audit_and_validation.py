"""Final Technical Audit, Integration Testing & Validation Suite for Black Box.

Validates the complete 14-stage workflow and all technical requirements:
1. Normal successful run with 8 standardized observable stages.
2. All 5 controlled failure modes (budget_violation, wrong_tool, timeout, unexpected_output, wrong_interpretation).
3. 6 structured signals computed and normalized [0, 1].
4. Rule-based and Random Forest failure localization (no ground truth leakage).
5. Evidence packet generation with concrete observable facts.
6. Checkpoint persistence, state snapshot immutability, and restoration.
7. Controlled replay and alternative execution branch (original run unmodified).
8. Independent verifier validation and recovery confirmation.
9. 3-way trace comparison (Original vs Alternative vs Reference).
10. Benchmark evaluation and leakage-safe dataset split metrics.
11. REST API endpoints (GET /runs, GET /runs/{id}, GET /diagnosis, GET /evidence, GET /compare, etc.).
12. Graceful error handling for missing runs, invalid checkpoints, and bad overrides.
"""

import os
import uuid
import pytest
from fastapi.testclient import TestClient

from agent.agent import LaptopAgent
from agent.events import EventType
from agent.tools import load_products
from failure_intelligence.service import FailureIntelligenceService
from recorder.recorder import ExecutionRecorder
from replay.checkpoint_manager import CheckpointManager
from replay.replay_engine import ReplayEngine
from server import app
from storage.database import SessionLocal, init_db
from storage.repository import TraceRepository

client = TestClient(app)

STAGE_NAMES = {
    1: "Request Understanding",
    2: "Planning",
    3: "Information Retrieval",
    4: "Tool Selection",
    5: "Tool Execution",
    6: "Result Processing / Interpretation",
    7: "Decision / State Update",
    8: "Final Response",
}


@pytest.fixture(scope="module")
def repo():
    init_db()
    return TraceRepository()


@pytest.fixture(scope="module")
def fi_service(repo):
    return FailureIntelligenceService(repository=repo)


# ===========================================================================
# 1. TEST A NORMAL SUCCESSFUL RUN (8 Standardized Stages)
# ===========================================================================

def test_successful_run_pipeline(repo):
    """Test normal request with no injected fault."""
    run_id = f"test-succ-{uuid.uuid4().hex[:6]}"
    recorder = ExecutionRecorder(run_id=run_id, repository=repo)
    agent = LaptopAgent(sinks=[recorder.record])

    query = "Find me a laptop under 80000 with 16GB RAM for programming"
    res = agent.run(query)

    assert res.status == "success"
    assert res.final_response is not None and len(res.final_response) > 0

    # Retrieve from repository
    trace = repo.get_run_trace(run_id)
    assert trace is not None
    assert trace["run_id"] == run_id
    assert trace["status"] == "success"
    assert trace["user_request"] == query
    assert trace["final_output"] is not None

    # Verify observable steps
    steps = trace.get("steps", [])
    assert len(steps) >= 8, f"Expected at least 8 observable steps, got {len(steps)}"

    step_ids = [s["step_id"] for s in steps]
    # Check that steps are strictly ordered 1..N
    assert step_ids == list(range(1, len(steps) + 1))

    # Verify inputs, outputs, timestamps, and latency
    for s in steps:
        assert s["step_id"] in STAGE_NAMES or s["step_id"] > 8
        assert s["status"] in ("success", "running", "error")
        if s.get("latency") is not None:
            assert s["latency"] >= 0.0

    # Verify run is retrievable via API
    api_resp = client.get(f"/runs/{run_id}")
    assert api_resp.status_code == 200
    assert api_resp.json()["status"] == "success"


# ===========================================================================
# 2. TEST EVERY CONTROLLED FAILURE MODE
# ===========================================================================

@pytest.mark.parametrize("f_mode", [
    "budget_violation",
    "wrong_tool",
    "timeout",
    "unexpected_output",
    "wrong_interpretation",
])
def test_all_controlled_failure_modes(repo, fi_service, f_mode):
    """Verifies that all 5 failure modes are correctly captured, recorded, and diagnosed."""
    run_id = f"test-fail-{f_mode}-{uuid.uuid4().hex[:6]}"
    recorder = ExecutionRecorder(run_id=run_id, repository=repo)
    agent = LaptopAgent(sinks=[recorder.record])

    query = f"Find a laptop under 50000 for coursework ({f_mode})"
    res = agent.run(query, failure_mode=f_mode)

    # Verify trace persisted
    trace = repo.get_run_trace(run_id)
    assert trace is not None
    assert trace["run_id"] == run_id

    # Create checkpoints
    cm = CheckpointManager(repository=repo)
    checkpoints = cm.create_checkpoints_for_run(
        run_id=run_id,
        user_request=query,
        steps=trace.get("steps", []),
        failure_mode=f_mode,
    )
    assert len(checkpoints) >= 1

    # Run Diagnosis (without passing ground-truth label to diagnosis)
    diag = fi_service.diagnose_run(run_id)
    assert diag is not None
    assert diag["run_id"] == run_id

    # Verify Top-1 and Top-candidates
    top_candidates = diag.get("top_candidates", [])
    assert len(top_candidates) >= 1

    top1 = diag.get("likely_failure_causing_step")
    assert top1 is not None
    assert "suspicion_score" in top1
    assert 0.0 <= top1["suspicion_score"] <= 1.0

    # Verify 6 structured signals are normalized in [0, 1]
    signals = top1.get("signals", {})
    signal_keys = [
        "tool_mismatch",
        "output_divergence",
        "state_divergence",
        "downstream_impact",
        "latency_anomaly",
        "error_status",
    ]
    for key in signal_keys:
        assert key in signals, f"Missing signal {key} in evidence packet"
        assert 0.0 <= signals[key] <= 1.0, f"Signal {key} value {signals[key]} out of range [0, 1]"

    # Verify trace facts
    facts = diag.get("trace_facts", [])
    assert isinstance(facts, list)


# ===========================================================================
# 3. TEST CHECKPOINTING, REPLAY & ALTERNATIVE EXECUTION
# ===========================================================================

def test_checkpoint_replay_alternative_lifecycle(repo, fi_service):
    """Tests restoring a checkpoint, applying an override, creating an alternative branch,
    and verifying with the Independent Verifier and 3-Way Comparator.
    """
    # 1. Create a failed run with wrong_tool
    orig_run_id = f"test-replay-orig-{uuid.uuid4().hex[:6]}"
    recorder = ExecutionRecorder(run_id=orig_run_id, repository=repo)
    agent = LaptopAgent(sinks=[recorder.record])
    query = "Find lightweight laptop under 75000"
    agent.run(query, failure_mode="wrong_tool")

    orig_trace = repo.get_run_trace(orig_run_id)
    assert orig_trace is not None

    cm = CheckpointManager(repository=repo)
    cps = cm.create_checkpoints_for_run(
        run_id=orig_run_id,
        user_request=query,
        steps=orig_trace.get("steps", []),
        failure_mode="wrong_tool",
    )
    assert len(cps) > 0

    # Select checkpoint before the failure step (Step 4)
    target_cp = next((c for c in cps if c.step_id == 4), cps[0])
    cp_model = cm.get_checkpoint(target_cp.checkpoint_id)
    assert cp_model is not None
    assert cp_model.state is not None

    # Checkpoint immutability
    original_state_snapshot = dict(cp_model.state)

    # 2. Execute replay with controlled override
    engine = ReplayEngine(repository=repo)
    override = {"tool_name": "search_products", "failure_mode": "none"}
    replay_result = engine.replay(
        run_id=orig_run_id,
        checkpoint=cp_model,
        override=override,
    )

    alt_run_id = replay_result.replay_run_id
    assert alt_run_id != orig_run_id
    assert replay_result.original_run_id == orig_run_id

    # Verify original run is intact and unmodified
    orig_check = repo.get_run_trace(orig_run_id)
    assert orig_check["run_id"] == orig_run_id
    cp_recheck = cm.get_checkpoint(target_cp.checkpoint_id)
    assert cp_recheck.state == original_state_snapshot

    # Verify alternative run exists and has parent linkage
    alt_trace = repo.get_run_trace(alt_run_id)
    assert alt_trace is not None
    assert alt_trace["parent_run_id"] == orig_run_id
    assert alt_trace["status"] == "success"

    # 3. Independent Verifier
    v_orig = fi_service.verifier.verify(orig_trace)
    v_alt = fi_service.verifier.verify(alt_trace)
    assert v_alt.passed is True

    # 4. 3-Way Trace Comparison
    comp = fi_service.compare_traces(
        original_run_id=orig_run_id,
        alternative_run_id=alt_run_id,
    )
    assert comp["original_run_id"] == orig_run_id
    assert comp["alternative_run_id"] == alt_run_id
    assert comp["recovered"] is True
    assert len(comp["steps"]) >= 4

    # Check that changed step is flagged
    changed_steps = [s for s in comp["steps"] if s["is_changed"]]
    assert len(changed_steps) >= 1


# ===========================================================================
# 4. TEST BENCHMARK EVALUATION & LEAKAGE ISOLATION
# ===========================================================================

def test_evaluation_benchmark_and_split_isolation(fi_service):
    """Verifies that evaluation metrics are computed across splits without data leakage."""
    # Test evaluation_not_ready when no benchmark cached
    summary_initial = fi_service.get_evaluation_summary()
    assert summary_initial["status"] in ("evaluation_not_ready", "success")

    # Run benchmark
    eval_res = fi_service.get_or_run_benchmark(
        force_regenerate=True,
        target_success_count=12,
        target_failure_count=24,
    )
    assert eval_res["status"] in ("completed", "success")
    assert "summary" in eval_res

    summ = eval_res["summary"]
    for method in ("random_baseline", "rule_based", "random_forest"):
        assert method in summ
        m_data = summ[method]
        assert "top1_accuracy" in m_data
        assert "top3_accuracy" in m_data
        assert "mrr" in m_data
        assert 0.0 <= m_data["top1_accuracy"] <= 1.0
        assert 0.0 <= m_data["top3_accuracy"] <= 1.0
        assert 0.0 <= m_data["mrr"] <= 1.0

    # Rule-Based and Random Forest should beat or match random baseline
    assert summ["rule_based"]["top1_accuracy"] >= summ["random_baseline"]["top1_accuracy"]
    assert summ["random_forest"]["top1_accuracy"] >= summ["random_baseline"]["top1_accuracy"]

    # Check known vs held-out evaluation
    assert "known_vs_held_out" in eval_res
    kvh = eval_res["known_vs_held_out"]
    assert "known_categories" in kvh
    assert "held_out_category" in kvh

    # Check per-category breakdown
    assert "by_category" in eval_res
    assert len(eval_res["by_category"]) > 0


# ===========================================================================
# 5. TEST REST API ENDPOINTS
# ===========================================================================

def test_rest_api_endpoints(repo):
    """Tests all primary REST endpoints."""
    # 1. Health Check
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # 2. List Runs
    r = client.get("/runs?limit=10")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

    # 3. Create run & step via API
    run_id = f"api-test-{uuid.uuid4().hex[:6]}"
    r = client.post("/runs", json={"run_id": run_id, "user_request": "API test laptop query"})
    assert r.status_code == 201
    assert r.json()["run_id"] == run_id

    # 4. Add Event
    r = client.post(f"/runs/{run_id}/events", json={
        "step_type": "request_understanding",
        "status": "success",
        "input": {"query": "API test"},
    })
    assert r.status_code == 201

    # 5. Get Run Trace
    r = client.get(f"/runs/{run_id}")
    assert r.status_code == 200
    assert r.json()["run_id"] == run_id

    # 6. Evaluation endpoints
    r = client.get("/evaluation/summary")
    assert r.status_code == 200

    r = client.get("/evaluation/localization")
    assert r.status_code == 200

    r = client.get("/evaluation/by-category")
    assert r.status_code == 200

    r = client.get("/evaluation/comparison")
    assert r.status_code == 200

    r = client.get("/evaluation/replay")
    assert r.status_code == 200


# ===========================================================================
# 6. TEST EDGE CASES & GRACEFUL ERROR HANDLING
# ===========================================================================

def test_graceful_error_handling():
    """Verifies that invalid requests return proper HTTP error codes without crashing."""
    # Non-existent run
    r = client.get("/runs/non-existent-run-id-999")
    assert r.status_code == 404

    # Non-existent checkpoint
    r = client.get("/checkpoints/non-existent-cp-id-999")
    assert r.status_code == 404

    # Replay on non-existent run
    r = client.post("/runs/non-existent-run-id/replay", json={"checkpoint_id": "cp-123"})
    assert r.status_code == 404

    # Replay with invalid override key
    r = client.post("/runs/any-run/replay", json={
        "checkpoint_id": "cp-123",
        "override": {"unsupported_key": "hack"},
    })
    # Either 404 (run not found) or 422 (unsupported override key)
    assert r.status_code in (404, 422)
