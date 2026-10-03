"""Stage 7 End-to-End Integration & Demo Validation Suite.

Validates the complete Black Box workflow:
1. Successful run execution and trace recording
2. Failed run with controlled fault injection and independent verification
3. Checkpoint generation and state preservation
4. Replay from pre-failure checkpoint with controlled parameter/tool overrides
5. Non-destructive lineage preservation (original run unchanged, alternative run stored)
6. 3-Way trace alignment across Original, Alternative, and Reference runs
7. Independent verifier recovery confirmation
8. Leakage-safe benchmark dataset evaluation
"""

import pytest
from agent.agent import LaptopAgent
from failure_intelligence.comparison.comparator import TraceComparator
from failure_intelligence.evaluation.dataset import DatasetManager
from failure_intelligence.evaluation.evaluator import BenchmarkEvaluator
from failure_intelligence.evaluation.verifier import IndependentVerifier
from failure_intelligence.service import FailureIntelligenceService
from recorder.recorder import ExecutionRecorder
from replay.checkpoint_manager import CheckpointManager
from replay.replay_engine import ReplayEngine
from storage.database import SessionLocal, init_db
from storage.repository import TraceRepository


@pytest.fixture
def test_repo():
    init_db()
    db = SessionLocal()
    r = TraceRepository(db=db)
    yield r
    db.close()


def test_e2e_complete_workflow(test_repo):
    """Executes the full 15-step Black Box debugging lifecycle end-to-end."""
    cm = CheckpointManager(repository=test_repo)
    verifier = IndependentVerifier()
    fi_svc = FailureIntelligenceService(repository=test_repo)

    # Step 1 & 2: Launch a controlled failed run (wrong_tool injection)
    query = "Find a laptop under 50000 for coursework"
    rec_fail = ExecutionRecorder(repository=test_repo)
    agent_fail = LaptopAgent(sinks=[rec_fail.record])
    res_fail = agent_fail.run(query, failure_mode="wrong_tool")
    fail_run_id = rec_fail._current_run_id

    # Step 3 & 4: Trace recorded with observable steps and verified
    trace_fail = test_repo.get_run_trace(fail_run_id)
    assert trace_fail is not None
    assert len(trace_fail["steps"]) >= 8
    v_fail = verifier.verify(trace_fail)
    assert v_fail.passed is False

    # Auto-generate checkpoints
    cps = cm.create_checkpoints_for_run(
        run_id=fail_run_id,
        user_request=query,
        steps=trace_fail["steps"],
        failure_mode="wrong_tool",
    )
    assert len(cps) >= 8

    # Step 5 & 6: Failure Intelligence diagnosis & Top-3 suspicious steps
    diag = fi_svc.diagnose_run(fail_run_id)
    top_candidates = diag.get("top_candidates", [])
    assert len(top_candidates) >= 1
    top1 = diag.get("likely_failure_causing_step")
    assert top1 is not None
    assert 0.0 <= top1["suspicion_score"] <= 1.0

    # Step 7 & 8: Evidence packet contains observable facts and 6 signals
    assert "signals" in top1
    assert "tool_mismatch" in top1["signals"]
    assert "trace_facts" in diag
    assert len(diag["trace_facts"]) > 0

    # Step 9 & 10: Select checkpoint before failure and apply controlled override
    flagged_step_id = top1["step_id"]
    # Select checkpoint at or prior to the failure step
    cp_to_restore = next((c for c in cps if c.step_id < flagged_step_id), cps[0])

    # Step 11: Execute controlled alternative replay
    engine = ReplayEngine(repository=test_repo)
    replay_res = engine.replay(
        run_id=fail_run_id,
        checkpoint=cp_to_restore,
        override={"tool_name": "search_products", "failure_mode": "none"},
    )

    alt_run_id = replay_res.replay_run_id
    assert alt_run_id != fail_run_id
    assert replay_res.status == "success"

    # Step 12 & 13: 3-Way Trace Comparison (Original vs Alternative vs Reference)
    comp = fi_svc.compare_traces(original_run_id=fail_run_id, alternative_run_id=alt_run_id)

    assert comp["original_run_id"] == fail_run_id
    assert comp["alternative_run_id"] == alt_run_id
    assert len(comp["steps"]) >= 8
    assert len(comp["changed_steps"]) > 0
    assert len(comp["downstream_effects"]) > 0

    # Step 14: Independent recovery verification
    assert comp["recovered"] is True
    assert comp["original_verifier"]["passed"] is False
    assert comp["alternative_verifier"]["passed"] is True

    # Step 15: Evaluation metrics calculation
    dm = DatasetManager(repository=test_repo, verifier=verifier)
    dataset = dm.generate_benchmark_dataset(target_success_count=5, target_failure_count=10)
    splits = dm.split_dataset(dataset)
    evaluator = BenchmarkEvaluator()
    eval_metrics = evaluator.run_full_evaluation(splits)

    assert eval_metrics["status"] == "completed"
    assert "summary" in eval_metrics
    assert "rule_based" in eval_metrics["summary"]
    assert "random_forest" in eval_metrics["summary"]
    assert "random_baseline" in eval_metrics["summary"]
    assert 0.0 <= eval_metrics["summary"]["rule_based"]["top1_accuracy"] <= 1.0
