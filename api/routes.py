"""FastAPI backend routes for Trace Storage, Retrieval, Agent Execution, and Stage 5 Replay."""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from storage.repository import TraceRepository
from storage.database import get_db, Session
from agent.agent import LaptopAgent
from recorder.recorder import ExecutionRecorder

router = APIRouter()


# --------------------------------------------------------------------------- Pydantic schemas

class CreateRunRequest(BaseModel):
    run_id: Optional[str] = Field(default=None, description="Custom run ID or auto-generated")
    user_request: str = Field(description="Natural-language user query")
    status: Optional[str] = Field(default="running", description="Initial run status")


class CreateEventRequest(BaseModel):
    event_id: Optional[str] = Field(default=None, description="Unique event identifier")
    step_type: Optional[str] = Field(default=None, description="Step type e.g. tool_execution")
    tool_name: Optional[str] = Field(default=None, description="Tool name if tool step")
    input: Optional[Any] = Field(default=None, description="Structured tool or step input")
    output: Optional[Any] = Field(default=None, description="Structured tool or step output")
    status: Optional[str] = Field(default="success", description="Step status")
    latency: Optional[float] = Field(default=None, description="Duration in seconds")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp")


class AgentRunRequest(BaseModel):
    request: str = Field(description="Prompt for the Laptop Recommendation Agent")
    run_id: Optional[str] = Field(default=None, description="Optional custom run ID")
    failure_mode: Optional[str] = Field(default=None, description="Controlled failure injection mode")


class ReplayRequest(BaseModel):
    """Stage 5: Request to replay from a checkpoint, with optional controlled override."""
    checkpoint_id: str = Field(description="Checkpoint ID to restore state from")
    override: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional controlled override. Allowed keys: "
            "max_budget (int), failure_mode (str), tool_name (str)"
        ),
    )


# --------------------------------------------------------------------------- Dependency

def get_repo(db: Session = Depends(get_db)) -> TraceRepository:
    return TraceRepository(db=db)


# --------------------------------------------------------------------------- Core Trace APIs


@router.post("/runs", status_code=status.HTTP_201_CREATED)
def create_run(body: CreateRunRequest, repo: TraceRepository = Depends(get_repo)):
    """POST /runs: Creates a new agent execution record."""
    run_id = body.run_id or f"run-{uuid.uuid4().hex[:8]}"
    run = repo.create_run(
        run_id=run_id,
        user_request=body.user_request,
        status=body.status or "running",
    )
    return run.to_dict(include_steps=False)


@router.post("/runs/{run_id}/events", status_code=status.HTTP_201_CREATED)
def add_execution_event(run_id: str, body: CreateEventRequest, repo: TraceRepository = Depends(get_repo)):
    """POST /runs/{run_id}/events: Stores an execution event belonging to that run."""
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    step = repo.add_step(run_id=run_id, event_dict=body.model_dump())
    return step.to_dict()


@router.get("/runs", response_model=List[Dict[str, Any]])
def get_all_runs(limit: int = 50, offset: int = 0, repo: TraceRepository = Depends(get_repo)):
    """GET /runs: Returns a list of previous executions."""
    return repo.list_runs(limit=limit, offset=offset)


@router.get("/runs/{run_id}")
def get_one_run(run_id: str, repo: TraceRepository = Depends(get_repo)):
    """GET /runs/{run_id}: Return the complete run and all its execution steps in order."""
    trace = repo.get_run_trace(run_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return trace


@router.get("/runs/{run_id}/diagnosis")
def diagnose_run(run_id: str, repo: TraceRepository = Depends(get_repo)):
    """GET /runs/{run_id}/diagnosis: Analyzes trace and returns root-cause failure intelligence."""
    trace = repo.get_run_trace(run_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    from intelligence.rules import RuleBasedLocalizer
    localizer = RuleBasedLocalizer()
    diagnosis = localizer.diagnose(trace)
    return diagnosis.to_dict()


@router.post("/evaluate/diagnosis")
def evaluate_diagnosis_accuracy(repo: TraceRepository = Depends(get_repo)):
    """POST /evaluate/diagnosis: Runs labeled evaluation scenarios and computes Top-1 & Top-3 accuracy."""
    from intelligence.benchmark import BenchmarkRunner
    runner = BenchmarkRunner(repository=repo)
    metrics = runner.run_benchmark()
    return metrics.to_dict()


@router.post("/agent/run")
def trigger_agent_execution(body: AgentRunRequest, repo: TraceRepository = Depends(get_repo)):
    """Executes the LangChain agent with automatic trace recording and optional failure injection.

    Stage 5 enhancement: also auto-generates checkpoints after the run completes.
    """
    run_id = body.run_id or f"run-{uuid.uuid4().hex[:8]}"
    recorder = ExecutionRecorder(run_id=run_id, repository=repo)

    agent = LaptopAgent(sinks=[recorder.record])
    agent_result = agent.run(body.request, failure_mode=body.failure_mode)

    # Auto-create checkpoints for this run (Stage 5)
    trace = repo.get_run_trace(run_id)
    if trace:
        try:
            from replay.checkpoint_manager import CheckpointManager
            cm = CheckpointManager(repository=repo)
            cm.create_checkpoints_for_run(
                run_id=run_id,
                user_request=body.request,
                steps=trace.get("steps", []),
                failure_mode=body.failure_mode,
            )
        except Exception as exc:
            # Checkpoint creation must not break the main response
            import logging
            logging.getLogger(__name__).warning("Checkpoint creation failed: %s", exc)

    return {
        "run_id": run_id,
        "status": agent_result.status,
        "final_response": agent_result.final_response,
        "trace": trace,
    }


# --------------------------------------------------------------------------- Stage 5: Checkpoint & Replay APIs


@router.get("/runs/{run_id}/checkpoints")
def get_run_checkpoints(run_id: str, repo: TraceRepository = Depends(get_repo)):
    """GET /runs/{run_id}/checkpoints: Returns all checkpoints for a run."""
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    checkpoints = repo.list_checkpoints_for_run(run_id)
    return {
        "run_id": run_id,
        "count": len(checkpoints),
        "checkpoints": checkpoints,
    }


@router.get("/runs/{run_id}/replays")
def get_run_replays(run_id: str, repo: TraceRepository = Depends(get_repo)):
    """GET /runs/{run_id}/replays: Returns all replay runs spawned from this original run."""
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    replays = repo.list_replays_for_run(run_id)
    return {
        "original_run_id": run_id,
        "count": len(replays),
        "replays": replays,
    }


@router.post("/runs/{run_id}/replay", status_code=status.HTTP_201_CREATED)
def replay_from_checkpoint(
    run_id: str,
    body: ReplayRequest,
    repo: TraceRepository = Depends(get_repo),
):
    """POST /runs/{run_id}/replay: Replay agent execution from a checkpoint.

    Validates the run and checkpoint, restores observable state, runs the agent
    from that point forward (with optional controlled overrides), and persists
    the result as a new separate run. The original run is NEVER modified.

    Allowed override keys:
    - max_budget (int): Override the user's budget constraint.
    - failure_mode (str): Change / clear the injected failure ("none" to remove).
    - tool_name (str): Force the first tool call (for wrong_tool recovery).
    """
    # 1. Validate run exists
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    # 2. Validate checkpoint exists and belongs to this run
    from replay.checkpoint_manager import CheckpointManager
    cm = CheckpointManager(repository=repo)
    try:
        checkpoint = cm.validate_checkpoint_belongs_to_run(body.checkpoint_id, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 3. Validate override keys
    from replay.replay_engine import ReplayEngine
    engine = ReplayEngine(repository=repo)
    try:
        validated_override = engine._validate_override(body.override)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # 4. Execute replay
    try:
        result = engine.replay(
            run_id=run_id,
            checkpoint=checkpoint,
            override=body.override,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Replay execution failed: {exc}")

    return {
        "original_run_id": result.original_run_id,
        "replay_run_id": result.replay_run_id,
        "checkpoint_id": result.checkpoint_id,
        "checkpoint_step_id": result.checkpoint_step_id,
        "replay_type": result.replay_type,
        "status": result.status,
        "final_response": result.final_response,
        "override_applied": result.override_applied,
    }


@router.get("/checkpoints/{checkpoint_id}")
def get_checkpoint(checkpoint_id: str, repo: TraceRepository = Depends(get_repo)):
    """GET /checkpoints/{checkpoint_id}: Returns a single checkpoint with its full state."""
    cp = repo.get_checkpoint(checkpoint_id)
    if not cp:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found.")
    return cp.to_dict()
