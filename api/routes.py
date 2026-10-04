"""FastAPI backend routes for Trace Storage, Retrieval, Agent Execution, and Stage 5 Replay."""

import logging
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from storage.repository import TraceRepository
from storage.database import get_db, Session
from agent.agent import LaptopAgent
from recorder.recorder import ExecutionRecorder

logger = logging.getLogger(__name__)
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
    request: str = Field(description="Prompt for the AI Electronics Product Consultant Agent")
    run_id: Optional[str] = Field(default=None, description="Optional custom run ID")
    failure_mode: Optional[str] = Field(default=None, description="Controlled failure injection mode")
    history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Prior conversation messages for context")
    conversation_id: Optional[str] = Field(default=None, description="Unique conversation session ID for multi-turn memory")


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


# --------------------------------------------------------------------------- Conversation & Agent APIs


@router.get("/conversations/{conversation_id}/messages")
def get_conversation_history(conversation_id: str, repo: TraceRepository = Depends(get_repo)):
    """GET /conversations/{conversation_id}/messages: Returns stored chat history for a session."""
    messages = repo.get_conversation_messages(conversation_id)
    return {
        "conversation_id": conversation_id,
        "count": len(messages),
        "messages": messages,
    }


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, repo: TraceRepository = Depends(get_repo)):
    """DELETE /conversations/{conversation_id}: Clears conversation history."""
    repo.clear_conversation_messages(conversation_id)
    return {"status": "success", "message": f"Conversation '{conversation_id}' cleared."}


@router.get("/conversations")
def list_all_conversations(limit: int = 50, offset: int = 0, repo: TraceRepository = Depends(get_repo)):
    """GET /conversations: Lists active conversation IDs."""
    convs = repo.list_conversations(limit=limit, offset=offset)
    return {"conversations": convs}


@router.post("/agent/run")
def trigger_agent_execution(body: AgentRunRequest, repo: TraceRepository = Depends(get_repo)):
    """Executes the agent with session-based conversation memory, automatic trace recording and optional failure injection.

    Stage 5 enhancement: also auto-generates checkpoints after the run completes.
    """
    conv_id = body.conversation_id or f"conv-{uuid.uuid4().hex[:12]}"

    # 1. Retrieve stored conversation messages for this conversation_id
    stored_messages = repo.get_conversation_messages(conv_id)
    if not stored_messages and body.history:
        # Initialize storage from provided history if storage was empty
        for item in body.history:
            if isinstance(item, dict):
                r = item.get("role", "user")
                c = item.get("content", "")
                if c and c.strip():
                    repo.save_conversation_message(conv_id, role=r, content=c.strip())
        stored_messages = repo.get_conversation_messages(conv_id)

    # 2. Context debug check (INTERNAL LOGGING ONLY - NEVER shown to user)
    logger.info(
        "\n--- [CONTEXT DEBUG CHECK] ---\nconversation_id: %s\nnumber_of_previous_messages: %d\ncurrent_message: %s\n-----------------------------",
        conv_id,
        len(stored_messages),
        body.request,
    )

    # 3. Store the current user message into conversation history
    repo.save_conversation_message(conversation_id=conv_id, role="user", content=body.request.strip())

    # 4. Execute the agent with the prior conversation history passed to LLM
    run_id = body.run_id or f"run-{uuid.uuid4().hex[:8]}"
    recorder = ExecutionRecorder(run_id=run_id, repository=repo)

    agent = LaptopAgent(sinks=[recorder.record])
    agent_result = agent.run(
        body.request,
        failure_mode=body.failure_mode,
        history=stored_messages,
        conversation_id=conv_id,
    )

    # 5. Store the assistant's final response into conversation history
    if agent_result.final_response and agent_result.status == "success":
        repo.save_conversation_message(
            conversation_id=conv_id,
            role="assistant",
            content=agent_result.final_response.strip(),
        )

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
            logger.warning("Checkpoint creation failed: %s", exc)

    return {
        "run_id": run_id,
        "status": agent_result.status,
        "final_response": agent_result.final_response,
        "products": agent_result.products,
        "conversation_id": conv_id,
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


# --------------------------------------------------------------------------- Stage 6: Evaluation & Trace Comparison APIs


@router.get("/runs/{run_id}/evidence")
def get_run_evidence(run_id: str, repo: TraceRepository = Depends(get_repo)):
    """GET /runs/{run_id}/evidence: Returns structured, observable evidence packet for run failure."""
    from failure_intelligence.service import FailureIntelligenceService
    service = FailureIntelligenceService(repository=repo)
    try:
        return service.diagnose_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evidence extraction failed: {exc}")


@router.get("/runs/{run_id}/compare/{alternative_run_id}")
def compare_run_traces(
    run_id: str,
    alternative_run_id: str,
    reference_run_id: Optional[str] = None,
    repo: TraceRepository = Depends(get_repo),
):
    """GET /runs/{run_id}/compare/{alternative_run_id}:

    Performs 3-way trace alignment across original run, alternative/replay run,
    and a verified successful reference run. Checks recovery with the Independent Verifier.
    """
    from failure_intelligence.service import FailureIntelligenceService
    service = FailureIntelligenceService(repository=repo)
    try:
        return service.compare_traces(
            original_run_id=run_id,
            alternative_run_id=alternative_run_id,
            reference_run_id=reference_run_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Trace comparison failed: {exc}")


@router.get("/evaluation/summary")
def get_evaluation_summary(repo: TraceRepository = Depends(get_repo)):
    """GET /evaluation/summary: Returns Top-1, Top-3, and MRR for Random Baseline, Rule-based, and RF."""
    from failure_intelligence.service import FailureIntelligenceService
    service = FailureIntelligenceService(repository=repo)
    return service.get_evaluation_summary()


@router.get("/evaluation/localization")
def get_evaluation_localization(repo: TraceRepository = Depends(get_repo)):
    """GET /evaluation/localization: Returns localization accuracy comparisons and known vs held-out metrics."""
    from failure_intelligence.service import FailureIntelligenceService
    service = FailureIntelligenceService(repository=repo)
    return service.get_evaluation_localization()


@router.get("/evaluation/by-category")
def get_evaluation_by_category(repo: TraceRepository = Depends(get_repo)):
    """GET /evaluation/by-category: Returns Top-1 and Top-3 accuracy per failure category."""
    from failure_intelligence.service import FailureIntelligenceService
    service = FailureIntelligenceService(repository=repo)
    return service.get_evaluation_by_category()


@router.get("/evaluation/comparison")
def get_evaluation_method_comparison(repo: TraceRepository = Depends(get_repo)):
    """GET /evaluation/comparison: Compares Random, Rule-based, and Random Forest models side-by-side."""
    from failure_intelligence.service import FailureIntelligenceService
    service = FailureIntelligenceService(repository=repo)
    summary = service.get_evaluation_summary()
    return {
        "status": summary.get("status"),
        "comparison": summary.get("summary", {}),
    }


@router.get("/evaluation/replay")
def get_evaluation_replay(repo: TraceRepository = Depends(get_repo)):
    """GET /evaluation/replay: Returns branches attempted, recovered, not recovered, and recovery rate."""
    from failure_intelligence.service import FailureIntelligenceService
    service = FailureIntelligenceService(repository=repo)
    return service.get_evaluation_replay()


class BenchmarkRunRequest(BaseModel):
    target_success: Optional[int] = Field(default=20, description="Target successful runs")
    target_failure: Optional[int] = Field(default=40, description="Target failed runs across categories")
    force_regenerate: Optional[bool] = Field(default=True, description="Force re-generation of dataset")


@router.post("/evaluation/run-benchmark")
def run_evaluation_benchmark(body: Optional[BenchmarkRunRequest] = None, repo: TraceRepository = Depends(get_repo)):
    """POST /evaluation/run-benchmark: Generates benchmark runs and runs complete leakage-safe evaluation."""
    from failure_intelligence.service import FailureIntelligenceService
    service = FailureIntelligenceService(repository=repo)
    req = body or BenchmarkRunRequest()
    try:
        eval_res = service.get_or_run_benchmark(
            force_regenerate=req.force_regenerate,
            target_success_count=req.target_success or 20,
            target_failure_count=req.target_failure or 40,
        )
        return eval_res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark evaluation failed: {exc}")


# --------------------------------------------------------------------------- System & Catalogue APIs


@router.get("/catalogue")
def get_product_catalogue():
    """GET /catalogue: Returns full product catalogue from data/products.json."""
    import json
    from pathlib import Path
    data_path = Path(__file__).resolve().parent.parent / "data" / "products.json"
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Catalogue file not found.")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read catalogue: {exc}")


@router.get("/system/status")
def get_system_status(repo: TraceRepository = Depends(get_repo)):
    """GET /system/status: Health status for Agent, Backend, Database."""
    try:
        runs = repo.list_runs(limit=1)
        db_status = "connected"
    except Exception as exc:
        db_status = f"error: {exc}"

    return {
        "status": "healthy",
        "backend": "online",
        "database": db_status,
        "agent": "ready",
        "version": "1.0.0",
    }

