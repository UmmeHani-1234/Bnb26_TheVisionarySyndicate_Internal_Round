"""FastAPI backend routes for Trace Storage, Retrieval, and Agent Execution."""

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


# Pydantic schemas for request validation
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


def get_repo(db: Session = Depends(get_db)) -> TraceRepository:
    return TraceRepository(db=db)


# --------------------------------------------------------------------------- APIs


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


@router.post("/agent/run")
def trigger_agent_execution(body: AgentRunRequest, repo: TraceRepository = Depends(get_repo)):
    """Executes the LangChain agent with automatic trace recording."""
    run_id = body.run_id or f"run-{uuid.uuid4().hex[:8]}"
    recorder = ExecutionRecorder(run_id=run_id, repository=repo)

    agent = LaptopAgent(sinks=[recorder.record])
    agent_result = agent.run(body.request)

    trace = repo.get_run_trace(run_id)
    return {
        "run_id": run_id,
        "status": agent_result.status,
        "final_response": agent_result.final_response,
        "trace": trace,
    }
