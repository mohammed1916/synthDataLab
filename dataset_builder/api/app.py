"""FastAPI app for synthDataLab pipeline orchestration."""
from __future__ import annotations

from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from .service import RunStatusEnum, controller


class RunRequest(BaseModel):
    input_path: str | None = None
    input_text: str | None = None
    mock: bool = False
    workers: int = Field(1, ge=1, le=16)
    agent: bool = False
    steering: str = Field("auto")
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    force: bool = False
    reset_fingerprints: bool = False

    @validator("steering")
    def steering_must_be_valid(v: str) -> str:  # noqa: N805
        if v not in {"auto", "review-low", "review-all"}:
            raise ValueError("steering must be one of: auto, review-low, review-all")
        return v


class RunResponse(BaseModel):
    run_id: str
    status: str
    created_at: str
    updated_at: str
    cancel_requested: bool = False
    error: str | None = None
    outputs: dict[str, str] | None = None


app = FastAPI(
    title="synthDataLab API",
    description="API for managing synthetic dataset pipeline runs",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    result: dict[str, Any] = {"status": "ok", "service": "synthdatalab"}
    db_health = getattr(controller, "_db", None)
    if db_health is not None and getattr(db_health, "enabled", False):
        result["database"] = db_health.health()
    else:
        result["database"] = {"enabled": False}
    return result


@app.get("/api/runs/{run_id}/summary")
def run_summary(run_id: str):
    summary = controller.get_run_summary(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Run summary not found")
    return summary


@app.post("/api/runs", status_code=202, response_model=RunResponse)
def create_run(payload: RunRequest, background_tasks: BackgroundTasks):
    if not payload.input_path and not payload.input_text:
        raise HTTPException(status_code=400, detail="Either input_path or input_text must be provided")

    run_status = controller.create_run(
        input_path=payload.input_path,
        input_text=payload.input_text,
        mock=payload.mock,
        workers=payload.workers,
        agent=payload.agent,
        steering=payload.steering,
        threshold=payload.threshold,
        force=payload.force,
        reset_fingerprints=payload.reset_fingerprints,
    )

    return RunResponse(
        run_id=run_status.run_id,
        status=run_status.status.value,
        created_at=run_status.created_at,
        updated_at=run_status.updated_at,
        cancel_requested=run_status.cancel_requested,
        error=run_status.error,
        outputs=run_status.outputs,
    )


@app.get("/api/runs", response_model=list[dict[str, Any]])
def list_runs(status: str | None = None):
    runs = controller.list_runs()
    if status:
        allowed = {s.value for s in RunStatusEnum}
        if status not in allowed:
            raise HTTPException(status_code=400, detail=f"Invalid status filter. Allowed: {sorted(allowed)}")
        runs = [r for r in runs if r.get("status") == status]
    return runs


@app.get("/api/runs/{run_id}", response_model=dict[str, Any])
def get_run(run_id: str):
    run = controller.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/api/runs/{run_id}/cancel", response_model=RunResponse)
def cancel_run(run_id: str):
    run_status = controller.cancel_run(run_id)
    if not run_status:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunResponse(
        run_id=run_status.run_id,
        status=run_status.status.value,
        created_at=run_status.created_at,
        updated_at=run_status.updated_at,
        cancel_requested=run_status.cancel_requested,
        error=run_status.error,
        outputs=run_status.outputs,
    )


@app.get("/api/runs/{run_id}/logs")
def get_run_logs(run_id: str):
    logs = controller.get_run_logs(run_id)
    if logs is None:
        raise HTTPException(status_code=404, detail="Run logs not found")
    return {"run_id": run_id, "logs": logs}
