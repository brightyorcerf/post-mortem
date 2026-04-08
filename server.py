"""
server.py  —  SHADOW_REGISTER OpenEnv HTTP Wrapper
====================================================
Thin FastAPI layer exposing the three OpenEnv endpoints:

    GET  /ping    → health check
    POST /reset   → start / restart episode
    POST /step    → execute one ForensicAction
    GET  /state   → full InternalState (grader only)

One environment instance is held in process memory per session.
For the HF Space / single-agent evaluation use-case this is correct.
If you need concurrent sessions, replace _session with a dict keyed
by session_id and pass session_id in each request body.
"""

from __future__ import annotations
from fastapi.responses import JSONResponse

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import ShadowRegisterEnv, StepResult
from grader import calculate_final_score
from schema import ForensicAction
from worldGen import VALID_TASKS, generate_world

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SHADOW_REGISTER",
    description="Post-Mortem: A Deterministic Benchmark for Forensic Attribution",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# In-process session  (one env at a time)
# ---------------------------------------------------------------------------

class _Session:
    env:  Optional[ShadowRegisterEnv] = None
    task: Optional[str]               = None
    seed: int                         = 42

_session = _Session()


def _require_env() -> ShadowRegisterEnv:
    if _session.env is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    return _session.env


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "noisy_entry"
    seed: int = 42


class StepRequest(BaseModel):
    action: ForensicAction


def _serialise_result(result: StepResult) -> Dict[str, Any]:
    """Convert StepResult → plain dict for JSON serialisation."""
    obs = result.observation
    return {
        "observation": {
            "current_view":      obs.current_view,
            "working_directory": obs.working_directory,
            "artifact_metadata": obs.artifact_metadata.model_dump()
                                  if obs.artifact_metadata else None,
            "tagged_evidence":   obs.tagged_evidence,
            "remaining_budget":  obs.remaining_budget,
            "last_action_log":   obs.last_action_log,
        },
        "reward": result.reward,
        "done":   result.done,
        "info":   result.info,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/ping")
def ping() -> JSONResponse:
    """Health check — required by the OpenEnv validator."""
    return JSONResponse({"status": "ok", "service": "shadow_register"})


@app.post("/reset")
def reset(req: ResetRequest) -> JSONResponse:
    """
    Start a fresh episode.

    Body
    ----
        task : one of "noisy_entry" | "stealthy_persistence" | "timestomp_proxy"
        seed : integer seed for deterministic world generation (default 42)
    """
    if req.task not in VALID_TASKS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task '{req.task}'. Valid: {sorted(VALID_TASKS)}",
        )

    world = generate_world(req.task, req.seed)
    _session.env  = ShadowRegisterEnv(world)
    _session.task = req.task
    _session.seed = req.seed

    result = _session.env.reset()
    return JSONResponse(_serialise_result(result))


@app.post("/step")
def step(req: StepRequest) -> JSONResponse:
    """
    Execute one ForensicAction and return the next observation.

    Body
    ----
        action : ForensicAction (see schema.py / openenv.yaml for field spec)

    If the episode is already done, returns 400.
    When the agent submits SubmitCase and done=True, the grader score is
    included in the response under info.grader_report.
    """
    env = _require_env()
    result = env.step(req.action)
    payload = _serialise_result(result)

    # When the episode ends via SubmitCase, attach the grader report
    if result.done and env.last_pivots:
        report = calculate_final_score(
            pivots=env.last_pivots,
            truth=env.state().truth_dag,
            remaining_budget=result.observation.remaining_budget,
        )
        payload["info"]["grader_report"] = {
            "score":     report.score,
            "verdict":   report.verdict,
            "breakdown": report.breakdown,
            "penalties": report.penalties,
            "bonuses":   report.bonuses,
        }

    return JSONResponse(payload)


@app.get("/state")
def state() -> JSONResponse:
    """
    Return the full InternalState including TruthDAG.
    GRADER / EVALUATOR USE ONLY — never pass this to the agent.
    """
    env = _require_env()
    raw = env.state().model_dump()
    return JSONResponse(raw)

@app.get("/")
def read_root():
    content = {
        "project": "post-mortem",
        "status": "Online",
        "endpoints": {
            "health": "/ping",
            "init": "/reset",
            "action": "/step",
            "debug": "/state"
        },
        "documentation": "https://huggingface.co/spaces/brightyorcerf/post-mortem/blob/main/README.md",
        "message": "Forensics Lab Environment Active."
    }
    return JSONResponse(content=content, indent=4)


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False,
    )