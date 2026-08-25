"""
server.py  —  SHADOW_REGISTER OpenEnv HTTP Wrapper
====================================================
Thin FastAPI layer exposing the three OpenEnv endpoints:

    GET  /ping    → health check
    POST /reset   → start / restart episode
    POST /step    → execute one ForensicAction
    GET  /state   → full InternalState (grader only)

One environment instance is held in process memory.
For the HF Space / single-agent evaluation use-case this is correct.
If you need concurrent sessions, replace _env with a dict keyed by
session_id and pass session_id in each request body.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

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

DEFAULT_TASK = "noisy_entry"

_env: Optional[ShadowRegisterEnv] = None


def _require_env() -> ShadowRegisterEnv:
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    return _env


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: int = 42


class StepRequest(BaseModel):
    # Deliberately untyped: a malformed action must be charged a budget unit by
    # the env, not rejected with a 422 that costs the agent nothing.
    action: Dict[str, Any] = {}


def _serialise_result(result: StepResult) -> Dict[str, Any]:
    """Convert StepResult → plain dict for JSON serialisation."""
    return {
        "observation": result.observation.model_dump(),
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
def reset(req: Optional[ResetRequest] = Body(default=None)) -> JSONResponse:
    """
    Start a fresh episode.

    Body (optional — defaults to noisy_entry with seed=42)
    ----
        task : one of "noisy_entry" | "stealthy_persistence" | "timestomp_proxy"
        seed : integer seed for deterministic world generation (default 42)
    """
    if req is None:
        req = ResetRequest()

    requested_task = req.task
    if requested_task is None:
        # No task named at all — the documented default. Keeps the bare-body
        # validator probe working.
        requested_task = DEFAULT_TASK
    elif requested_task not in VALID_TASKS:
        # A named-but-invalid task must NOT silently become the easy scenario:
        # a typo would otherwise be graded against the wrong answer key.
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{requested_task}'. "
                   f"Valid tasks: {sorted(VALID_TASKS)}",
        )

    global _env
    _env = ShadowRegisterEnv(generate_world(requested_task, req.seed))

    result = _env.reset()
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
    try:
        action = ForensicAction.model_validate(req.action)
    except ValidationError as exc:
        result = env.step_rejected(
            f"Malformed action: {exc.error_count()} schema violation(s)."
        )
    else:
        result = env.step(action)
    payload = _serialise_result(result)

    # Always attach grader report when episode ends (validator requires it)
    if result.done:
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
        # Standard OpenEnv spec requirement: score MUST be at the top level of info
        payload["info"]["score"] = report.score

    return JSONResponse(payload)


@app.get("/state")
def state(x_grader_token: Optional[str] = Header(default=None)) -> JSONResponse:
    """
    Return the full InternalState including TruthDAG.

    GRADER / EVALUATOR USE ONLY.  This is the answer key: it contains every
    expected IOC and the is_honeypot flags.  The agent under evaluation talks
    to this same HTTP surface, so the route is closed unless GRADER_TOKEN is
    set in the environment and echoed back in the X-Grader-Token header.
    """
    expected = os.getenv("GRADER_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=403,
            detail="/state is disabled. Set GRADER_TOKEN in the server "
                   "environment and send it as the X-Grader-Token header.",
        )
    if x_grader_token != expected:
        raise HTTPException(status_code=403, detail="Invalid X-Grader-Token.")

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
    return JSONResponse(content=content)


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()