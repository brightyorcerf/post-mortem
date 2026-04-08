"""
inference.py  —  SHADOW_REGISTER Baseline Inference Script
===========================================================
Runs a single episode of SHADOW_REGISTER using an OpenAI-compatible LLM.
Emits strictly formatted [START] / [STEP] / [END] logs to stdout so the
automated evaluator can parse scores without regex fragility.

Required environment variables
-------------------------------
    API_BASE_URL   LLM API base URL  (e.g. https://api.openai.com/v1)
    MODEL_NAME     Model identifier  (e.g. gpt-4o)
    HF_TOKEN       Hugging Face / API key

Usage
-----
    python inference.py                          # default: noisy_entry, seed=42
    python inference.py --task timestomp_proxy --seed 7
    python inference.py --task stealthy_persistence --max-steps 40
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "gpt-4o")
HF_TOKEN:     str = os.getenv("HF_TOKEN")          # NO default — spec requirement

SERVER_URL:   str = os.environ.get("SERVER_URL",   "http://localhost:7860")

TEMPERATURE:  float = 0.0
MAX_TOKENS:   int   = 1024
MAX_STEPS:    int   = 40          # leave 10 budget units as safety margin
MAX_TOTAL_REWARD: float = 1.0    # used for score normalisation
SUCCESS_SCORE_THRESHOLD: float = 0.80

BENCHMARK = "shadow_register"

# ---------------------------------------------------------------------------
# Structured stdout logging  (DO NOT alter field names or order)
# ---------------------------------------------------------------------------

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    *,
    step:   int,
    action: str,
    reward: float,
    done:   bool,
    error:  Optional[str],
) -> None:
    done_str  = "true" if done else "false"
    error_str = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f}"
        f" done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    *,
    success: bool,
    steps:   int,
    score:   float,
    rewards: List[float],
) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps}"
        f" score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Server client  (thin httpx wrapper around the FastAPI server)
# ---------------------------------------------------------------------------

class ShadowRegisterClient:
    """
    HTTP client for the SHADOW_REGISTER OpenEnv server.
    Mirrors the env.reset() / env.step() interface so inference.py
    reads identically to the OpenEnv SDK pattern.
    """

    def __init__(self, base_url: str = SERVER_URL) -> None:
        import httpx
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(timeout=30.0)
        self.last_grader_report: Optional[Dict[str, Any]] = None

    def ping(self) -> bool:
        try:
            r = self._http.get(f"{self._base}/ping")
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, task: str, seed: int = 42) -> Dict[str, Any]:
        r = self._http.post(
            f"{self._base}/reset",
            json={"task": task, "seed": seed},
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self._http.post(
            f"{self._base}/step",
            json={"action": action},
        )
        r.raise_for_status()
        data = r.json()
        self.last_grader_report = data.get("info", {}).get("grader_report")
        return data

    def close(self) -> None:
        self._http.close()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Digital Forensics & Incident Response (DFIR) analyst \
operating a terminal on a compromised Linux server.

Your goal is to reconstruct the attacker's Kill Chain and file a case report \
using SubmitCase.

AVAILABLE COMMANDS
──────────────────
Search  {"action": "Search",  "query": "<keyword>"}
Inspect {"action": "Inspect", "path":  "<absolute/path>"}
Read    {"action": "Read",    "path":  "<absolute/path>", "offset": <int>}
Tag     {"action": "Tag",     "label": "<key>",           "value": "<evidence>"}
SubmitCase {
  "action": "SubmitCase",
  "pivots": [
    {
      "artifact": "<path>",
      "ioc":      "<value>",
      "type":     "NETWORK_IP|EVENT_TIMESTAMP|PATH_TO_FILE|COMMAND_STRING|USER_ACCOUNT|FILE_HASH",
      "reason":   "<brief explanation>"
    }
  ]
}

RULES
─────
• Every action costs 1 budget unit (max 50). Budget = 0 → episode ends.
• Search returns filenames + hit counts only — NOT file content.
• Read returns a 1000-character window. Use offset to page through large files.
• Inspect returns stat metadata (mtime / atime / ctime / size / permissions).
• Tag records evidence for your own reference. Use it to track findings.
• SubmitCase ends the episode immediately. Only call it when confident.
• Some files are HONEYPOTS. Tagging them penalises your score heavily.
• Respond with EXACTLY ONE JSON action object and nothing else.
"""


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def get_model_action(
    client:       OpenAI,
    step:         int,
    current_view: str,
    last_reward:  float,
    history:      List[str],
) -> Dict[str, Any]:
    """
    Ask the LLM for the next ForensicAction.
    Returns a parsed dict or a fallback Search action on failure.
    """
    history_block = "\n".join(history[-10:]) if history else "(none)"

    user_msg = (
        f"=== Step {step} ===\n"
        f"Budget consumed: {step - 1} / 40\n"
        f"Last reward: {last_reward:+.3f}\n\n"
        f"Recent history:\n{history_block}\n\n"
        f"Current terminal output:\n{current_view}\n\n"
        f"Issue your next command as a single JSON object."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return _parse_action(raw)
    except Exception as exc:
        print(f"[DEBUG] Model call failed: {exc}", flush=True)
        return {"action": "Search", "query": "failed"}


def _parse_action(raw: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from the model's response.
    Handles markdown fences, stray prose, and single-quote JSON.
    """
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Last resort: ast.literal_eval for single-quoted dicts
        try:
            result = ast.literal_eval(candidate)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    # Absolute fallback
    print(f"[DEBUG] Could not parse model output: {raw[:200]}", flush=True)
    return {"action": "Search", "query": "log"}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(task: str, seed: int, max_steps: int) -> None:
    # ── validate server is up ──────────────────────────────────────────
    client_env = ShadowRegisterClient(SERVER_URL)
    if not client_env.ping():
        print(
            f"[ERROR] Server not reachable at {SERVER_URL}. "
            "Start it with: uvicorn server:app --port 8000",
            file=sys.stderr,
        )
        sys.exit(1)

    client_llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── reset ─────────────────────────────────────────────────────
        result = client_env.reset(task=task, seed=seed)
        obs    = result["observation"]
        last_view   = obs["current_view"]
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            if result.get("done", False):
                break

            # ── model decision ────────────────────────────────────────
            action_dict = get_model_action(
                client_llm,
                step=step,
                current_view=last_view,
                last_reward=last_reward,
                history=history,
            )

            # Serialise for logging (compact, no newlines)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            # ── environment step ──────────────────────────────────────
            try:
                result      = client_env.step(action_dict)
                obs         = result["observation"]
                reward      = float(result.get("reward", 0.0))
                done        = bool(result.get("done", False))
                error       = None
                last_view   = obs["current_view"]
                last_reward = reward
            except Exception as exc:
                reward = 0.0
                done   = False
                error  = str(exc)
                print(f"[DEBUG] Step {step} error: {exc}", flush=True)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error)

            history.append(
                f"Step {step}: {action_dict.get('action','?')} → reward {reward:+.3f}"
            )

            if done:
                # Pull grader score if server attached it
                grader = client_env.last_grader_report
                if grader:
                    score   = float(grader.get("score", 0.0))
                    success = score >= SUCCESS_SCORE_THRESHOLD
                    print(
                        f"[DEBUG] Grader verdict: {grader.get('verdict')}",
                        flush=True,
                    )
                break

        # Fallback normalisation if server didn't return grader report
        if not success and rewards:
            score   = min(max(sum(rewards) / MAX_TOTAL_REWARD, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            client_env.close()
        except Exception as e:
            print(f"[DEBUG] Client close error: {e}", flush=True)

        # Clamp score to [0.0, 1.0] before final log — spec requirement
        score = max(0.0, min(score, 1.0))
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHADOW_REGISTER baseline inference script"
    )
    parser.add_argument(
        "--task",
        default="noisy_entry",
        choices=["noisy_entry", "stealthy_persistence", "timestomp_proxy"],
        help="Which scenario to run (default: noisy_entry)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="World-generation seed (default: 42)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Max steps before forced termination (default: {MAX_STEPS})",
    )
    args = parser.parse_args()

    main(task=args.task, seed=args.seed, max_steps=args.max_steps)