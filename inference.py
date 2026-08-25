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
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import httpx
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
SUCCESS_SCORE_THRESHOLD: float = 0.80

BENCHMARK = "shadow_register"

# ---------------------------------------------------------------------------
# Structured stdout logging  (DO NOT alter field names or order)
# ---------------------------------------------------------------------------

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def _token(value: str) -> str:
    """
    Collapse a value to a single whitespace-free token.

    The [STEP] grammar is space-separated `key=value` pairs, so any embedded
    whitespace (a multi-word Search query, a SubmitCase `reason`) silently
    breaks every field after it.  Underscores keep the line parseable and
    the value readable.
    """
    return "_".join(str(value).split()) or "null"


def log_step(
    *,
    step:   int,
    action: str,
    reward: float,
    done:   bool,
    error:  Optional[str],
) -> None:
    done_str  = "true" if done else "false"
    error_str = _token(error) if error is not None else "null"
    print(
        f"[STEP] step={step} action={_token(action)} reward={reward:.2f}"
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
        f"[END] success={success_str} steps={steps} score={score:.3f}"
        f" rewards={rewards_str}",
        flush=True,
    )


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
      "type":     "NETWORK_IP|EVENT_TIMESTAMP|PATH_TO_FILE|COMMAND_STRING",
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
        f"Budget consumed: {step - 1} / {MAX_STEPS}\n"
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

    Returns {} when nothing parses — the server rejects it and charges a
    budget unit, which is honest.  Substituting a plausible-looking Search
    hid parse failures as ordinary steps.
    """
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    for candidate in (raw, match.group(0) if match else None):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    print(f"[DEBUG] Could not parse model output: {raw[:200]}", flush=True)
    return {}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(task: str, seed: int, max_steps: int) -> None:
    # ── validate server is up ──────────────────────────────────────────
    base = SERVER_URL.rstrip("/")
    http = httpx.Client(timeout=30.0)
    try:
        reachable = http.get(f"{base}/ping").status_code == 200
    except Exception:
        reachable = False
    if not reachable:
        print(
            f"[ERROR] Server not reachable at {SERVER_URL}. "
            "Start it with: python -m server.app",
            file=sys.stderr,
        )
        sys.exit(1)

    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN is not set. Export it before running inference.",
              file=sys.stderr)
        sys.exit(1)

    client_llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    graded:      bool        = False   # did the server return a grader report?

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── reset ─────────────────────────────────────────────────────
        r = http.post(f"{base}/reset", json={"task": task, "seed": seed})
        r.raise_for_status()
        result = r.json()
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
                r = http.post(f"{base}/step", json={"action": action_dict})
                r.raise_for_status()
                result      = r.json()
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

            log_step(step=step, action=action_dict.get("action", "?"),
                     reward=reward, done=done, error=error)
            print(f"[DEBUG] step={step} payload={action_str}", flush=True)

            history.append(
                f"Step {step}: {action_dict.get('action','?')} → reward {reward:+.3f}"
            )

            if done:
                # Pull grader score if server attached it
                grader = result.get("info", {}).get("grader_report")
                if grader:
                    graded  = True
                    score   = float(grader.get("score", 0.0))
                    success = score >= SUCCESS_SCORE_THRESHOLD
                    print(
                        f"[DEBUG] Grader verdict: {grader.get('verdict')}",
                        flush=True,
                    )
                break

        # Fallback normalisation ONLY if the server never returned a grader
        # report.  The grader score is authoritative — a partial-credit run
        # (score < SUCCESS_SCORE_THRESHOLD) must not be overwritten by the
        # step-cost-dominated reward sum.
        if not graded and rewards:
            score   = min(max(sum(rewards), 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            http.close()
        except Exception as e:
            print(f"[DEBUG] Client close error: {e}", flush=True)

        # Clamp score to [0.0, 1.0] then map to (0.05, 0.95)
        # Validator requires strictly (0, 1) — not 0.0 and not 1.0
        score = max(0.0, min(score, 1.0))
        score = score * 0.90 + 0.05
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_TASKS = ["noisy_entry", "stealthy_persistence", "timestomp_proxy"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHADOW_REGISTER baseline inference script"
    )
    parser.add_argument(
        "--task",
        default="noisy_entry",
        choices=["all", "noisy_entry", "stealthy_persistence", "timestomp_proxy"],
        help="Which scenario to run (default: noisy_entry). "
             "'all' runs every task and emits one [START]/[END] block per task, "
             "which single-run log parsers do not expect.",
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

    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]

    for task_name in tasks_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"  Running task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)
        main(task=task_name, seed=args.seed, max_steps=args.max_steps)