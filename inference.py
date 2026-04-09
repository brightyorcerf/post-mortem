"""
inference.py — SHADOW_REGISTER Inference with Schema Validation
================================================================
FULLY FIXED V3 - Includes ACTION SCHEMA VALIDATION
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.getenv("HF_TOKEN")
SERVER_URL: str = os.environ.get("SERVER_URL", "http://localhost:7860")

TEMPERATURE: float = 0.0
MAX_TOKENS: int = 1024
MAX_STEPS: int = 40
MAX_TOTAL_REWARD: float = 1.0
SUCCESS_SCORE_THRESHOLD: float = 0.80
BENCHMARK = "shadow_register"

# ---------------------------------------------------------------------------
# ACTION SCHEMA VALIDATION - CRITICAL FIX
# ---------------------------------------------------------------------------

ACTION_SCHEMA = {
    "Inspect": {
        "required": ["action", "path"],
        "optional": [],
        "description": "Get file metadata (mtime/ctime/size)"
    },
    "Read": {
        "required": ["action", "path", "offset"],
        "optional": [],
        "description": "Read 1000-char window from file at offset"
    },
    "Search": {
        "required": ["action", "query"],
        "optional": [],
        "description": "Search filesystem for keyword"
    },
    "Tag": {
        "required": ["action", "label", "value"],
        "optional": [],
        "description": "Record an IOC for later submission"
    },
    "SubmitCase": {
        "required": ["action"],
        "optional": ["pivots"],
        "description": "End episode and submit evidence"
    }
}

def validate_action(action_dict: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate action against schema."""
    if not isinstance(action_dict, dict):
        return False, "Action must be a JSON object"
    
    if "action" not in action_dict:
        return False, "Missing 'action' field"
    
    action_type = action_dict["action"]
    
    if action_type not in ACTION_SCHEMA:
        valid_actions = ", ".join(sorted(ACTION_SCHEMA.keys()))
        return False, f"Unknown action '{action_type}'. Valid: {valid_actions}"
    
    schema = ACTION_SCHEMA[action_type]
    
    for required_key in schema["required"]:
        if required_key not in action_dict:
            return False, f"Missing required field '{required_key}' for {action_type}"
    
    allowed_keys = set(schema["required"]) | set(schema["optional"])
    actual_keys = set(action_dict.keys())
    extra_keys = actual_keys - allowed_keys
    
    if extra_keys:
        return False, f"Unexpected fields for {action_type}: {', '.join(extra_keys)}"
    
    if action_type == "Inspect":
        if not isinstance(action_dict.get("path"), str):
            return False, "Inspect: 'path' must be a string"
    
    elif action_type == "Read":
        if not isinstance(action_dict.get("path"), str):
            return False, "Read: 'path' must be a string"
        if not isinstance(action_dict.get("offset"), int) or action_dict["offset"] < 0:
            return False, "Read: 'offset' must be non-negative integer"
    
    elif action_type == "Search":
        if not isinstance(action_dict.get("query"), str):
            return False, "Search: 'query' must be a string"
    
    elif action_type == "Tag":
        if not isinstance(action_dict.get("label"), str):
            return False, "Tag: 'label' must be a string"
        if not isinstance(action_dict.get("value"), str):
            return False, "Tag: 'value' must be a string"
    
    return True, None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(*, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str = "true" if done else "false"
    error_str = error if error is not None else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# Server client
# ---------------------------------------------------------------------------

class ShadowRegisterClient:
    """HTTP client for the SHADOW_REGISTER server."""

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
        r = self._http.post(f"{self._base}/reset", json={"task": task, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self._http.post(f"{self._base}/step", json={"action": action})
        r.raise_for_status()
        data = r.json()
        self.last_grader_report = data.get("info", {}).get("grader_report")
        return data

    def close(self) -> None:
        self._http.close()

# =========================================================================
# TASK DEFINITIONS
# =========================================================================

TASK_DEFINITIONS = {
    "noisy_entry": {
        "name": "SSH Brute-Force Attack (Easy)",
        "description": "Find: (1) Attacker source IP, (2) Success login timestamp.",
        "kill_chain": [
            "1. Attacker IP: Failed password attempts from single source",
            "2. Success timestamp: 'Accepted password' after flood",
        ],
        "key_files": ["/var/log/auth.log"],
        "search_keywords": ["Failed password", "Accepted password"],
    },
    "stealthy_persistence": {
        "name": "Cron-Based C2 Persistence (Medium)",
        "description": "Find: (1) Cron job path, (2) Base64 C2 command.",
        "kill_chain": [
            "1. Cron file: /var/spool/cron/crontabs hidden www-data entry",
            "2. Command: Base64-encoded curl+bash beacon",
        ],
        "key_files": ["/var/spool/cron/crontabs", "/var/www/.config"],
        "search_keywords": ["cron", "base64", "curl", "beacon"],
    },
    "timestomp_proxy": {
        "name": "Timestomped Binary + Proxy (Hard)",
        "description": "Find: (1) Binary with mtime!=ctime, (2) Embedded C2 IP.",
        "kill_chain": [
            "1. Binary path: /usr/bin/sudo or /usr/bin/login suspicious timestamps",
            "2. C2 proxy: Extracted from binary as embedded IP",
        ],
        "key_files": ["/usr/bin/sudo", "/usr/bin/login"],
        "search_keywords": ["modified", "trojan", "proxy"],
    },
}

# =========================================================================
# SYSTEM PROMPT - ENFORCES CORRECT ACTION FORMAT
# =========================================================================

UNIVERSAL_SYSTEM_PROMPT = """You are a DFIR Analyst. Goal: Find 2 IOCs, Tag them, then SubmitCase.

=== MANDATORY: CORRECT ACTION FORMAT ===
You MUST use these EXACT parameter names or your actions will FAIL.

Inspect action:
{"action":"Inspect", "path":"/var/log/auth.log"}

Read action (offset REQUIRED, no 'window' key):
{"action":"Read", "path":"/var/log/auth.log", "offset":0}

Search action:
{"action":"Search", "query":"Failed password"}

Tag action (use EXACTLY these field names):
{"action":"Tag", "label":"ATTACKER_IP", "value":"1.2.3.4"}

SubmitCase (with pivots array):
{"action":"SubmitCase", "pivots":[{"artifact":"/path", "ioc":"value", "type":"NETWORK_IP", "reason":"why"}]}

=== PENALTIES FOR WRONG FORMAT ===
• Using "file" instead of "path" → ACTION FAILS (-0.05)
• Using "window" instead of "offset" → ACTION FAILS (-0.05)
• Missing required fields → ACTION FAILS (-0.05)

=== WORKFLOW ===
1. Inspect key files → get metadata
2. Read suspicious files → extract IOCs
3. Tag each IOC → {"action":"Tag", "label":"IOC_NAME", "value":"extracted_value"}
4. Only after 2+ Tags → SubmitCase with pivots

RESPOND: EXACTLY ONE valid JSON action. NO EXPLANATIONS.
"""

def generate_task_aware_prompt(task: str) -> str:
    """Generate task-specific prompt."""
    spec = TASK_DEFINITIONS.get(task, {})
    return UNIVERSAL_SYSTEM_PROMPT + f"""
=== THIS CASE: {spec.get('name')} ===
{spec.get('description')}

Kill chain:
{chr(10).join('  ' + line for line in spec.get('kill_chain', []))}

Key files: {', '.join(spec.get('key_files', []))}
Search terms: {', '.join(spec.get('search_keywords', []))}
"""

# =========================================================================
# ENHANCED EVIDENCE TRACKER
# =========================================================================

class EvidenceTracker:
    """Tracks IOCs and action failures."""
    
    def __init__(self, task: str):
        self.task = task
        self.tagged_iocs: Dict[str, str] = {}
        self.files_read: set = set()
        self.action_errors: List[str] = []
        self.last_action_type: Optional[str] = None
        self.same_action_count: int = 0
    
    def tag_ioc(self, label: str, value: str) -> None:
        self.tagged_iocs[label] = value
    
    def file_read(self, path: str) -> None:
        self.files_read.add(path)
    
    def add_error(self, error: str) -> None:
        self.action_errors.append(error)
    
    def should_break_loop(self) -> Tuple[bool, Optional[str]]:
        """Check if stuck in loop."""
        if len(self.action_errors) >= 3:
            recent = self.action_errors[-3:]
            if all("path" in e or "offset" in e or "query" in e for e in recent):
                return True, f"Stuck in action loop: {recent[-1]}"
        return False, None
    
    def get_ioc_count(self) -> int:
        return len(self.tagged_iocs)
    
    def build_context(self, grader_feedback: Optional[str]) -> str:
        """Build context for next action."""
        lines = []
        lines.append(f"CONFIRMED IOCs: {self.get_ioc_count()}")
        if self.tagged_iocs:
            for label, value in self.tagged_iocs.items():
                lines.append(f"  ✓ {label}: {value[:50]}")
        else:
            lines.append("  (none yet)")
        
        if self.action_errors:
            lines.append(f"\nRECENT ERRORS ({len(self.action_errors)}):")
            for err in self.action_errors[-3:]:
                lines.append(f"  ✗ {err}")
        
        if grader_feedback:
            lines.append(f"\nGRADER: {grader_feedback}")
        
        return "\n".join(lines)
    
    def track_action(self, action_type: str) -> None:
        """Track action type to detect loops."""
        if action_type == self.last_action_type:
            self.same_action_count += 1
        else:
            self.same_action_count = 1
        self.last_action_type = action_type

# =========================================================================
# IMPROVED GET_MODEL_ACTION WITH VALIDATION
# =========================================================================

def get_model_action(
    client: OpenAI,
    task: str,
    step: int,
    current_view: str,
    tracker: EvidenceTracker,
    grader_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate action with schema validation feedback."""
    system_prompt = generate_task_aware_prompt(task)
    context = tracker.build_context(grader_feedback)
    ioc_count = tracker.get_ioc_count()
    ready = ioc_count >= 2
    
    instructions = ""
    if step <= 5:
        instructions = 'PHASE 1: Inspect key files. Use EXACTLY: {"action":"Inspect", "path":"/var/log/auth.log"}'
    elif step <= 15:
        if ioc_count == 0:
            instructions = 'PHASE 2: Read files. Use: {"action":"Read", "path":"/var/log/auth.log", "offset":0}'
        else:
            instructions = f'Found {ioc_count} IOC(s). Use Tag: {{"action":"Tag", "label":"NAME", "value":"value"}}'
    elif ready:
        instructions = f'Ready! Submit with {ioc_count} IOCs: {{"action":"SubmitCase", "pivots":[ ... ]}}'
    else:
        instructions = f"Need {2-ioc_count} more IOCs. Keep searching."
    
    user_msg = (
        f"[STEP {step}/40]\n{instructions}\n\n"
        f"EVIDENCE:\n{context}\n\n"
        f"TERMINAL:\n{current_view[:500]}\n\n"
        f"NEXT ACTION (valid JSON only):"
    )
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=256,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        action_dict = _parse_action(raw)
        
        is_valid, error_msg = validate_action(action_dict)
        if not is_valid:
            tracker.add_error(error_msg or "Unknown validation error")
            print(f"[DEBUG] Action validation failed: {error_msg}", flush=True)
            return {"action": "Search", "query": "evidence"}
        
        return action_dict
        
    except Exception as exc:
        tracker.add_error(f"Model call failed: {str(exc)[:50]}")
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return {"action": "Search", "query": "evidence"}

# =========================================================================
# ACTION PARSING
# =========================================================================

def _parse_action(raw: str) -> Dict[str, Any]:
    """Extract JSON action."""
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return {"action": "Search", "query": "evidence"}

# =========================================================================
# MAIN LOOP
# =========================================================================

def main(task: str, seed: int, max_steps: int) -> None:
    client_env = ShadowRegisterClient(SERVER_URL)
    if not client_env.ping():
        print(f"[ERROR] Server not reachable at {SERVER_URL}", file=sys.stderr)
        sys.exit(1)

    client_llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tracker = EvidenceTracker(task)
    
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = client_env.reset(task=task, seed=seed)
        obs = result["observation"]
        last_view = obs["current_view"]

        for step in range(1, max_steps + 1):
            if result.get("done", False):
                break

            stuck, stuck_msg = tracker.should_break_loop()
            if stuck:
                print(f"[DEBUG] {stuck_msg}", flush=True)
                break

            grader_report = client_env.last_grader_report
            grader_feedback = None
            if grader_report:
                verdict = grader_report.get("verdict", "")
                nodes = grader_report.get("nodes_resolved", 0)
                total = grader_report.get("total_nodes", 0)
                grader_feedback = f"{verdict} ({nodes}/{total})"

            action_dict = get_model_action(
                client_llm, task=task, step=step, current_view=last_view,
                tracker=tracker, grader_feedback=grader_feedback,
            )

            action_str = json.dumps(action_dict, separators=(",", ":"))
            tracker.track_action(action_dict.get("action", "?"))

            try:
                result = client_env.step(action_dict)
                obs = result["observation"]
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                error = None
                last_view = obs["current_view"]
                
                action_type = action_dict.get("action")
                if action_type == "Tag":
                    label = action_dict.get("label", "")
                    value = action_dict.get("value", "")
                    tracker.tag_ioc(label, value)
                    print(f"[DEBUG] Tagged: {label}={value[:40]}", flush=True)
                elif action_type == "Read" and reward > 0:
                    tracker.file_read(action_dict.get("path", ""))
                
            except Exception as exc:
                reward = 0.0
                done = False
                error = str(exc)
                print(f"[DEBUG] Step error: {exc}", flush=True)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                grader = client_env.last_grader_report
                if grader:
                    score = float(grader.get("score", 0.0))
                    success = score >= SUCCESS_SCORE_THRESHOLD
                break

        if not success and rewards:
            score = min(max(sum(rewards) / MAX_TOTAL_REWARD, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            client_env.close()
        except Exception as e:
            print(f"[DEBUG] Close error: {e}", flush=True)

    score = max(0.0, min(score, 1.0)) * 0.98 + 0.01
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_TASKS = ["noisy_entry", "stealthy_persistence", "timestomp_proxy"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHADOW_REGISTER inference")
    parser.add_argument("--task", default="all", choices=["all"] + ALL_TASKS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]

    for task_name in tasks_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f" Running task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)
        main(task=task_name, seed=args.seed, max_steps=args.max_steps)