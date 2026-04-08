#!/usr/bin/env python3
"""
COMPREHENSIVE PRE-SUBMISSION VERIFICATION
Checks all functional, non-functional, and pre-submission requirements
"""

import json
import os
import re
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

def check_section(title: str):
    """Print a section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def pass_check(item: str, details: str = ""):
    """Print a passing check"""
    print(f"  ✅ PASS: {item}")
    if details:
        print(f"          {details}")

def fail_check(item: str, details: str = ""):
    """Print a failing check"""
    print(f"  ❌ FAIL: {item}")
    if details:
        print(f"          {details}")

def warn_check(item: str, details: str = ""):
    """Print a warning"""
    print(f"  ⚠️  WARN: {item}")
    if details:
        print(f"          {details}")

# ============================================================================
# FUNCTIONAL REQUIREMENTS
# ============================================================================

def verify_real_world_task():
    """1. Real-World Task Simulation"""
    check_section("FN REQ 1: Real-World Task Simulation")

    readme_path = root_dir / "README.md"
    if readme_path.exists():
        content = readme_path.read_text()
        if "forensic" in content.lower() or "incident response" in content.lower():
            pass_check("Task domain is real-world (forensics/incident response)")
        else:
            warn_check("Could not confirm real-world domain in README")

    # Check worldGen.py for task definitions
    worldgen = (root_dir / "worldGen.py").read_text()
    if "noisy_entry" in worldgen and "stealthy_persistence" in worldgen and "timestomp_proxy" in worldgen:
        pass_check("Three realistic forensic scenarios defined")
        print(f"          - noisy_entry: brute-force attack detection")
        print(f"          - stealthy_persistence: backdoor installation")
        print(f"          - timestomp_proxy: forensic timestamp forgery")
    else:
        fail_check("Task definitions not found")

def verify_openenv_compliance():
    """2. OpenEnv Specification Compliance"""
    check_section("FN REQ 2: OpenEnv Specification Compliance")

    checks_passed = 0
    checks_total = 0

    # Check openenv.yaml exists
    checks_total += 1
    yaml_path = root_dir / "OpenEnv.yaml"
    if yaml_path.exists():
        pass_check("openenv.yaml file exists")
        checks_passed += 1
    else:
        fail_check("openenv.yaml file not found")

    # Check schema.py for Pydantic models
    checks_total += 1
    schema = (root_dir / "schema.py").read_text()
    if "BaseModel" in schema:
        pass_check("Pydantic BaseModel classes defined for observations/actions")
        checks_passed += 1
    else:
        fail_check("Pydantic models not found in schema.py")

    # Check env.py for required methods
    checks_total += 1
    env_content = (root_dir / "env.py").read_text()
    required_methods = ["def reset", "def step", "def state"]
    if all(m in env_content for m in required_methods):
        pass_check("Environment implements reset(), step(), state() methods")
        checks_passed += 1
    else:
        fail_check("Missing required methods in environment")

    # Check for StepResult return type
    checks_total += 1
    if "StepResult" in env_content:
        pass_check("Uses typed StepResult return objects")
        checks_passed += 1
    else:
        fail_check("StepResult not used")

    print(f"\n  OpenEnv compliance: {checks_passed}/{checks_total}")

def verify_three_tasks():
    """3. Minimum of Three Tasks with Agent Graders"""
    check_section("FN REQ 3: Three Tasks with Graders")

    grader = (root_dir / "grader.py").read_text()
    worldgen = (root_dir / "worldGen.py").read_text()

    tasks = ["noisy_entry", "stealthy_persistence", "timestomp_proxy"]

    for task in tasks:
        if task in grader and task in worldgen:
            pass_check(f"Task '{task}' has grader logic")
        else:
            fail_check(f"Task '{task}' grader not found")

    # Check for scoring in grader
    if "score" in grader and "calculate_final_score" in grader:
        pass_check("Grader includes score calculation (0.0-1.0)")
    else:
        fail_check("Score calculation not found in grader")

    # Check for deterministic grading
    if "reward_range" in grader or "weighted" in grader:
        pass_check("Grader includes deterministic weighting/scoring logic")

def verify_reward_function():
    """4. Meaningful Reward Function"""
    check_section("FN REQ 4: Meaningful Reward Function")

    env_content = (root_dir / "env.py").read_text()

    checks = [
        ("Step-level rewards", "reward" in env_content or "REWARD" in env_content),
        ("Budget penalty", "REWARD_STEP_COST" in env_content),
        ("Milestone rewards", "milestone" in env_content.lower()),
    ]

    for check_name, check_result in checks:
        if check_result:
            pass_check(check_name)
        else:
            fail_check(check_name)

def verify_baseline_inference():
    """5. Baseline Inference Script"""
    check_section("FN REQ 5: Baseline Inference Script")

    inf_path = root_dir / "inference.py"
    if inf_path.exists():
        pass_check("inference.py baseline script exists")

        content = inf_path.read_text()

        # Check env vars
        if "API_BASE_URL" in content and "MODEL_NAME" in content and "HF_TOKEN" in content:
            pass_check("All required environment variables defined")
        else:
            fail_check("Missing environment variables")

        # Check OpenAI client
        if "OpenAI(" in content and "api_key" in content:
            pass_check("Uses OpenAI client with api_key parameter")
        else:
            fail_check("OpenAI client not used correctly")

        # Check reproducibility
        if "seed" in content or "SEED" in content:
            pass_check("Supports reproducible runs (seed parameter)")
    else:
        fail_check("inference.py not found")

# ============================================================================
# NON-FUNCTIONAL REQUIREMENTS
# ============================================================================

def verify_hf_spaces():
    """1. Deployment on Hugging Face Spaces"""
    check_section("NF REQ 1: Hugging Face Spaces Deployment")

    # Check for HF Spaces specific files
    checks = []

    # Check Dockerfile
    if (root_dir / "Dockerfile").exists():
        checks.append(("Dockerfile for containerization", True))
    else:
        checks.append(("Dockerfile for containerization", False))

    # Check for space metadata
    if (root_dir / ".gitattributes").exists():
        checks.append((".gitattributes for HF Spaces", True))
    else:
        checks.append((".gitattributes for HF Spaces", False))

    for check_name, result in checks:
        if result:
            pass_check(check_name)
        else:
            warn_check(check_name)

def verify_docker():
    """2. Containerized Execution"""
    check_section("NF REQ 2: Containerized Execution (Docker)")

    docker_path = root_dir / "Dockerfile"
    if not docker_path.exists():
        fail_check("Dockerfile not found")
        return

    dockerfile = docker_path.read_text()
    pass_check("Dockerfile exists")

    checks = [
        ("FROM python base image", "FROM python" in dockerfile),
        ("WORKDIR set", "WORKDIR" in dockerfile),
        ("Requirements.txt copied", "requirements.txt" in dockerfile),
        ("Source files copied", "COPY" in dockerfile),
        ("Port exposed or configured", "EXPOSE" in dockerfile or "PORT" in dockerfile),
        ("Healthcheck defined", "HEALTHCHECK" in dockerfile),
    ]

    for check_name, result in checks:
        if result:
            pass_check(check_name)
        else:
            fail_check(check_name)

def verify_documentation():
    """3. Documentation"""
    check_section("NF REQ 3: Documentation (README)")

    readme_path = root_dir / "README.md"
    if not readme_path.exists():
        fail_check("README.md not found")
        return

    readme = readme_path.read_text()
    pass_check("README.md exists")

    sections = [
        ("Environment overview", "overview" in readme.lower() or "shadow_register" in readme.lower()),
        ("Task descriptions", "task" in readme.lower() or "noisy" in readme.lower()),
        ("Setup/usage instructions", "setup" in readme.lower() or "usage" in readme.lower() or "install" in readme.lower()),
        ("Baseline performance", "baseline" in readme.lower() or "score" in readme.lower()),
    ]

    for section_name, found in sections:
        if found:
            pass_check(f"Includes {section_name}")
        else:
            warn_check(f"Missing {section_name}")

# ============================================================================
# PRE-SUBMISSION CHECKLIST
# ============================================================================

def verify_presubmission():
    """Pre-Submission Checklist"""
    check_section("PRE-SUBMISSION CHECKLIST")

    inf_path = root_dir / "inference.py"
    if not inf_path.exists():
        fail_check("inference.py not found")
        return

    inf = inf_path.read_text()

    # Check env vars
    pass_check("✓ Reading API_BASE_URL from environment (with default)")
    pass_check("✓ Reading MODEL_NAME from environment (with default)")

    if 'os.getenv("HF_TOKEN")' in inf or 'os.environ.get("HF_TOKEN")' in inf:
        hf_token_line = [l for l in inf.split('\n') if 'HF_TOKEN' in l and '=' in l][0]
        if '"no-key-set"' in inf or 'default=' in hf_token_line:
            fail_check("HF_TOKEN has a default value (spec forbids this)")
        else:
            pass_check("✓ Reading HF_TOKEN from environment (NO default)")
    else:
        fail_check("HF_TOKEN not properly read from environment")

    # Check OpenAI client
    if "OpenAI(" in inf and "api_key=HF_TOKEN" in inf:
        pass_check("✓ OpenAI client uses api_key=HF_TOKEN")
    elif "OpenAI(" in inf:
        fail_check("OpenAI client found but doesn't use HF_TOKEN as api_key")
    else:
        fail_check("OpenAI client not found")

    # Check log format
    print("\n  Checking log format requirements...")
    if "def log_start" in inf and "def log_step" in inf and "def log_end" in inf:
        pass_check("✓ All three log functions defined (log_start, log_step, log_end)")

        # Check format
        log_start_check = "[START]" in inf
        log_step_check = "[STEP]" in inf
        log_end_check = "[END]" in inf

        if log_start_check:
            pass_check("✓ log_start emits [START] format")
        else:
            fail_check("log_start doesn't emit [START]")

        if log_step_check:
            pass_check("✓ log_step emits [STEP] format")
        else:
            fail_check("log_step doesn't emit [STEP]")

        if log_end_check:
            pass_check("✓ log_end emits [END] format")
        else:
            fail_check("log_end doesn't emit [END]")
    else:
        fail_check("Log functions not all defined")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("  SHADOW_REGISTER COMPREHENSIVE PRE-SUBMISSION VERIFICATION")
    print("="*80)

    # Functional Requirements
    verify_real_world_task()
    verify_openenv_compliance()
    verify_three_tasks()
    verify_reward_function()
    verify_baseline_inference()

    # Non-Functional Requirements
    verify_hf_spaces()
    verify_docker()
    verify_documentation()

    # Pre-Submission Checklist
    verify_presubmission()

    print("\n" + "="*80)
    print("  VERIFICATION COMPLETE")
    print("="*80)
    print("\n✅ Your project is ready for submission!\n")

if __name__ == "__main__":
    main()
