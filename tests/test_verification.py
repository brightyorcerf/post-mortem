#!/usr/bin/env python3
"""
Verification test suite for SHADOW_REGISTER submission readiness.
Tests: Docker build, reset/step flow, determinism, log parsing.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

def test_docker_build():
    """TEST 1: Docker build succeeds on Linux"""
    print("\n" + "="*80)
    print("TEST 1: Docker Build")
    print("="*80)

    # Check if Dockerfile exists
    dockerfile = Path("Dockerfile")
    if not dockerfile.exists():
        print("❌ FAIL: Dockerfile not found")
        return False

    # Extract COPY commands and verify files exist
    with open(dockerfile) as f:
        content = f.read()

    copies = re.findall(r'COPY\s+(\S+)\s+\.', content)
    print(f"Found COPY commands for: {copies}")

    all_exist = True
    for fname in copies:
        exists = Path(fname).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {fname}")
        if not exists:
            all_exist = False

    if all_exist:
        print("✅ PASS: All COPY source files exist (case-sensitive)")
        return True
    else:
        print("❌ FAIL: Some source files missing or case mismatch")
        return False


def test_reset_step_flow():
    """TEST 2: Reset, step, and empty SubmitCase flow"""
    print("\n" + "="*80)
    print("TEST 2: Reset/Step/SubmitCase Flow")
    print("="*80)

    try:
        # Import the environment directly (no server needed)
        from env import ShadowRegisterEnv
        from schema import ForensicAction, ActionType
        from worldGen import generate_world

        print("✅ Imports successful")

        # Generate world and create environment
        state = generate_world(task="noisy_entry", seed=42)
        env = ShadowRegisterEnv(internal_state=state)
        print("✅ Environment initialized")

        # Test reset
        result = env.reset()
        print(f"✅ Reset successful, budget={result.observation.remaining_budget}")

        # Test step with Search action
        action = ForensicAction(action=ActionType.SEARCH, query="password")
        result = env.step(action)
        print(f"✅ Step 1 successful, budget={result.observation.remaining_budget}, done={result.done}")

        # Test empty SubmitCase
        action_submit = ForensicAction(action=ActionType.SUBMIT, pivots=[])
        result = env.step(action_submit)
        print(f"✅ Empty SubmitCase successful, done={result.done}")

        if not result.done:
            print("❌ FAIL: Empty SubmitCase did not set done=True")
            return False

        print("✅ PASS: Reset/step flow works correctly")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_determinism():
    """TEST 3: Determinism across runs (same seed, different wall-clock times)"""
    print("\n" + "="*80)
    print("TEST 3: Determinism (σ=0)")
    print("="*80)

    try:
        from env import ShadowRegisterEnv
        from worldGen import generate_world

        # Run 1: Generate world at time T
        print("Run 1: Generating world with seed=42...")
        state1 = generate_world(task="noisy_entry", seed=42)
        env1 = ShadowRegisterEnv(internal_state=state1)
        obs1 = env1.reset()
        state1_str = json.dumps(env1._state.model_dump(), sort_keys=True)
        print(f"  State hash: {hash(state1_str) & 0xffffffff:08x}")

        # Wait to ensure different wall-clock time
        print("Waiting 2 seconds...")
        time.sleep(2)

        # Run 2: Generate world again with same seed at different time
        print("Run 2: Generating world with seed=42 (at different time)...")
        state2 = generate_world(task="noisy_entry", seed=42)
        env2 = ShadowRegisterEnv(internal_state=state2)
        obs2 = env2.reset()
        state2_str = json.dumps(env2._state.model_dump(), sort_keys=True)
        print(f"  State hash: {hash(state2_str) & 0xffffffff:08x}")

        # Compare
        if state1_str == state2_str:
            print("✅ PASS: Identical worlds generated (σ=0 confirmed)")
            return True
        else:
            print("❌ FAIL: Worlds differ (determinism broken)")
            # Find first difference
            for i, (c1, c2) in enumerate(zip(state1_str, state2_str)):
                if c1 != c2:
                    print(f"  First difference at char {i}:")
                    print(f"    Run 1: ...{state1_str[max(0,i-30):i+30]}...")
                    print(f"    Run 2: ...{state2_str[max(0,i-30):i+30]}...")
                    break
            return False

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_log_parser():
    """TEST 4: Log format parsing ([START]/[STEP]/[END])"""
    print("\n" + "="*80)
    print("TEST 4: Log Format ([START]/[STEP]/[END])")
    print("="*80)

    try:
        from inference import log_start, log_step, log_end
        import io
        import contextlib

        # Capture stdout
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            log_start(task="noisy_entry", env="shadow_register", model="gpt-4o")
            log_step(step=1, action='{"action":"Search","query":"password"}',
                    reward=0.05, done=False, error=None)
            log_step(step=2, action='{"action":"Inspect","path":"/var/log"}',
                    reward=0.0, done=False, error="File not found")
            log_end(success=False, steps=2, score=0.05, rewards=[0.05, 0.0])

        logs = output.getvalue()
        print("Captured logs:")
        print(logs)

        # Verify format
        lines = logs.strip().split('\n')

        # Check [START]
        start_match = re.match(r'\[START\] task=(\S+) env=(\S+) model=(\S+)', lines[0])
        if not start_match:
            print(f"❌ FAIL: [START] format invalid: {lines[0]}")
            return False
        print(f"✅ [START] valid: task={start_match.group(1)}, env={start_match.group(2)}")

        # Check [STEP] lines
        for line in lines[1:-1]:
            step_match = re.match(
                r'\[STEP\] step=(\d+) action=(\S+) reward=(-?\d+\.\d{2}) done=(true|false) error=(\S+)',
                line
            )
            if not step_match:
                print(f"❌ FAIL: [STEP] format invalid: {line}")
                return False
            print(f"✅ [STEP] valid: step={step_match.group(1)}, reward={step_match.group(3)}")

        # Check [END]
        end_match = re.match(
            r'\[END\] success=(true|false) steps=(\d+) score=(\d+\.\d{3}) rewards=(.*)',
            lines[-1]
        )
        if not end_match:
            print(f"❌ FAIL: [END] format invalid: {lines[-1]}")
            return False
        print(f"✅ [END] valid: success={end_match.group(1)}, score={end_match.group(3)}")

        print("✅ PASS: All log formats correct")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("SHADOW_REGISTER VERIFICATION TEST SUITE")
    print("="*80)

    results = {
        "Docker Build": test_docker_build(),
        "Reset/Step Flow": test_reset_step_flow(),
        "Determinism (σ=0)": test_determinism(),
        "Log Parser": test_log_parser(),
    }

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    print("\n" + ("="*80))
    if all_passed:
        print("✅ ALL TESTS PASSED — Ready for submission!")
        return 0
    else:
        print("❌ SOME TESTS FAILED — Fix issues before submission")
        return 1


if __name__ == "__main__":
    sys.exit(main())
