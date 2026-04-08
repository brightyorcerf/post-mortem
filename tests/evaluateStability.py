"""
evaluate_stability.py  —  SHADOW_REGISTER Determinism Verifier
===============================================================
Runs each of the three scenarios N times with a fixed seed and asserts
that every generated InternalState and every grader score is byte-for-byte
identical across all runs.

If ANY variance is detected the script exits with code 1 and prints a
detailed diff, making it suitable as a CI gate.

Usage
-----
    python evaluate_stability.py                  # 100 iterations, seed=42
    python evaluate_stability.py --n 20 --seed 7  # custom
    python evaluate_stability.py --verbose         # print per-run detail
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Dict, List

from grader import calculate_final_score
from schema import (
    ForensicAction,
    ForensicPivot,
    ActionType,
    IOCType,
)
from worldGen import TASK_EASY, TASK_MEDIUM, TASK_HARD, VALID_TASKS, generate_world
from env import ShadowRegisterEnv


# ---------------------------------------------------------------------------
# Deterministic "oracle" agent  —  always submits the correct answer
# ---------------------------------------------------------------------------
# This agent reads the TruthDAG (internal state) directly — legal here
# because evaluate_stability.py is an evaluator script, not an agent.

def _make_oracle_pivots(env: ShadowRegisterEnv) -> list[ForensicPivot]:
    """Build a perfect SubmitCase from the TruthDAG ground truth."""
    dag = env.state().truth_dag
    return [
        ForensicPivot(
            artifact=node.required_artifact,
            ioc=node.expected_ioc,
            type=node.type,
            reason=f"oracle_submission_node_{node_id}",
        )
        for node_id, node in dag.nodes.items()
        if not node.is_honeypot
    ]


# ---------------------------------------------------------------------------
# World fingerprint  —  a deterministic hash of InternalState
# ---------------------------------------------------------------------------

def _fingerprint_state(env: ShadowRegisterEnv) -> str:
    """
    SHA-256 of the serialised InternalState.
    We serialise via Pydantic's .model_dump() → json.dumps with sorted keys
    so dict-ordering can never produce false positives.
    """
    raw = json.dumps(
        env.state().model_dump(),
        sort_keys=True,
        default=str,      # handles datetime if any slip through
    ).encode()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Single-task stability check
# ---------------------------------------------------------------------------

def check_task_stability(
    task: str,
    seed: int,
    n_iterations: int,
    verbose: bool = False,
) -> bool:
    """
    Returns True if all n_iterations runs are identical; False otherwise.
    Prints diagnostics to stdout.
    """
    print(f"\n  Task: {task!r}  seed={seed}  n={n_iterations}")
    print(f"  {'─' * 48}")

    reference_fingerprint: str | None = None
    reference_score:       float | None = None
    fingerprints: List[str]  = []
    scores:       List[float] = []

    t0 = time.perf_counter()

    for i in range(n_iterations):
        # Catch datetime.now() drift bugs by sleeping between the first two iterations
        if i == 1:
            time.sleep(1.5)

        # --- generate world ---
        world = generate_world(task, seed)
        env   = ShadowRegisterEnv(world)
        env.reset()

        fp = _fingerprint_state(env)
        fingerprints.append(fp)

        # --- run oracle agent to a SubmitCase ---
        pivots = _make_oracle_pivots(env)
        action = ForensicAction(action=ActionType.SUBMIT, pivots=pivots)
        result = env.step(action)

        # --- grade ---
        report = calculate_final_score(
            pivots=pivots,
            truth=env.state().truth_dag,
            remaining_budget=result.observation.remaining_budget,
        )
        scores.append(report.score)

        if verbose and i < 3:
            print(f"    run {i:03d}  fp={fp[:16]}…  score={report.score:.6f}")

        # Capture reference from first run
        if i == 0:
            reference_fingerprint = fp
            reference_score       = report.score

    elapsed = time.perf_counter() - t0

    # --- variance analysis ---
    unique_fps    = set(fingerprints)
    unique_scores = set(scores)

    fp_ok    = len(unique_fps)    == 1
    score_ok = len(unique_scores) == 1

    # Manual σ calculation — no numpy/statistics import needed for a scalar check
    mean_score = sum(scores) / len(scores)
    variance   = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    sigma      = variance ** 0.5

    status = "✓ PASS" if (fp_ok and score_ok) else "✗ FAIL"
    print(f"  {status}  |  {n_iterations} iterations  |  {elapsed:.2f}s")
    print(f"    World fingerprints : {len(unique_fps)} unique  (want 1)")
    print(f"    Grader scores      : {len(unique_scores)} unique  (want 1)")
    print(f"    Score σ            : {sigma:.8f}  (want 0.00000000)")
    print(f"    Reference score    : {reference_score:.6f}")

    if not fp_ok:
        print("\n    ⚠  WORLD FINGERPRINT VARIANCE DETECTED:")
        for idx, fp in enumerate(fingerprints):
            if fp != reference_fingerprint:
                print(f"       run {idx:03d}: {fp}")
                print(f"       ref 000: {reference_fingerprint}")
                break

    if not score_ok:
        print("\n    ⚠  SCORE VARIANCE DETECTED:")
        for idx, s in enumerate(scores):
            if s != reference_score:
                print(f"       run {idx:03d}: score={s:.6f}")
                print(f"       ref 000: score={reference_score:.6f}")
                break

    return fp_ok and score_ok


# ---------------------------------------------------------------------------
# Full stability suite
# ---------------------------------------------------------------------------

def run_stability_suite(
    seed:         int = 42,
    n_iterations: int = 100,
    verbose:      bool = False,
) -> bool:
    """
    Run the stability check for all three tasks.
    Returns True only if every task passes.
    """
    print("=" * 56)
    print("  SHADOW_REGISTER // Stability Evaluator")
    print(f"  seed={seed}  iterations={n_iterations}")
    print("=" * 56)

    results: Dict[str, bool] = {}

    for task in sorted(VALID_TASKS):
        results[task] = check_task_stability(
            task=task,
            seed=seed,
            n_iterations=n_iterations,
            verbose=verbose,
        )

    # --- Summary ---
    print(f"\n{'=' * 56}")
    all_passed = all(results.values())
    for task, passed in results.items():
        icon = "✓" if passed else "✗"
        print(f"  {icon}  {task}")

    print(f"{'─' * 56}")
    if all_passed:
        print("  ✓  ALL TASKS STABLE  —  σ = 0  across all runs")
    else:
        failed = [t for t, p in results.items() if not p]
        print(f"  ✗  INSTABILITY DETECTED in: {', '.join(failed)}")
    print(f"{'=' * 56}\n")

    return all_passed


# ---------------------------------------------------------------------------
# Smoke-test: sanity-check reward range [0.0, 1.0]
# ---------------------------------------------------------------------------

def run_reward_range_check(seed: int = 42) -> bool:
    """
    Verify that scores stay within [0.0, 1.0] for both oracle and
    worst-case (honeypot-only) submissions.
    """
    print("\n  Reward range sanity check…")
    ok = True

    for task in sorted(VALID_TASKS):
        world = generate_world(task, seed)
        dag   = world.truth_dag

        # -- Oracle: should be near 1.0 --
        oracle_pivots = [
            ForensicPivot(
                artifact=n.required_artifact,
                ioc=n.expected_ioc,
                type=n.type,
                reason="range_check",
            )
            for n in dag.nodes.values()
            if not n.is_honeypot
        ]
        r_oracle = calculate_final_score(oracle_pivots, dag, remaining_budget=25)

        # -- Adversarial: submit only honeypots --
        honeypot_pivots = [
            ForensicPivot(
                artifact=n.required_artifact,
                ioc=n.expected_ioc,
                type=n.type,
                reason="range_check_honeypot",
            )
            for n in dag.nodes.values()
            if n.is_honeypot
        ] or oracle_pivots   # fallback if no honeypots

        r_honey = calculate_final_score(honeypot_pivots, dag, remaining_budget=0)

        oracle_in_range = 0.0 <= r_oracle.score <= 1.0
        honey_in_range  = 0.0 <= r_honey.score  <= 1.0

        icon = "✓" if (oracle_in_range and honey_in_range) else "✗"
        print(
            f"    {icon}  {task:28s}  "
            f"oracle={r_oracle.score:.4f}  "
            f"honeypot_only={r_honey.score:.4f}"
        )

        if not (oracle_in_range and honey_in_range):
            ok = False

    return ok


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHADOW_REGISTER determinism / stability verifier"
    )
    parser.add_argument("--n",       type=int,  default=100, help="Iterations per task (default 100)")
    parser.add_argument("--seed",    type=int,  default=42,  help="Random seed (default 42)")
    parser.add_argument("--verbose", action="store_true",    help="Print per-run detail for first 3 runs")
    parser.add_argument("--quick",   action="store_true",    help="Run 5 iterations (smoke test mode)")
    args = parser.parse_args()

    n = 5 if args.quick else args.n

    passed_stability = run_stability_suite(
        seed=args.seed,
        n_iterations=n,
        verbose=args.verbose,
    )
    passed_range = run_reward_range_check(seed=args.seed)

    if passed_stability and passed_range:
        print("All checks passed.\n")
        sys.exit(0)
    else:
        print("One or more checks FAILED. See output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()