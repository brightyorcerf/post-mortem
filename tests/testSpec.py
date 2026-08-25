#!/usr/bin/env python3
"""
PRE-SUBMISSION VERIFICATION

Runs the code rather than grepping it.  The previous version asserted that
strings appeared in source files ("does Dockerfile contain WORKDIR", "does
inference.py contain log_start"), which stayed green through 23 real bugs.
"""

import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

FAILURES = []


def check_section(title: str):
    print(f"\n{'=' * 80}\n  {title}\n{'=' * 80}")


def check(item: str, ok: bool, details: str = ""):
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}: {item}")
    if details:
        print(f"          {details}")
    if not ok:
        FAILURES.append(item)


def verify_tasks_grade():
    """Three tasks, each scored end-to-end by the grader."""
    check_section("Three Tasks with Working Graders")
    from grader import calculate_final_score
    from schema import ForensicPivot
    from worldGen import VALID_TASKS, generate_world

    check("Three scenarios registered", len(VALID_TASKS) == 3, sorted(VALID_TASKS))
    for task in sorted(VALID_TASKS):
        dag = generate_world(task, 42).truth_dag
        oracle = [ForensicPivot(artifact=n.required_artifact, ioc=n.expected_ioc,
                                type=n.type, reason="oracle")
                  for n in dag.nodes.values() if not n.is_honeypot]
        best  = calculate_final_score(oracle, dag, 50).score
        worst = calculate_final_score([], dag, 50).score
        check(f"'{task}' grades correctly", best == 1.0 and worst == 0.0,
              f"oracle={best}  empty={worst}")


def verify_openenv_contract():
    """reset/step/state actually run and return typed results."""
    check_section("OpenEnv Contract")
    from env import ShadowRegisterEnv, StepResult
    from schema import ActionType, ForensicAction, ForensicObs
    from worldGen import generate_world

    env = ShadowRegisterEnv(generate_world("noisy_entry", 42))
    r = env.reset()
    check("reset() returns StepResult", isinstance(r, StepResult))
    check("observation is a ForensicObs", isinstance(r.observation, ForensicObs))
    r = env.step(ForensicAction(action=ActionType.SEARCH, query="ssh"))
    check("step() charges budget", r.observation.remaining_budget == 49)
    check("state() exposes the TruthDAG", env.state().truth_dag is not None)


def verify_determinism():
    """Same (task, seed) → identical world."""
    check_section("Determinism")
    from worldGen import generate_world
    for task in ["noisy_entry", "stealthy_persistence", "timestomp_proxy"]:
        a = generate_world(task, 42).model_dump_json()
        b = generate_world(task, 42).model_dump_json()
        check(f"'{task}' is byte-identical across runs", a == b)


def verify_inference_wiring():
    """Config comes from the environment; HF_TOKEN has no default."""
    check_section("Baseline Inference Wiring")
    os.environ.pop("HF_TOKEN", None)
    os.environ["API_BASE_URL"] = "https://example.invalid/v1"
    os.environ["MODEL_NAME"]   = "test-model"
    sys.modules.setdefault("openai", type(sys)("openai")).OpenAI = object
    for mod in [m for m in sys.modules if m == "inference"]:
        del sys.modules[mod]
    import inference

    check("API_BASE_URL read from environment",
          inference.API_BASE_URL == "https://example.invalid/v1")
    check("MODEL_NAME read from environment", inference.MODEL_NAME == "test-model")
    check("HF_TOKEN has NO default (spec requirement)", inference.HF_TOKEN is None)
    check("Emits [START] / [STEP] / [END]",
          all(callable(getattr(inference, f, None))
              for f in ("log_start", "log_step", "log_end")))


def verify_documentation():
    check_section("Documentation")
    readme = (root_dir / "README.md").read_text().lower()
    for topic, needle in [("environment overview", "forensic"),
                          ("task descriptions", "noisy_entry"),
                          ("setup instructions", "docker")]:
        check(f"README covers {topic}", needle in readme)


def main():
    print(f"\n{'=' * 80}\n  SHADOW_REGISTER PRE-SUBMISSION VERIFICATION\n{'=' * 80}")
    verify_tasks_grade()
    verify_openenv_contract()
    verify_determinism()
    verify_inference_wiring()
    verify_documentation()

    check_section("VERIFICATION COMPLETE")
    if FAILURES:
        print(f"\n❌ {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"   • {f}")
        sys.exit(1)
    print("\n✅ Ready for submission.")


if __name__ == "__main__":
    main()
