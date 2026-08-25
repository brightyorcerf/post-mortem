#!/usr/bin/env python3
"""
Regression checks for the bugs found in the 2026-08 audit.
Each assert corresponds to one fix; run directly: python3 tests/test_audit_regressions.py
"""
import io, json, os, re, sys, contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from env import ShadowRegisterEnv, READ_WINDOW
from grader import calculate_final_score
from schema import ActionType, ForensicAction, ForensicPivot, IOCType
from worldGen import generate_world

TASKS = ["noisy_entry", "stealthy_persistence", "timestomp_proxy"]


def _world(task="noisy_entry", seed=42):
    env = ShadowRegisterEnv(generate_world(task, seed))
    env.reset()
    return env


def test_state_route_closed_without_token():
    import server.app as app_mod
    os.environ.pop("GRADER_TOKEN", None)
    c = TestClient(app_mod.app)
    c.post("/reset", json={"task": "noisy_entry", "seed": 42})
    assert c.get("/state").status_code == 403, "answer key served without a token"

    os.environ["GRADER_TOKEN"] = "s3cret"
    assert c.get("/state").status_code == 403, "wrong/missing header accepted"
    r = c.get("/state", headers={"X-Grader-Token": "s3cret"})
    assert r.status_code == 200 and r.json()["truth_dag"]["nodes"]
    os.environ.pop("GRADER_TOKEN")


def test_grader_score_is_not_overwritten_by_reward_sum():
    src = Path(__file__).resolve().parent.parent / "inference.py"
    body = src.read_text()
    assert "if not graded and rewards:" in body, \
        "fallback must key off 'did the grader report', not 'was the run successful'"


def test_step_log_line_is_parseable():
    sys.modules.setdefault("openai", type(sys)("openai")).OpenAI = object
    import inference
    rx = (r'\[STEP\] step=(\d+) action=(\S+) reward=(-?\d+\.\d{2}) '
          r'done=(true|false) error=(\S+)')
    for action, err in [("SubmitCase", None), ("Search", "read timeout on host")]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            inference.log_step(step=1, action=action, reward=-0.05,
                               done=False, error=err)
        assert re.match(rx, buf.getvalue().strip()), buf.getvalue()


def test_no_free_credit_for_empty_or_tiny_ioc():
    for task in TASKS:
        dag = generate_world(task, 42).truth_dag
        for junk in ["", " ", "-", "m", "2025"]:
            pivots = [
                ForensicPivot(artifact=n.required_artifact, ioc=junk,
                              type=n.type, reason="guess")
                for n in dag.nodes.values() if not n.is_honeypot
            ]
            score = calculate_final_score(pivots, dag, 50).score
            assert score == 0.0, f"{task}: ioc={junk!r} scored {score}"


def test_partial_timestamp_still_earns_credit():
    """The distinctiveness floor must not make the hard task's node C unreachable."""
    st = generate_world("timestomp_proxy", 42)
    node_c = st.truth_dag.nodes["C"]
    ctime = node_c.expected_ioc.split("ctime=")[1]
    pivots = [ForensicPivot(artifact=node_c.required_artifact, ioc=ctime,
                            type=node_c.type, reason="inode change time")]
    assert calculate_final_score(pivots, st.truth_dag, 0).breakdown["C"]["matched"]


def test_read_paging_is_lossless():
    """Page every file by the advance its own header advertises; nothing may be skipped."""
    checked = 0
    for task in TASKS:
        state = generate_world(task, 42)
        for path, vf in state.filesystem.items():
            env = ShadowRegisterEnv(state)   # fresh budget per file
            env.reset()
            seen, offset = "", 0
            while offset < len(vf.content):
                view = env.step(ForensicAction(action=ActionType.READ, path=path,
                                               offset=offset)).observation.current_view
                assert len(view) <= READ_WINDOW, f"{path}: view {len(view)} > cap"
                advance = int(re.search(r"\+(\d+) chars\]", view).group(1))
                assert advance > 0, f"{path}: header advertises a zero advance"
                body = view.split("\n", 2)[2].rsplit("\n", 2)[0]
                assert body == vf.content[offset:offset + advance], \
                    f"{path}: rendered body disagrees with the advertised advance"
                seen += body
                offset += advance
            assert seen == vf.content, f"{task}:{path} lost bytes while paging"
            checked += 1
    assert checked >= 30, f"only {checked} files exercised"


def test_tagging_a_legitimate_account_is_not_penalised():
    env = _world("noisy_entry")
    r = env.step(ForensicAction(action=ActionType.TAG, label="u", value="ubuntu"))
    assert r.reward == -0.05, f"legit /etc/passwd user penalised: {r.reward}"


def test_tagging_real_evidence_pays_once():
    st = generate_world("timestomp_proxy", 42)
    env = ShadowRegisterEnv(st); env.reset()
    c2 = st.truth_dag.nodes["B"].expected_ioc
    first = env.step(ForensicAction(action=ActionType.TAG, label="a", value=c2)).reward
    again = env.step(ForensicAction(action=ActionType.TAG, label="b", value=c2)).reward
    assert first > 0, "Tag can only ever lose points"
    assert again == -0.05, "repeat tag farms reward"


def test_inspect_earns_the_milestone_too():
    env = _world("timestomp_proxy")
    assert env.step(ForensicAction(action=ActionType.INSPECT,
                                   path="/usr/bin/login")).reward > 0
    view = env.step(ForensicAction(action=ActionType.INSPECT,
                                   path="/usr/bin/login")).observation.current_view
    assert "mtime" in view and "ctime" in view, "stat field vocabulary hidden"


def test_unknown_task_is_rejected_not_silently_downgraded():
    import server.app as app_mod
    c = TestClient(app_mod.app)
    assert c.post("/reset", json={"task": "timestomp-proxy"}).status_code == 400
    assert c.post("/reset", json={}).status_code == 200          # documented default
    assert c.post("/reset", json={"task": "timestomp_proxy"}).status_code == 200


def test_malformed_action_costs_budget():
    import server.app as app_mod
    c = TestClient(app_mod.app)
    before = c.post("/reset", json={"task": "noisy_entry"}).json()
    r = c.post("/step", json={"action": {"action": "ls"}}).json()
    assert r["observation"]["remaining_budget"] == \
        before["observation"]["remaining_budget"] - 1, "free step for junk input"
    assert r["reward"] == -0.05


def test_efficiency_bonus_is_reported_only_when_applied():
    dag = generate_world("timestomp_proxy", 42).truth_dag
    every = [ForensicPivot(artifact=n.required_artifact, ioc=n.expected_ioc,
                           type=n.type, reason="r")
             for n in dag.nodes.values() if not n.is_honeypot]
    assert calculate_final_score(every, dag, 50).bonuses == [], \
        "claims a bonus the 1.0 clamp swallowed"
    partial = calculate_final_score(every[:2], dag, 50)
    assert partial.bonuses and partial.score > calculate_final_score(every[:2], dag, 0).score


def test_truth_artifacts_survive_noise_generation():
    for task in TASKS:
        st = generate_world(task, 42)
        for n in st.truth_dag.nodes.values():
            assert n.required_artifact in st.filesystem, f"{task}: {n.required_artifact}"


def test_state_readable_before_reset():
    ShadowRegisterEnv(generate_world("noisy_entry", 1)).state()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ❌ {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
