#!/usr/bin/env python3
"""
Final Integration Test: API Endpoints with Grader Exposure
===========================================================

This script simulates the exact HTTP flow that Phase 2 validator performs:

1. POST /reset with task_id → StepResult
2. GET /state → InternalState with TruthDAG
3. POST /step with SubmitCase → StepResult with grader_report
4. Verify all responses are JSON serializable
"""

import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from schema import ForensicAction, ActionType, InternalState
from worldGen import generate_world, VALID_TASKS
from env import ShadowRegisterEnv, StepResult
import yaml


def section(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def pass_test(msg: str, detail: str = ""):
    print(f"  ✅ {msg}")
    if detail:
        print(f"     {detail}")


def fail_test(msg: str, detail: str = ""):
    print(f"  ❌ {msg}")
    if detail:
        print(f"     {detail}")
    return False


def test_endpoint_flow():
    """Simulate the full HTTP flow for all three tasks"""
    
    section("ENDPOINT FLOW TEST: Simulating API Validator")
    
    # Load openenv spec
    yaml_path = root_dir / "openenv.yaml"
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)
    
    all_passed = True
    
    for task_id in sorted(VALID_TASKS):
        print(f"\n--- Testing Task: {task_id} ---")
        
        try:
            # ================================================================
            # STEP 1: Simulate POST /reset
            # ================================================================
            print(f"\n  1️⃣  POST /reset with task_id='{task_id}'")
            
            # Generate the world (what reset() initializes)
            world = generate_world(task_id, seed=42)
            env = ShadowRegisterEnv(world)
            
            # Call reset() and get the response
            reset_result = env.reset()
            
            # Simulate JSON serialization of StepResult
            reset_response = {
                "observation": {
                    "current_view":      reset_result.observation.current_view,
                    "working_directory": reset_result.observation.working_directory,
                    "artifact_metadata": reset_result.observation.artifact_metadata.model_dump()
                                          if reset_result.observation.artifact_metadata else None,
                    "tagged_evidence":   reset_result.observation.tagged_evidence,
                    "remaining_budget":  reset_result.observation.remaining_budget,
                    "last_action_log":   reset_result.observation.last_action_log,
                },
                "reward": reset_result.reward,
                "done":   reset_result.done,
                "info":   reset_result.info,
            }
            
            # Try JSON serialization
            reset_json = json.dumps(reset_response)
            pass_test("/reset response serializable to JSON")
            
            # ================================================================
            # STEP 2: Simulate GET /state
            # ================================================================
            print(f"\n  2️⃣  GET /state")
            
            env_state = env.state()
            
            # Verify InternalState structure
            if not isinstance(env_state, InternalState):
                fail_test("env.state() didn't return InternalState")
                all_passed = False
                continue
            
            if env_state.truth_dag is None:
                fail_test("env.state().truth_dag is None")
                all_passed = False
                continue
            
            pass_test("env.state() returns InternalState with truth_dag",
                     f"scenario={env_state.truth_dag.scenario_name}")
            
            # Try to serialize the entire state to JSON
            state_dump = env_state.model_dump()
            state_json = json.dumps(state_dump)
            pass_test("InternalState fully serializable to JSON",
                     f"state size: {len(state_json)} bytes")
            
            # Verify truth_dag is in the serialized output
            state_parsed = json.loads(state_json)
            if 'truth_dag' not in state_parsed or state_parsed['truth_dag'] is None:
                fail_test("JSON InternalState missing truth_dag")
                all_passed = False
                continue
            
            pass_test("Serialized state includes truth_dag with all fields",
                     f"nodes: {len(state_parsed['truth_dag']['nodes'])}, "
                     f"edges: {len(state_parsed['truth_dag']['edges'])}")
            
            # Verify edges are lists (not tuples)
            for edge in state_parsed['truth_dag']['edges']:
                if not isinstance(edge, list):
                    fail_test(f"Edge {edge} is not a list - validator may reject")
                    all_passed = False
                    break
            else:
                pass_test("All edges are proper lists (JSON compatible)")
            
            # ================================================================
            # STEP 3: Simulate POST /step with SubmitCase
            # ================================================================
            print(f"\n  3️⃣  POST /step with SubmitCase action")
            
            submit_action = ForensicAction(
                action=ActionType.SUBMIT,
                pivots=[]
            )
            
            step_result = env.step(submit_action)
            
            if not step_result.done:
                fail_test("Episode didn't end after SubmitCase")
                all_passed = False
                continue
            
            pass_test("SubmitCase correctly ended episode")
            
            # ================================================================
            # STEP 4: Verify grader_report is attached
            # ================================================================
            print(f"\n  4️⃣  Grader Attachment Verification")
            
            # Check if grader is accessible
            if env.grader is None:
                fail_test("env.grader is None")
                all_passed = False
                continue
            
            if not callable(env.grader):
                fail_test("env.grader is not callable")
                all_passed = False
                continue
            
            pass_test("env.grader is attached and callable")
            
            # Simulate what /step endpoint does when done=True
            try:
                report = env.grader(
                    pivots=[],
                    truth=env.state().truth_dag,
                    remaining_budget=step_result.observation.remaining_budget
                )
                
                grader_dict = {
                    "score":     report.score,
                    "verdict":   report.verdict,
                    "breakdown": report.breakdown,
                    "penalties": report.penalties,
                    "bonuses":   report.bonuses,
                }
                
                grader_json = json.dumps(grader_dict)
                pass_test("Grader report generated and JSON serializable",
                         f"score={report.score:.4f}")
                
                # Final step response (what the API would return)
                step_response = {
                    "observation": {
                        "current_view":      step_result.observation.current_view,
                        "working_directory": step_result.observation.working_directory,
                        "artifact_metadata": step_result.observation.artifact_metadata.model_dump()
                                              if step_result.observation.artifact_metadata else None,
                        "tagged_evidence":   step_result.observation.tagged_evidence,
                        "remaining_budget":  step_result.observation.remaining_budget,
                        "last_action_log":   step_result.observation.last_action_log,
                    },
                    "reward": step_result.reward,
                    "done":   step_result.done,
                    "info":   {
                        "grader_report": grader_dict,
                        "score": report.score,
                    },
                }
                
                step_json = json.dumps(step_response)
                pass_test("Full /step response with grader_report serializable",
                         f"response size: {len(step_json)} bytes")
                
            except Exception as e:
                fail_test(f"Grader invocation failed: {e}")
                all_passed = False
                continue
            
            print(f"\n  ✅ Task '{task_id}' fully validated")
            
        except Exception as e:
            fail_test(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def main():
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + " FINAL INTEGRATION TEST ".center(78) + "║")
    print("║" + " API Endpoints with Grader Exposure ".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    passed = test_endpoint_flow()
    
    section("FINAL VERDICT")
    
    if passed:
        print("\n  🎉 ALL TESTS PASSED 🎉")
        print("\n  The environment should now pass Phase 2 validation.")
        print("  All three tasks have proper grader exposure and JSON serialization.\n")
        return True
    else:
        print("\n  ❌ SOME TESTS FAILED")
        print("\n  Please review the failures above.\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
