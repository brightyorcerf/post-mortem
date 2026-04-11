#!/usr/bin/env python3
"""
Comprehensive validation that all three tasks expose their graders correctly
to the Phase 2 validator.

Tests:
1. Serialization: All TruthDAG instances serialize to JSON without error
2. API Structure: /state endpoint returns InternalState with non-null truth_dag
3. Grader Integration: /step endpoint with SubmitCase includes grader_report
4. Task Discovery: openenv.yaml defines all three tasks with grader field
"""

import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from schema import (
    ForensicAction, 
    ForensicPivot, 
    IOCType, 
    InternalState, 
    TruthDAG,
    ActionType,
)
from worldGen import generate_world, VALID_TASKS
from env import ShadowRegisterEnv
from grader import calculate_final_score
import yaml

def print_section(title: str):
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

def warn_test(msg: str, detail: str = ""):
    print(f"  ⚠️  {msg}")
    if detail:
        print(f"     {detail}")

# ============================================================================
# TEST 1: TruthDAG Serialization
# ============================================================================

def test_truthdag_serialization():
    print_section("TEST 1: TruthDAG Serialization & JSON Compatibility")
    
    for task in sorted(VALID_TASKS):
        try:
            # Generate world
            state = generate_world(task, seed=42)
            truth_dag = state.truth_dag
            
            # Test 1a: TruthDAG is not None
            if truth_dag is None:
                fail_test(f"{task}: truth_dag is None")
                continue
            pass_test(f"{task}: truth_dag exists", f"scenario={truth_dag.scenario_name}")
            
            # Test 1b: model_dump() works
            try:
                dumped = truth_dag.model_dump()
                pass_test(f"  - model_dump() works", f"edges: {dumped.get('edges')}")
            except Exception as e:
                fail_test(f"  - model_dump() failed: {e}")
                continue
            
            # Test 1c: JSON serialization works
            try:
                json_str = json.dumps(dumped)
                pass_test(f"  - JSON serializable")
                # Verify we can parse it back
                reparsed = json.loads(json_str)
                if reparsed['edges'] == dumped['edges']:
                    pass_test(f"  - Round-trip JSON OK")
                else:
                    fail_test(f"  - JSON round-trip mismatch")
            except Exception as e:
                fail_test(f"  - JSON serialization failed: {e}")
                
        except Exception as e:
            fail_test(f"{task}: Generation failed", str(e))


# ============================================================================
# TEST 2: InternalState Serialization
# ============================================================================

def test_internalstate_serialization():
    print_section("TEST 2: InternalState Serialization (should include TruthDAG)")
    
    for task in sorted(VALID_TASKS):
        try:
            state = generate_world(task, seed=42)
            
            # Test 2a: InternalState has truth_dag field
            if not hasattr(state, 'truth_dag'):
                fail_test(f"{task}: InternalState missing truth_dag field")
                continue
            pass_test(f"{task}: InternalState.truth_dag exists")
            
            # Test 2b: model_dump() works
            try:
                dumped = state.model_dump()
                if 'truth_dag' not in dumped or dumped['truth_dag'] is None:
                    fail_test(f"  - model_dump() missing/null truth_dag")
                    continue
                pass_test(f"  - model_dump() includes truth_dag", 
                         f"dag_scenario={dumped['truth_dag'].get('scenario_name')}")
            except Exception as e:
                fail_test(f"  - model_dump() failed: {e}")
                continue
            
            # Test 2c: JSON serialization of full InternalState
            try:
                json_str = json.dumps(dumped)
                reparsed = json.loads(json_str)
                if reparsed.get('truth_dag'):
                    pass_test(f"  - Full InternalState JSON serializable")
                else:
                    fail_test(f"  - JSON InternalState missing truth_dag")
            except Exception as e:
                fail_test(f"  - Full InternalState JSON failed: {e}")
                
        except Exception as e:
            fail_test(f"{task}: Generation failed", str(e))


# ============================================================================
# TEST 3: Environment State Endpoint Simulation
# ============================================================================

def test_env_state_method():
    print_section("TEST 3: Environment.state() Method (simulates /state endpoint)")
    
    for task in sorted(VALID_TASKS):
        try:
            state = generate_world(task, seed=42)
            env = ShadowRegisterEnv(state)
            
            # Simulate what /state endpoint does
            env_state = env.state()
            
            if env_state.truth_dag is None:
                fail_test(f"{task}: env.state().truth_dag is None")
                continue
            
            pass_test(f"{task}: env.state() returns InternalState with truth_dag")
            
            # Try to serialize for JSON response
            try:
                serialized = env_state.model_dump()
                json_str = json.dumps(serialized)
                pass_test(f"  - Serialization for /state response OK")
            except Exception as e:
                fail_test(f"  - Serialization failed: {e}")
                
        except Exception as e:
            fail_test(f"{task}: Test failed", str(e))


# ============================================================================
# TEST 4: Grader Integration
# ============================================================================

def test_grader_integration():
    print_section("TEST 4: Grader Integration (simulate /step SubmitCase)")
    
    for task in sorted(VALID_TASKS):
        try:
            state = generate_world(task, seed=42)
            env = ShadowRegisterEnv(state)
            env.reset()
            
            # Simulate an empty submission (worst case)
            submit_action = ForensicAction(
                action=ActionType.SUBMIT,
                pivots=[]
            )
            
            result = env.step(submit_action)
            
            # Check that grader_report is attempted (even if score is 0)
            if result.done:
                internal_state = env.state()
                truth_dag = internal_state.truth_dag
                
                # Manually calculate grader report (simulating /step endpoint)
                try:
                    report = calculate_final_score(
                        pivots=[],
                        truth=truth_dag,
                        remaining_budget=result.observation.remaining_budget,
                    )
                    pass_test(f"{task}: Grader calculates score", 
                             f"score={report.score:.4f}")
                    
                    # Test serialization of grader report
                    report_dict = {
                        "score":     report.score,
                        "verdict":   report.verdict,
                        "breakdown": report.breakdown,
                        "penalties": report.penalties,
                        "bonuses":   report.bonuses,
                    }
                    json_str = json.dumps(report_dict)
                    pass_test(f"  - Grader report JSON serializable")
                except Exception as e:
                    fail_test(f"  - Grader calculation/serialization failed: {e}")
            else:
                fail_test(f"{task}: episode did not end after SubmitCase")
                
        except Exception as e:
            fail_test(f"{task}: Test failed", str(e))


# ============================================================================
# TEST 5: OpenEnv YAML Task Discovery
# ============================================================================

def test_openenv_yaml_tasks():
    print_section("TEST 5: OpenEnv YAML Task Registration")
    
    yaml_path = root_dir / "openenv.yaml"
    if not yaml_path.exists():
        fail_test("openenv.yaml not found")
        return
    
    try:
        with open(yaml_path) as f:
            spec = yaml.safe_load(f)
    except Exception as e:
        fail_test(f"Failed to parse openenv.yaml: {e}")
        return
    
    pass_test("openenv.yaml parses successfully")
    
    # Check tasks section
    if "tasks" not in spec:
        fail_test("openenv.yaml missing 'tasks' section")
        return
    
    tasks_in_yaml = {t['id']: t for t in spec['tasks']}
    pass_test(f"Found {len(tasks_in_yaml)} task(s) in openenv.yaml")
    
    # Check each expected task
    for expected_task in sorted(VALID_TASKS):
        if expected_task not in tasks_in_yaml:
            fail_test(f"Task '{expected_task}' missing from openenv.yaml")
            continue
        
        task_spec = tasks_in_yaml[expected_task]
        
        # Check for grader field
        if 'grader' not in task_spec:
            fail_test(f"  {expected_task}: 'grader' field missing")
        elif task_spec['grader']:
            pass_test(f"  {expected_task}: grader field = '{task_spec['grader']}'")
        else:
            fail_test(f"  {expected_task}: grader field is empty")
    
    # Check for extra/unknown tasks
    extra_tasks = set(tasks_in_yaml.keys()) - VALID_TASKS
    if extra_tasks:
        warn_test(f"Extra tasks in openenv.yaml: {extra_tasks}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + " SHADOW_REGISTER: Grader Exposure Validation ".center(78) + "║")
    print("║" + " Phase 2 Validator Audit ".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    test_truthdag_serialization()
    test_internalstate_serialization()
    test_env_state_method()
    test_grader_integration()
    test_openenv_yaml_tasks()
    
    print_section("VALIDATION COMPLETE")
    print("\n✅ If all tests pass above, the grader exposure is correct.\n")
