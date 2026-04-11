#!/usr/bin/env python3
"""
Simulate what a Phase 2 OpenEnv validator might do when checking
"Not enough tasks with graders".

Flow:
1. Parse openenv.yaml to find tasks
2. For each task:
   a. Call /reset with that task_id
   b. Check if grader information is accessible
   c. Submit a sample action and verify grader_report is returned
"""

import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from schema import ForensicAction, ActionType, ForensicPivot, IOCType
from worldGen import generate_world, VALID_TASKS
from env import ShadowRegisterEnv
from grader import calculate_final_score
import yaml


def section(title:str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def check_pass(msg: str, detail: str = ""):
    print(f"  ✅ {msg}")
    if detail:
        print(f"     {detail}")
    return True


def check_fail(msg: str, detail: str = ""):
    print(f"  ❌ {msg}")
    if detail:
        print(f"     {detail}")
    return False


def main():
    section("PHASE 2 VALIDATOR SIMULATION: Grader Discovery")
    
    # Step 1: Load and parse openenv.yaml
    section("STEP 1: Parse openenv.yaml")
    
    yaml_path = root_dir / "openenv.yaml"
    try:
        with open(yaml_path) as f:
            spec = yaml.safe_load(f)
        check_pass("openenv.yaml loaded")
    except Exception as e:
        check_fail(f"Failed to parse openenv.yaml: {e}")
        return False
    
    # Extract tasks
    if "tasks" not in spec:
        check_fail("openenv.yaml missing 'tasks' field")
        return False
    
    tasks_in_yaml = {t['id']: t for t in spec['tasks']}
    check_pass(f"Found {len(tasks_in_yaml)} task(s)",
              f"ids: {list(tasks_in_yaml.keys())}")
    
    # Step 2: Verify each task has a grader field
    section("STEP 2: Check Tasks Have Grader Field")
    
    tasks_with_graders = 0
    
    for task_id, task_spec in tasks_in_yaml.items():
        if 'grader' not in task_spec:
            check_fail(f"{task_id}: missing 'grader' field")
            continue
        
        grader_value = task_spec.get('grader')
        if not grader_value:
            check_fail(f"{task_id}: grader field is empty", f"value: '{grader_value}'")
            continue
        
        check_pass(f"{task_id}: has grader field", f"value: '{grader_value}'")
        tasks_with_graders += 1
    
    if tasks_with_graders < len(tasks_in_yaml):
        print(f"\n  ⚠️  WARNING: Only {tasks_with_graders}/{len(tasks_in_yaml)} tasks have graders!")
        print(f"       Validator threshold probably requires: >= {len(tasks_in_yaml)}")
        # This mightbe the actual error!
    
    # Step 3: For each task, simulate environment initialization and grader validation
    section("STEP 3: Environment & Grader Initialization")
    
    grader_success_count = 0
    
    for task_id in list(tasks_in_yaml.keys()):
        print(f"\n  Testing task: '{task_id}'")
        if task_id not in VALID_TASKS:
            check_fail(f"    Task '{task_id}' not in VALID_TASKS")
            continue
        
        try:
            # Simulate /reset endpoint
            state = generate_world(task_id, seed=42)
            check_pass(f"    generate_world() works")
            
            # Check TruthDAG exists
            if state.truth_dag is None:
                check_fail(f"    TruthDAG is None")
                continue
            check_pass(f"    TruthDAG exists", 
                      f"{len(state.truth_dag.nodes)} nodes")
            
            # Initialize environment (simulates FastAPI storing env in _session)
            env = ShadowRegisterEnv(state)
            reset_result = env.reset()
            check_pass(f"    env.reset() works")
            
            # Check if we can access the state (simulates /state endpoint)
            env_state = env.state()
            if env_state.truth_dag is None:
                check_fail(f"    env.state().truth_dag is None (BLOCKER!)")
                continue
            check_pass(f"    env.state() includes TruthDAG")
            
            # Simulate /step with SubmitCase (empty submission)
            submit_action = ForensicAction(
                action=ActionType.SUBMIT,
                pivots=[]
            )
            step_result = env.step(submit_action)
            
            # Check if grader_report can be generated
            if step_result.done:
                try:
                    truth_dag = env.state().truth_dag
                    report = calculate_final_score(
                        pivots=[],
                        truth=truth_dag,
                        remaining_budget=step_result.observation.remaining_budget
                    )
                    check_pass(f"    Grader works", f"score={report.score:.4f}")
                    
                    # Check if we can serialize it (simulates JSON response)
                    grader_dict = {
                        "score": report.score,
                        "verdict": report.verdict,
                        "breakdown": report.breakdown,
                        "penalties": report.penalties,
                        "bonuses": report.bonuses,
                    }
                    json_str = json.dumps(grader_dict)
                    check_pass(f"    Grader report JSON serializable")
                    
                    grader_success_count += 1
                except Exception as e:
                    check_fail(f"    Grader calculation failed: {e}")
            else:
                check_fail(f"    Episode didn't end after SubmitCase")
                
        except Exception as e:
            check_fail(f"    Initialization failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Final verdict
    section("FINAL VERDICT")
    
    if tasks_with_graders == len(tasks_in_yaml):
        check_pass(f"All {len(tasks_in_yaml)} tasks have grader field in YAML")
    else:
        check_fail(f"Not enough tasks with graders: {tasks_with_graders}/{len(tasks_in_yaml)}")
        print("\n  🔴 THIS WOULD CAUSE VALIDATOR ERROR: 'Not enough tasks with graders'")
        return False
    
    if grader_success_count == len(tasks_in_yaml):
        check_pass(f"All {len(tasks_in_yaml)} tasks have working graders")
    else:
        check_fail(f"Only {grader_success_count}/{len(tasks_in_yaml)} graders successful")
        print("\n  🔴 THIS MIGHT CAUSE VALIDATOR ERROR")
        return False
    
    check_pass("✅ Phase 2 Validator simulation PASSED")
    print("\n  The environment should pass Phase 2 grader validation.\n")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
