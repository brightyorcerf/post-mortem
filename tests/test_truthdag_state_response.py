#!/usr/bin/env python3
"""
Validation Test: TruthDAG State Response Test
==============================================

Tests that GET /state endpoint returns a valid TruthDAG for all three tasks.

This is the specific test requested:
"Generate a test script that validates the presence of a TruthDAG in the /state 
response for all three scenarios."
"""

import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worldGen import generate_world, VALID_TASKS
from env import ShadowRegisterEnv
from schema import TruthDAG, InternalState


def test_trustdag_in_state_response():
    """
    Test that simulates GET /state endpoint for all three tasks.
    
    The /state endpoint in app.py does:
        env = _require_env()
        raw = env.state().model_dump()
        return JSONResponse(raw)
    
    This test verifies:
    1. env.state() returns InternalState
    2. InternalState.truth_dag is not None
    3. truth_dag is valid and contains expected fields
    4. model_dump() can serialize it
    5. JSON serialization works
    """
    
    print("\n" + "="*80)
    print("  TruthDAG State Response Validation Test")
    print("="*80)
    
    test_results = []
    
    for task_id in sorted(VALID_TASKS):
        print(f"\nTask: {task_id}")
        print("-" * 80)
        
        task_passed = True
        
        # Step 1: Generate world (simulates /reset preparation)
        try:
            world = generate_world(task_id, seed=42)
            print(f"  ✅ World generated")
        except Exception as e:
            print(f"  ❌ World generation failed: {e}")
            test_results.append((task_id, False))
            continue
        
        # Step 2: Create environment
        try:
            env = ShadowRegisterEnv(world)
            print(f"  ✅ Environment created")
        except Exception as e:
            print(f"  ❌ Environment creation failed: {e}")
            test_results.append((task_id, False))
            continue
        
        # Step 3: Simulate /state endpoint
        try:
            # Must call reset() first to initialize _state
            env.reset()
            state = env.state()
            
            # Verify it's InternalState
            if not isinstance(state, InternalState):
                print(f"  ❌ env.state() returned {type(state)}, not InternalState")
                task_passed = False
            else:
                print(f"  ✅ env.state() returns InternalState")
            
        except Exception as e:
            print(f"  ❌ env.state() call failed: {e}")
            test_results.append((task_id, False))
            continue
        
        # Step 4: Check truth_dag field exists
        try:
            if not hasattr(state, 'truth_dag'):
                print(f"  ❌ InternalState missing 'truth_dag' field")
                task_passed = False
            elif state.truth_dag is None:
                print(f"  ❌ InternalState.truth_dag is None")
                task_passed = False
            else:
                print(f"  ✅ InternalState.truth_dag exists")
        except Exception as e:
            print(f"  ❌ Error accessing truth_dag: {e}")
            task_passed = False
        
        # Step 5: Verify TruthDAG structure
        if task_passed and state.truth_dag:
            try:
                dag = state.truth_dag
                
                # Check fields
                if not hasattr(dag, 'scenario_name'):
                    print(f"  ❌ TruthDAG missing 'scenario_name'")
                    task_passed = False
                elif dag.scenario_name != task_id:
                    print(f"  ❌ scenario_name mismatch: {dag.scenario_name} != {task_id}")
                    task_passed = False
                else:
                    print(f"  ✅ TruthDAG.scenario_name = '{dag.scenario_name}'")
                
                if not hasattr(dag, 'nodes') or dag.nodes is None:
                    print(f"  ❌ TruthDAG missing 'nodes'")
                    task_passed = False
                else:
                    print(f"  ✅ TruthDAG.nodes = {len(dag.nodes)} nodes")
                    for node_id, node in dag.nodes.items():
                        print(f"     - {node_id}: {node.type.value}")
                
                if not hasattr(dag, 'edges') or dag.edges is None:
                    print(f"  ❌ TruthDAG missing 'edges'")
                    task_passed = False
                else:
                    print(f"  ✅ TruthDAG.edges = {dag.edges}")
                    # Verify edges are lists not tuples
                    for edge in dag.edges:
                        if not isinstance(edge, list):
                            print(f"     ❌ Edge {edge} is {type(edge)}, not list")
                            task_passed = False
                        else:
                            print(f"     ✅ Edge is list: {edge}")
                
                if not hasattr(dag, 'seed'):
                    print(f"  ❌ TruthDAG missing 'seed'")
                    task_passed = False
                else:
                    print(f"  ✅ TruthDAG.seed = {dag.seed}")
                    
            except Exception as e:
                print(f"  ❌ Error verifying TruthDAG structure: {e}")
                task_passed = False
        
        # Step 6: Test model_dump() serialization
        if task_passed:
            try:
                dumped = state.model_dump()
                if 'truth_dag' not in dumped:
                    print(f"  ❌ model_dump() doesn't include 'truth_dag'")
                    task_passed = False
                elif dumped['truth_dag'] is None:
                    print(f"  ❌ model_dump() has null 'truth_dag'")
                    task_passed = False
                else:
                    print(f"  ✅ model_dump() includes truth_dag")
            except Exception as e:
                print(f"  ❌ model_dump() failed: {e}")
                task_passed = False
        
        # Step 7: Test JSON serialization
        if task_passed:
            try:
                dumped = state.model_dump()
                json_str = json.dumps(dumped)
                print(f"  ✅ InternalState JSON serializable ({len(json_str)} bytes)")
                
                # Parse it back
                parsed = json.loads(json_str)
                if 'truth_dag' not in parsed:
                    print(f"  ❌ JSON parsing lost truth_dag")
                    task_passed = False
                elif parsed['truth_dag'] is None:
                    print(f"  ❌ Parsed JSON has null truth_dag")
                    task_passed = False
                else:
                    print(f"  ✅ JSON round-trip preserves truth_dag")
                    
                    # Verify edges structure in JSON
                    json_edges = parsed['truth_dag']['edges']
                    print(f"  ✅ Edges in JSON: {json_edges}")
                    for edge in json_edges:
                        if not isinstance(edge, list):
                            print(f"     ❌ JSON edge {edge} is not list")
                            task_passed = False
                    
            except Exception as e:
                print(f"  ❌ JSON serialization failed: {e}")
                task_passed = False
        
        # Summary for this task
        if task_passed:
            print(f"\n  🎉 {task_id}: PASSED")
            test_results.append((task_id, True))
        else:
            print(f"\n  ❌ {task_id}: FAILED")
            test_results.append((task_id, False))
    
    # Final summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for _, passed in test_results if passed)
    total_count = len(test_results)
    
    for task_id, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {task_id}")
    
    print(f"\nTotal: {passed_count}/{total_count} tasks passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED - TruthDAG is properly exposed in /state responses\n")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed\n")
        return False


if __name__ == "__main__":
    success = test_trustdag_in_state_response()
    sys.exit(0 if success else 1)
