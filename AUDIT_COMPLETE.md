# Phase 2 Grader Exposure Audit - Complete Solution

## Problem Statement
The environment was failing Phase 2 validation with error: **"Not enough tasks with graders"**

The validator could not discover the graders associated with each task because they were not exposed in the environment in a discoverable way.

---

## Solution Overview

Three focused changes were made to expose graders properly:

### 1. **Added Grader to Environment** (env.py)
- Attached `calculate_final_score` function to `ShadowRegisterEnv` as `_grader` attribute
- Exposed via `@property grader` for validator discovery
- Ensures grader is accessible: `env.grader()`

### 2. **Fixed TruthDAG Serialization** (schema.py)  
- Changed `edges: List[tuple[str, str]]` → `edges: List[List[str]]`
- Ensures JSON serialization compatibility
- Aligns with Pydantic v2 best practices

### 3. **Updated Edge Definitions** (worldGen.py)
- Updated 3 task builders to use lists instead of tuples
- Maintains consistency with schema definition
- `edges=[("A", "B")]` → `edges=[["A", "B"]]`

---

## Files Modified

### env.py
```python
# Added in __init__ (line 108)
from grader import calculate_final_score
self._grader = calculate_final_score

# Added property (lines 428-443)
@property
def grader(self):
    """Expose grader for Phase 2 validator discovery."""
    return getattr(self, "_grader", None)
```

### schema.py  
```python
# Line 95: Changed from
edges: List[tuple[str, str]]

# To
edges: List[List[str]]
```

### worldGen.py
```python
# Line 462: edges=[("A", "B")] → edges=[["A", "B"]]
# Line 632: edges=[("A", "B"), ("B", "C")] → edges=[["A", "B"], ["B", "C"]]
# Line 851: edges=[("A", "B"), ("B", "C")] → edges=[["A", "B"], ["B", "C"]]
```

---

## Validation Results

### Test 1: Phase 2 Validator Flow Simulation
✅ All 3 tasks have grader field in openenv.yaml
✅ Each task's grader is discoverable
✅ Each task's grader is callable and works

### Test 2: Full Integration Test (API Endpoint Flow)
For each task:
- ✅ POST /reset → StepResult (JSON serializable)
- ✅ GET /state → InternalState with truth_dag (not null)
- ✅ POST /step → grader_report attached and JSON serializable

### Test 3: TruthDAG State Response
✅ noisy_entry: 3 nodes, 1 edge (list format)
✅ stealthy_persistence: 4 nodes, 2 edges (list format)
✅ timestomp_proxy: 5 nodes, 2 edges (list format)

All edges verified as lists (JSON compatible), not tuples.

---

## Key Verification Points

| Check | Status | Evidence |
|-------|--------|----------|
| Grader attached to env | ✅ | `env.grader` returns callable |
| TruthDAG in /state | ✅ | `env.state().truth_dag` is not null |
| JSON serializable | ✅ | `json.dumps(env.state().model_dump())` works |
| Edges as lists | ✅ | `edges = [['A', 'B']]` not `[('A', 'B')]` |
| All 3 tasks work | ✅ | test_truthdag_state_response.py passes |
| No breaking changes | ✅ | All prior tests still pass |

---

## Test Scripts Provided

1. **tests/test_phase2_validator_flow.py**
   - Simulates Phase 2 validator discovery process
   - Verifies all 3 tasks have graders

2. **tests/final_integration_test.py** ⭐ (Most Comprehensive)
   - Full HTTP endpoint flow simulation
   - All passing: `🎉 ALL TESTS PASSED`

3. **tests/test_truthdag_state_response.py** ⭐ (User Requested)
   - Specific test for TruthDAG in /state responses
   - All passing: `🎉 ALL TESTS PASSED`

---

## How to Run Tests

```bash
# Full integration test
python tests/final_integration_test.py

# TruthDAG state response test (specific request)
python tests/test_truthdag_state_response.py

# Phase 2 validator flow simulation
python tests/test_phase2_validator_flow.py
```

All tests should show: ✅ ALL TESTS PASSED

---

## Impact Assessment

### What Changed
- ✅ Graders are now discoverable
- ✅ JSON serialization is robust
- ✅ All 3 tasks properly expose graders

### What Stayed the Same
- ✅ No API contract changes
- ✅ No scoring logic changes
- ✅ No inference script changes required
- ✅ Backward compatible

### Risk Level: LOW
- Changes are minimal (~20 lines)
- Changes are focused on exposure/serialization
- No core logic modifications
- Existing tests still pass

---

## Next Steps

1. Run the integration tests to verify:
   ```bash
   python tests/final_integration_test.py
   ```

2. Submit to Phase 2 validator - should now pass "grader discovery" check

3. If validator still reports issues, the three comprehensive test scripts
   can be used to diagnose and pinpoint any remaining issues

---

## Summary

The Phase 2 validator error **"Not enough tasks with graders"** has been resolved by:

1. **Explicitly attaching graders** to the environment class
2. **Fixing JSON serialization** of TruthDAG edges (tuples → lists)
3. **Maintaining consistency** across all three tasks

All three tasks now properly expose their graders with complete JSON serialization compatibility.

**Status: READY FOR PHASE 2 VALIDATION** ✅
