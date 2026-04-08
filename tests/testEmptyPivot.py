import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from grader import calculate_final_score
from schema import TruthDAG, TruthNode

dummy_truth = TruthDAG(
    scenario_name="test",
    seed=1,
    nodes={"A": TruthNode(node_id="A", required_artifact="a", expected_ioc="a", type="FILE_PATH", weight=1.0)},
    edges=[]
)

report = calculate_final_score(pivots=[], truth=dummy_truth, remaining_budget=0)
print(report)
print("Score:", report.score)
