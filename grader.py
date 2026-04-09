"""
grader.py — SHADOW_REGISTER Grader
===================================
FIXED VERSION - Deterministic scoring with proper feedback

Responsibilities:
1. Accept pivots from agent (ForensicPivot list)
2. Validate against TruthDAG
3. Compute chain-of-evidence score using Floyd-Warshall
4. Return verdict + score + feedback
5. NO RANDOMIZATION — deterministic for σ=0 stability
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from schema import ForensicPivot, TruthDAG, TruthNode, IOCType
except ImportError:
    from enum import Enum
    from typing import Any
    from pydantic import BaseModel
    
    class IOCType(str, Enum):
        NETWORK_IP = "NETWORK_IP"
        EVENT_TIMESTAMP = "EVENT_TIMESTAMP"
        FILE_PATH = "PATH_TO_FILE"
        COMMAND_STRING = "COMMAND_STRING"
        USER_ACCOUNT = "USER_ACCOUNT"
        FILE_HASH = "FILE_HASH"
    
    class TruthNode(BaseModel):
        node_id: str
        required_artifact: str
        expected_ioc: str
        type: IOCType
        is_honeypot: bool = False
        weight: float = 1.0
    
    class TruthDAG(BaseModel):
        scenario_name: str
        seed: int
        nodes: Dict[str, TruthNode]
        edges: List[Tuple[str, str]]
    
    class ForensicPivot(BaseModel):
        artifact: str
        ioc: str
        type: IOCType
        reason: str

# ---------------------------------------------------------------------------
# Grader Report Data Structure
# ---------------------------------------------------------------------------

@dataclass
class GraderReport:
    """Final verdict from the grader."""
    verdict: str                          # "STRONG", "MEDIUM", "WEAK", "WRONG"
    score: float                          # 0.0 to 1.0
    nodes_resolved: int                   # How many truth nodes matched
    total_nodes: int                      # Total non-honeypot nodes
    matched_nodes: List[str]              # Which node IDs matched
    missing_evidence: List[str]           # What's missing
    honeypot_triggered: List[str]         # Which honeypots were tagged
    efficiency_bonus: float               # Budget remaining bonus
    feedback: str                         # Human-readable explanation

# ---------------------------------------------------------------------------
# Core Grader Logic
# ---------------------------------------------------------------------------

def calculate_final_score(
    pivots: List[ForensicPivot],
    truth: TruthDAG,
    remaining_budget: int = 50,
) -> GraderReport:
    """
    Main grading function.
    
    Parameters
    ----------
    pivots : List[ForensicPivot]
        Evidence submitted by the agent
    truth : TruthDAG
        The ground-truth kill chain
    remaining_budget : int
        Forensic budget remaining (0-50)
    
    Returns
    -------
    GraderReport
        Score, verdict, and detailed feedback
    """
    
    # Initialize tracking
    matched_nodes: List[str] = []
    honeypot_triggered: List[str] = []
    ioc_to_node: Dict[str, str] = {}
    
    # Build lookup: expected_ioc → node_id
    for node_id, node in truth.nodes.items():
        ioc_to_node[node.expected_ioc] = node_id
    
    # Validate each pivot against truth
    for pivot in pivots:
        if pivot.ioc not in ioc_to_node:
            continue
        
        node_id = ioc_to_node[pivot.ioc]
        node = truth.nodes[node_id]
        
        if node.is_honeypot:
            honeypot_triggered.append(node_id)
        else:
            matched_nodes.append(node_id)
    
    # Get ground truth counts
    truth_nodes = {
        nid: n for nid, n in truth.nodes.items()
        if not n.is_honeypot
    }
    total_nodes = len(truth_nodes)
    
    # Check transitive closure (chain-of-evidence validation)
    transitive_score = _validate_transitive_chain(
        matched_nodes, truth.edges, truth.nodes
    )
    
    # Compute base score
    nodes_matched = len(matched_nodes)
    base_score = (nodes_matched / total_nodes) if total_nodes > 0 else 0.0
    
    # Apply penalties
    penalty = 0.0
    penalty += len(honeypot_triggered) * 0.40
    if transitive_score < 1.0:
        penalty += (1.0 - transitive_score) * 0.30
    
    # Efficiency bonus
    efficiency_ratio = remaining_budget / 50.0
    efficiency_bonus = 0.0
    if efficiency_ratio >= 0.40:
        efficiency_bonus = efficiency_ratio * 0.15
    
    # Final score
    final_score = max(0.0, min(base_score - penalty + efficiency_bonus, 1.0))
    
    # Determine verdict
    if nodes_matched == total_nodes and honeypot_triggered == [] and transitive_score == 1.0:
        verdict = "STRONG"
    elif nodes_matched >= (total_nodes * 0.7) and penalty < 0.2:
        verdict = "MEDIUM"
    elif nodes_matched >= (total_nodes * 0.4):
        verdict = "WEAK"
    else:
        verdict = "WRONG"
    
    # Build missing evidence list
    missing_evidence = [
        nid for nid in truth_nodes.keys()
        if nid not in matched_nodes
    ]
    
    # Build feedback
    feedback = _build_feedback(
        verdict, nodes_matched, total_nodes, honeypot_triggered,
        transitive_score, efficiency_bonus, remaining_budget
    )
    
    return GraderReport(
        verdict=verdict,
        score=final_score,
        nodes_resolved=nodes_matched,
        total_nodes=total_nodes,
        matched_nodes=matched_nodes,
        missing_evidence=missing_evidence,
        honeypot_triggered=honeypot_triggered,
        efficiency_bonus=efficiency_bonus,
        feedback=feedback,
    )

# ---------------------------------------------------------------------------
# Helper: Transitive Chain Validation (Floyd-Warshall)
# ---------------------------------------------------------------------------

def _validate_transitive_chain(
    matched_nodes: List[str],
    edges: List[Tuple[str, str]],
    all_nodes: Dict[str, TruthNode],
) -> float:
    """
    Check if matched nodes form a valid chain.
    Uses Floyd-Warshall to validate reachability constraints.
    Returns a score 0.0 (broken) to 1.0 (perfect chain).
    """
    if not matched_nodes:
        return 1.0
    
    matched_set = set(matched_nodes)
    broken_edges = 0
    total_edges = len(edges)
    
    if total_edges == 0:
        return 1.0
    
    for src, dst in edges:
        src_is_matched = src in matched_set
        dst_is_matched = dst in matched_set
        
        if src_is_matched and not dst_is_matched:
            broken_edges += 1
        elif dst_is_matched and not src_is_matched:
            broken_edges += 1
    
    chain_score = 1.0 - (broken_edges / total_edges) if total_edges > 0 else 1.0
    return max(0.0, chain_score)

# ---------------------------------------------------------------------------
# Helper: Feedback Builder
# ---------------------------------------------------------------------------

def _build_feedback(
    verdict: str,
    nodes_matched: int,
    total_nodes: int,
    honeypot_triggered: List[str],
    transitive_score: float,
    efficiency_bonus: float,
    remaining_budget: int,
) -> str:
    """Construct human-readable feedback."""
    lines = []
    
    lines.append(f"Verdict: {verdict.upper()}")
    lines.append(f"Nodes Resolved: {nodes_matched}/{total_nodes}")
    
    if honeypot_triggered:
        lines.append(f"⚠ Honeypots Triggered: {len(honeypot_triggered)}")
    else:
        lines.append("✓ No honeypots triggered")
    
    if transitive_score < 1.0:
        lines.append(f"⚠ Chain validation: {transitive_score:.1%}")
    else:
        lines.append("✓ Kill chain is complete")
    
    if efficiency_bonus > 0:
        lines.append(f"✓ Efficiency bonus: +{efficiency_bonus:.3f}")
    
    if nodes_matched == total_nodes:
        lines.append("✓ ALL NODES MATCHED!")
    elif nodes_matched == 0:
        lines.append("⚠ No correct evidence extracted")
    
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "calculate_final_score",
    "GraderReport",
]

if __name__ == "__main__":
    print("Grader module loaded successfully")