"""
grader.py  —  SHADOW_REGISTER Scoring Referee
==============================================
Calculates the final score for a SubmitCase action by comparing the
agent's ForensicPivot list against the TruthDAG.

Public API
----------
    calculate_final_score(
        pivots:           List[ForensicPivot],
        truth:            TruthDAG,
        remaining_budget: int = 0,
    ) -> GraderReport

GraderReport fields
-------------------
    score         float   clamped to [0.0, 1.0]
    breakdown     dict    per-node match result
    penalties     list    triggered penalty descriptions
    bonuses       list    triggered bonus descriptions
    verdict       str     human-readable summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from schema import ForensicPivot, IOCType, TruthDAG, TruthNode

# ---------------------------------------------------------------------------
# Scoring constants  (tune these without touching logic)
# ---------------------------------------------------------------------------

HONEYPOT_PENALTY      = -0.40
EFFICIENCY_BONUS      = +0.10
EFFICIENCY_THRESHOLD  = 0.40    # fraction of BUDGET_MAX remaining
BUDGET_MAX            = 50      # must match env.py
IOC_FUZZY_STRIP_CHARS = " \t\r\n\"'`"  # strip these before comparison


# ---------------------------------------------------------------------------
# GraderReport
# ---------------------------------------------------------------------------

@dataclass
class GraderReport:
    score:     float
    breakdown: Dict[str, dict]   = field(default_factory=dict)
    penalties: List[str]         = field(default_factory=list)
    bonuses:   List[str]         = field(default_factory=list)
    verdict:   str               = ""

    def __str__(self) -> str:
        lines = [
            f"{'═' * 52}",
            f"  SHADOW_REGISTER // Grader Report",
            f"{'═' * 52}",
            f"  Final Score : {self.score:.4f}",
            f"{'─' * 52}",
            "  Node Results:",
        ]
        for node_id, result in self.breakdown.items():
            status = "✓ HIT " if result["matched"] else "✗ MISS"
            hp     = " [HONEYPOT]" if result.get("is_honeypot") else ""
            lines.append(
                f"    {status}  {node_id}{hp}"
                f"  weight={result['weight']:.2f}"
                f"  contribution={result['contribution']:+.4f}"
            )
        if self.penalties:
            lines += ["", "  Penalties:"] + [f"    • {p}" for p in self.penalties]
        if self.bonuses:
            lines += ["", "  Bonuses:"]   + [f"    • {b}" for b in self.bonuses]
        lines += ["", f"  Verdict: {self.verdict}", f"{'═' * 52}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IOC matching
# ---------------------------------------------------------------------------

def _normalise(ioc: str) -> str:
    """Strip whitespace/quotes for tolerant comparison."""
    return ioc.strip(IOC_FUZZY_STRIP_CHARS).lower()


def _ioc_matches(submitted: str, expected: str, ioc_type: IOCType) -> bool:
    """
    Type-aware IOC comparison.

    NETWORK_IP        — exact match after normalisation
    EVENT_TIMESTAMP   — exact match (ISO 8601 string from the DAG)
                        OR agent may submit just the timestamp part of the
                        discrepancy string for the hard task
    FILE_PATH         — exact match; agent may omit leading slash
    COMMAND_STRING    — substring match (base64 strings are long)
    USER_ACCOUNT      — exact after normalisation
    FILE_HASH         — exact after normalisation
    """
    sub = _normalise(submitted)
    exp = _normalise(expected)

    if ioc_type == IOCType.COMMAND_STRING:
        # Accept if the submitted value is contained in the expected (or vice-versa)
        return sub in exp or exp in sub

    if ioc_type == IOCType.FILE_PATH:
        # Tolerate missing leading slash
        return sub.lstrip("/") == exp.lstrip("/")

    if ioc_type == IOCType.EVENT_TIMESTAMP:
        # Agent may submit the full discrepancy string or just one of the timestamps
        return sub == exp or sub in exp

    # Default: exact match
    return sub == exp


def _pivot_matches_node(pivot: ForensicPivot, node: TruthNode) -> bool:
    """
    A pivot matches a TruthNode if BOTH:
      1. The artifact path points to the correct file.
      2. The IOC value matches the expected IOC.
    """
    artifact_ok = _normalise(pivot.artifact).lstrip("/") == _normalise(node.required_artifact).lstrip("/")
    ioc_ok      = _ioc_matches(pivot.ioc, node.expected_ioc, node.type)
    return artifact_ok and ioc_ok


# ---------------------------------------------------------------------------
# DAG Chain Validation
# ---------------------------------------------------------------------------

def _validate_chain(
    matched_node_ids: set,
    truth: TruthDAG,
) -> float:
    """
    Check that matched nodes satisfy the DAG ordering.
    If an agent matched node B but NOT node A (where A→B), the chain is
    broken and node B's weight contribution is halved.

    Returns a chain-validity multiplier between 0.5 and 1.0.
    """
    multiplier = 1.0
    for (src, dst) in truth.edges:
        # If dst was matched but src was not → broken chain
        if dst in matched_node_ids and src not in matched_node_ids:
            multiplier *= 0.5  # halve for each broken link
    return max(multiplier, 0.25)  # floor at 25% — partial credit


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def calculate_final_score(
    pivots:           List[ForensicPivot],
    truth:            TruthDAG,
    remaining_budget: int = 0,
) -> GraderReport:
    """
    Compare agent's pivots against the TruthDAG and produce a GraderReport.

    Scoring logic (in order)
    -------------------------
    1. For each non-honeypot TruthNode, check if any submitted pivot matches.
       Accumulate weighted score.
    2. For each pivot that matches a HONEYPOT node: subtract HONEYPOT_PENALTY.
    3. Apply DAG-chain multiplier (broken prerequisite chains reduce credit).
    4. Add efficiency bonus if remaining_budget / BUDGET_MAX >= threshold.
    5. Clamp to [0.0, 1.0].

    Parameters
    ----------
    pivots           : Agent's submitted ForensicPivot list
    truth            : TruthDAG (from InternalState.truth_dag)
    remaining_budget : How many budget units remain when SubmitCase was called
    """

    _supported = ['noisy_entry', 'stealthy_persistence', 'timestomp_proxy']
    report = GraderReport(score=0.0)

    if not pivots:
        report.verdict = "No pivots submitted — score 0."
        return report

    # ------------------------------------------------------------------ #
    # 1. Per-node weighted scoring                                         #
    # ------------------------------------------------------------------ #
    raw_score       = 0.0
    matched_ids     = set()
    honeypot_hits   = 0

    for node_id, node in truth.nodes.items():
        # Find the best matching pivot for this node
        matched = any(_pivot_matches_node(p, node) for p in pivots)

        if node.is_honeypot:
            if matched:
                honeypot_hits += 1
                penalty_val = HONEYPOT_PENALTY
                report.breakdown[node_id] = {
                    "matched":      True,
                    "is_honeypot":  True,
                    "weight":       node.weight,
                    "contribution": penalty_val,
                }
                report.penalties.append(
                    f"Honeypot '{node.required_artifact}' tagged  →  {penalty_val:.2f}"
                )
                raw_score += penalty_val
            # Honeypot not hit — no entry needed, no penalty
            continue

        contribution = node.weight if matched else 0.0
        raw_score   += contribution

        if matched:
            matched_ids.add(node_id)

        report.breakdown[node_id] = {
            "matched":      matched,
            "is_honeypot":  False,
            "weight":       node.weight,
            "contribution": contribution,
        }

    # ------------------------------------------------------------------ #
    # 2. DAG chain validation                                              #
    # ------------------------------------------------------------------ #
    # Only apply the chain multiplier to the *positive* portion of the score
    positive_score = sum(
        v["contribution"] for v in report.breakdown.values()
        if v["contribution"] > 0 and not v.get("is_honeypot")
    )
    negative_score = raw_score - positive_score

    chain_mult = _validate_chain(matched_ids, truth)
    if chain_mult < 1.0:
        broken_links = [
            f"{src}→{dst}"
            for (src, dst) in truth.edges
            if dst in matched_ids and src not in matched_ids
        ]
        report.penalties.append(
            f"Broken kill-chain prerequisite(s): {', '.join(broken_links)}  "
            f"→  chain multiplier {chain_mult:.2f}x"
        )

    adjusted_positive = positive_score * chain_mult
    score = adjusted_positive + negative_score   # negatives are not multiplied

    # ------------------------------------------------------------------ #
    # 3. Efficiency bonus                                                  #
    # ------------------------------------------------------------------ #
    efficiency_ratio = remaining_budget / BUDGET_MAX
    if efficiency_ratio >= EFFICIENCY_THRESHOLD and score > 0:
        score += EFFICIENCY_BONUS
        report.bonuses.append(
            f"Efficiency bonus: {remaining_budget}/{BUDGET_MAX} budget remaining "
            f"({efficiency_ratio:.0%} ≥ {EFFICIENCY_THRESHOLD:.0%})  "
            f"→  +{EFFICIENCY_BONUS:.2f}"
        )

    # ------------------------------------------------------------------ #
    # 4. Clamp & verdict                                                   #
    # ------------------------------------------------------------------ #
    report.score = round(max(0.0, min(score, 1.0)), 6)

    matched_count = len(matched_ids)
    total_truth   = sum(1 for n in truth.nodes.values() if not n.is_honeypot)
    report.verdict = _compose_verdict(
        score=report.score,
        matched=matched_count,
        total=total_truth,
        honeypot_hits=honeypot_hits,
        chain_mult=chain_mult,
    )

    return report


# ---------------------------------------------------------------------------
# Verdict helper
# ---------------------------------------------------------------------------

def _compose_verdict(
    score: float,
    matched: int,
    total: int,
    honeypot_hits: int,
    chain_mult: float,
) -> str:
    hp_note    = f"  Triggered {honeypot_hits} honeypot(s)." if honeypot_hits else ""
    chain_note = f"  Chain broken ({chain_mult:.2f}x)." if chain_mult < 1.0 else ""

    if score >= 0.95:
        grade = "PERFECT — Full Kill Chain reconstructed."
    elif score >= 0.70:
        grade = f"STRONG — {matched}/{total} truth nodes resolved."
    elif score >= 0.40:
        grade = f"PARTIAL — {matched}/{total} truth nodes resolved."
    elif score > 0.0:
        grade = f"WEAK — only {matched}/{total} truth nodes resolved."
    else:
        grade = "FAILED — no valid evidence submitted."

    return f"{grade}{hp_note}{chain_note}"


# ---------------------------------------------------------------------------
# Convenience wrapper for env.py integration
# ---------------------------------------------------------------------------

def grade_submission(
    env_instance,            # ShadowRegisterEnv (avoids circular import)
    remaining_budget: int,
) -> GraderReport:
    """
    Pull pivots directly from a finished ShadowRegisterEnv and grade them.
    Call this after env.step(SubmitCase(...)) returns done=True.
    """
    pivots = env_instance.last_pivots
    truth  = env_instance.state().truth_dag
    return calculate_final_score(pivots, truth, remaining_budget)