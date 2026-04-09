"""
env.py — SHADOW_REGISTER OpenEnv Environment
================================================
Systems Architect: Game-loop engine wrapping the virtual filesystem.

FIXED VERSION - Includes:
1. ✓ Grader integration in _handle_submit()
2. ✓ Reward milestone for correct Tags
3. ✓ Search results in ForensicObs
4. ✓ Increased current_view buffer

Implements the OpenEnv standard:
reset() → StepResult
step() → StepResult
state() → InternalState (grader-only; never shown to agent)

All agent-visible information flows through ForensicObs.
The InternalState / TruthDAG is strictly internal.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Dict, Optional

from schema import (
    ActionType,
    FileMetadata,
    ForensicAction,
    ForensicObs,
    ForensicPivot,
    InternalState,
    IOCType,
    SearchResult,
    VirtualFile,
)
from grader import calculate_final_score

# ---------------------------------------------------------------------------
# StepResult — what the env returns after every action
# ---------------------------------------------------------------------------

class StepResult:
    """Mirrors the OpenEnv contract: observation + reward + done flag."""

    def __init__(
        self,
        observation: ForensicObs,
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info: Dict[str, Any] = info or {}

    def __repr__(self) -> str:
        return (
            f"StepResult(reward={self.reward:+.3f}, done={self.done}, "
            f"budget={self.observation.remaining_budget})"
        )

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

BUDGET_MAX = 50
READ_WINDOW = 5000  # INCREASED from 1000 to fit search results

REWARD_MILESTONE = +0.20  # first Read/Tag of a critical-path artifact
REWARD_STEP_COST = -0.05  # per-action analytical tax
REWARD_HONEYPOT = -0.40  # tagging a honeypot file
REWARD_RESOLUTION = +1.00  # correct SubmitCase

EFFICIENCY_THRESHOLD = 0.40  # fraction of budget remaining for bonus
EFFICIENCY_BONUS = +0.10

# ---------------------------------------------------------------------------
# ShadowRegisterEnv
# ---------------------------------------------------------------------------

class ShadowRegisterEnv:
    """
    OpenEnv-compliant environment for the SHADOW_REGISTER benchmark.

    Usage
    -----
    state = generate_world("noisy_entry", seed=42)
    env = ShadowRegisterEnv(state)
    result = env.reset()

    while not result.done:
        action = agent.act(result.observation)
        result = env.step(action)
    """

    # ------------------------------------------------------------------
    # Construction & OpenEnv lifecycle
    # ------------------------------------------------------------------

    def __init__(self, internal_state: InternalState) -> None:
        self._master_state = internal_state  # never mutated after init
        self._state: InternalState  # working copy reset() builds
        self._obs: ForensicObs
        self._milestones_hit: set  # artifact paths already rewarded
        self._episode_reward: float
        self._done: bool
        self._last_grader_report: Optional[Dict[str, Any]] = None

    def reset(self) -> StepResult:
        """
        Start a fresh episode from the provided InternalState.
        Resets budget, evidence bag, milestone tracker, and reward accumulator.
        """
        import copy
        self._state = copy.deepcopy(self._master_state)
        self._milestones_hit: set = set()
        self._episode_reward: float = 0.0
        self._done = False
        self._last_pivots = []
        self._last_grader_report = None

        self._obs = ForensicObs(
            current_view=self._welcome_banner(),
            working_directory="/",
            artifact_metadata=None,
            tagged_evidence={},
            remaining_budget=BUDGET_MAX,
            last_action_log="Episode started. Good luck, Analyst.",
            search_results=None,
        )
        return StepResult(observation=self._obs, reward=0.0, done=False)

    def step(self, action: ForensicAction) -> StepResult:
        """
        Execute one forensic action and return the next observation + reward.

        Every action costs 1 budget unit.
        Budget reaching 0 terminates the episode immediately.
        """
        if self._done:
            return StepResult(
                observation=self._obs, reward=0.0, done=True,
                info={"error": "Episode already finished. Call reset().",
                      "grader_report": self._last_grader_report},
            )

        # ----- dispatch -----------------------------------------------
        handler = {
            ActionType.SEARCH: self._handle_search,
            ActionType.INSPECT: self._handle_inspect,
            ActionType.READ: self._handle_read,
            ActionType.TAG: self._handle_tag,
            ActionType.SUBMIT: self._handle_submit,
        }.get(action.action)

        if handler is None:
            self._obs = ForensicObs(
                **{**self._obs.model_dump(),
                   "current_view": f"ERROR: Unknown action type: {action.action}",
                   "last_action_log": f"Unknown action type: {action.action}",
                   "remaining_budget": self._obs.remaining_budget - 1}
            )
            total_reward = REWARD_STEP_COST
            self._episode_reward += total_reward
            if self._obs.remaining_budget <= 0:
                self._done = True
            return StepResult(
                observation=self._obs, reward=total_reward, done=self._done,
                info={"error": f"Unknown action type: {action.action}"},
            )

        step_reward, view, meta, log_msg, search_results = handler(action)

        # ----- budget tick --------------------------------------------
        self._obs = ForensicObs(
            current_view=view[:READ_WINDOW],
            working_directory=self._obs.working_directory,
            artifact_metadata=meta,
            tagged_evidence=dict(self._obs.tagged_evidence),
            remaining_budget=self._obs.remaining_budget - 1,
            last_action_log=log_msg,
            search_results=search_results,
        )

        # ----- analytical cost ----------------------------------------
        total_reward = step_reward + REWARD_STEP_COST
        self._episode_reward += total_reward

        # ----- termination checks -------------------------------------
        if self._obs.remaining_budget <= 0:
            self._done = True
            log_msg = "⚠ Forensic budget exhausted. Episode terminated."
            self._obs = ForensicObs(
                **{**self._obs.model_dump(),
                   "last_action_log": log_msg,
                   "remaining_budget": 0}
            )

        return StepResult(
            observation=self._obs,
            reward=total_reward,
            done=self._done,
            info={"episode_reward_so_far": self._episode_reward,
                  "grader_report": self._last_grader_report},
        )

    def state(self) -> InternalState:
        """
        Return the full InternalState including TruthDAG.
        GRADER / EVALUATOR USE ONLY — never pass this to the agent.
        """
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # FIXED: Each now returns (step_reward, view_str, optional_meta, log_str, search_results)
    # ------------------------------------------------------------------

    def _handle_search(
        self, action: ForensicAction
    ) -> tuple[float, str, None, str, Optional[list]]:
        """
        Global keyword search across all virtual files.
        Returns a SearchResponse: list of (filename, hit_count, relevance_score).
        Raw file content is NEVER returned — only document-level metadata.
        """
        query = (action.query or "").strip()
        if not query:
            return 0.0, "ERROR: Search requires a non-empty query.", None, "Search failed: empty query.", None

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        results: list[SearchResult] = []

        for path, vf in self._state.filesystem.items():
            matches = pattern.findall(vf.content)
            if matches:
                line_count = max(vf.content.count("\n"), 1)
                relevance = round(min(len(matches) / line_count, 1.0), 3)
                results.append(
                    SearchResult(
                        filename=path,
                        hit_count=len(matches),
                        relevance_score=relevance,
                    )
                )

        if not results:
            view = f'SEARCH "{query}": 0 results.'
            return 0.0, view, None, f'Search "{query}" returned no hits.', None

        results.sort(key=lambda r: r.relevance_score, reverse=True)

        lines = [f'SEARCH "{query}": {len(results)} file(s) matched.\n']
        lines += [
            f" {r.filename} hits={r.hit_count} relevance={r.relevance_score:.3f}"
            for r in results
        ]
        view = "\n".join(lines)
        log = f'Search "{query}": {len(results)} hits.'
        return 0.0, view, None, log, results

    def _handle_inspect(
        self, action: ForensicAction
    ) -> tuple[float, str, Optional[FileMetadata], str, None]:
        """
        Retrieve full stat(1)-style metadata for a file.
        This is the primary action for detecting Timestomping.
        """
        path = (action.path or "").strip()
        vf = self._state.filesystem.get(path)

        if vf is None:
            return 0.0, f"INSPECT: No such file: {path}", None, f"Inspect failed: {path} not found.", None

        m = vf.metadata
        view = (
            f"File: {path}\n"
            f" Size: {m.size} bytes\n"
            f" Permissions: {m.permissions}\n"
            f" UID/GID: {m.uid}/{m.gid}\n"
            f" Modify: {m.mtime}\n"
            f" Access: {m.atime}\n"
            f" Change: {m.ctime}\n"
        )
        log = f"Inspected metadata for {path}."
        return 0.0, view, m, log, None

    def _handle_read(
        self, action: ForensicAction
    ) -> tuple[float, str, Optional[FileMetadata], str, None]:
        """
        Read a 1000-character window of a file at the given byte offset.
        Milestone reward: +0.20 on the first Read of a critical-path artifact.
        """
        path = (action.path or "").strip()
        offset = max(action.offset or 0, 0)
        vf = self._state.filesystem.get(path)

        if vf is None:
            return 0.0, f"READ: No such file: {path}", None, f"Read failed: {path} not found.", None

        chunk = vf.content[offset: offset + 1000]
        if not chunk:
            view = f"READ {path}@{offset}: End of file (size={len(vf.content)})."
            return 0.0, view, vf.metadata, f"Read past EOF: {path}.", None

        view = (
            f"READ {path} [offset={offset}, +{len(chunk)} chars]\n"
            f"{'─' * 60}\n"
            f"{chunk}\n"
            f"{'─' * 60}\n"
            f"[EOF in {max(len(vf.content) - offset - len(chunk), 0)} chars]"
        )

        # Milestone: first read of a truth artifact
        reward = 0.0
        truth_paths = {n.required_artifact for n in self._state.truth_dag.nodes.values()
                       if not n.is_honeypot}
        if path in truth_paths and path not in self._milestones_hit:
            self._milestones_hit.add(path)
            reward = REWARD_MILESTONE
            log = f"Read {path} — MILESTONE +{REWARD_MILESTONE:.2f}."
        else:
            log = f"Read {path} (offset={offset})."

        return reward, view, vf.metadata, log, None

    def _handle_tag(
        self, action: ForensicAction
    ) -> tuple[float, str, None, str, None]:
        """
        Formally record a piece of evidence in tagged_evidence.
        
        FIXED: Now checks if value matches truth node IOC and gives reward!
        Honeypot check: if the value is from a honeypot, apply the -0.4 penalty.
        """
        label = (action.label or "").strip()
        value = (action.value or "").strip()

        if not label or not value:
            return 0.0, "TAG: label and value are both required.", None, "Tag failed: missing field.", None

        reward = 0.0
        log_suffix = ""

        # FIXED: Check if matches truth node IOC (CORRECT EVIDENCE!)
        for node in self._state.truth_dag.nodes.values():
            if node.expected_ioc == value and not node.is_honeypot:
                if label not in self._milestones_hit:
                    self._milestones_hit.add(label)
                    reward = REWARD_MILESTONE  # +0.20
                    log_suffix = f" ✓ CORRECT IOC +{REWARD_MILESTONE:.2f}"
                    break

        # Honeypot check — if not already matched
        if reward == 0.0:
            honeypot_paths = {
                n.required_artifact
                for n in self._state.truth_dag.nodes.values()
                if n.is_honeypot
            }
            honeypot_iocs = {
                n.expected_ioc
                for n in self._state.truth_dag.nodes.values()
                if n.is_honeypot
            }

            if value in honeypot_paths or value in honeypot_iocs:
                reward = REWARD_HONEYPOT
                log_suffix = f" ⚠ HONEYPOT PENALTY {REWARD_HONEYPOT:.2f}"

        # Update the evidence bag (persists in observation)
        new_evidence = dict(self._obs.tagged_evidence)
        new_evidence[label] = value
        self._obs = ForensicObs(
            **{**self._obs.model_dump(), "tagged_evidence": new_evidence}
        )

        view = f"TAG recorded: [{label}] = {value!r}{log_suffix}"
        log = f"Tagged evidence: {label}={value!r}.{log_suffix}"
        return reward, view, None, log, None

    def _handle_submit(
        self, action: ForensicAction
    ) -> tuple[float, str, None, str, None]:
        """
        SubmitCase — end the episode.
        
        FIXED: Now calls grader and computes final reward!
        
        Scoring is now computed by grader.py for separation of concerns.
        The env grants reward based on grader score.
        """
        pivots = action.pivots or []
        
        if not pivots:
            self._done = True
            self._last_pivots = []
            view = "SUBMIT: No ForensicPivots provided. Score = 0. Episode terminated."
            return 0.0, view, None, "SubmitCase: empty pivot list — episode ended.", None

        # FIXED: Call grader to evaluate pivots
        grader_report = calculate_final_score(
            pivots=pivots,
            truth=self._state.truth_dag,
            remaining_budget=self._obs.remaining_budget
        )

        self._done = True
        self._last_pivots = pivots
        self._last_grader_report = grader_report  # Store for agent feedback

        # FIXED: Calculate reward based on grader score
        if grader_report.score >= 0.80:
            final_reward = REWARD_RESOLUTION  # +1.0
        else:
            final_reward = grader_report.score * 0.5 - 0.25

        view = (
            f"SUBMIT: Case filed with {len(pivots)} pivot(s).\n"
            f"Grader Verdict: {grader_report.verdict}\n"
            f"Score: {grader_report.score:.3f}\n"
            f"Nodes Resolved: {grader_report.nodes_resolved}/{grader_report.total_nodes}\n"
            + "\n".join(
                f" [{i+1}] {p.artifact} → {p.ioc} ({p.type}) | {p.reason}"
                for i, p in enumerate(pivots)
            )
        )

        log = f"SubmitCase: {len(pivots)} pivots filed. Score: {grader_report.score:.3f}. Verdict: {grader_report.verdict}"

        return final_reward, view, None, log, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _welcome_banner(self) -> str:
        scenario = self._state.truth_dag.scenario_name
        n_files = len(self._state.filesystem)
        return (
            f"=== SHADOW_REGISTER // Forensic Terminal ===\n"
            f"Scenario : {scenario}\n"
            f"Artifacts: {n_files} files indexed\n"
            f"Budget : {BUDGET_MAX} actions remaining\n"
            f"Objective: Reconstruct the Kill Chain and SubmitCase.\n"
            f"{'─' * 46}\n"
            f"Available commands: Search, Inspect, Read, Tag, SubmitCase\n"
            f"Tip: Use Search to locate artifacts, Inspect to check metadata."
        )

    @property
    def last_pivots(self) -> list[ForensicPivot]:
        """Retrieve the pivots from the most recent SubmitCase action."""
        return getattr(self, "_last_pivots", [])

    @property
    def last_grader_report(self) -> Optional[Dict[str, Any]]:
        """Retrieve the grader report from the most recent SubmitCase action."""
        return getattr(self, "_last_grader_report", None)