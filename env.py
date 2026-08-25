"""
env.py  —  SHADOW_REGISTER OpenEnv Environment
===============================================
Systems Architect: Game-loop engine wrapping the virtual filesystem.

Implements the OpenEnv standard:
    reset()  → StepResult
    step()   → StepResult
    state()  → InternalState   (grader-only; never shown to agent)

All agent-visible information flows through ForensicObs.
The InternalState / TruthDAG is strictly internal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from grader import calculate_final_score
from schema import (
    ActionType,
    FileMetadata,
    ForensicAction,
    ForensicObs,
    ForensicPivot,
    InternalState,
)

# ---------------------------------------------------------------------------
# StepResult  —  what the env returns after every action
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Mirrors the OpenEnv contract: observation + reward + done flag."""
    observation: ForensicObs
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

BUDGET_MAX           = 50
READ_WINDOW          = 1000          # characters per Read chunk

REWARD_MILESTONE     = +0.20         # first Read/Tag of a critical-path artifact
REWARD_STEP_COST     = -0.05         # per-action analytical tax
REWARD_HONEYPOT      = -0.40         # tagging a honeypot file
REWARD_TAG_HIT       = +0.10         # first tag of a genuine truth IOC
REWARD_RESOLUTION    = +1.00         # correct SubmitCase — proportional to grader score

# NOTE: the efficiency bonus lives in grader.py — it applies to the final
# score, not to any single step, so there is nothing to duplicate here.


# ---------------------------------------------------------------------------
# ShadowRegisterEnv
# ---------------------------------------------------------------------------

class ShadowRegisterEnv:
    """
    OpenEnv-compliant environment for the SHADOW_REGISTER benchmark.

    Usage
    -----
        state  = generate_world("noisy_entry", seed=42)
        env    = ShadowRegisterEnv(state)
        result = env.reset()

        while not result.done:
            action = agent.act(result.observation)
            result = env.step(action)
    """

    # ------------------------------------------------------------------
    # Construction & OpenEnv lifecycle
    # ------------------------------------------------------------------

    def __init__(self, internal_state: InternalState) -> None:
        self._master_state = internal_state   # never mutated after init
        # Usable before reset(): the grader and /state read truth_dag straight
        # off a freshly constructed env.  reset() swaps in the working copy.
        self._state: InternalState  = internal_state
        self._obs: ForensicObs      = None
        self._milestones_hit: set   = set()   # artifact paths already rewarded
        self._tag_hits: set         = set()   # truth IOCs already credited
        self._done: bool            = False
        self._last_pivots: list     = []

    def reset(self) -> StepResult:
        """
        Start a fresh episode from the provided InternalState.
        Resets budget, evidence bag, milestone tracker, and reward accumulator.
        """
        # Deep-copy only the mutable parts; filesystem is read-only
        import copy
        self._state         = copy.deepcopy(self._master_state)
        self._milestones_hit: set = set()
        self._tag_hits: set = set()
        self._done          = False
        self._last_pivots   = []     # wipe stale pivots from prior episode

        self._obs = ForensicObs(
            current_view=self._welcome_banner(),
            artifact_metadata=None,
            tagged_evidence={},
            remaining_budget=BUDGET_MAX,
            last_action_log="Episode started. Good luck, Analyst.",
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
                info={"error": "Episode already finished. Call reset()."},
            )

        # ----- dispatch -----------------------------------------------
        handler = {
            ActionType.SEARCH:  self._handle_search,
            ActionType.INSPECT: self._handle_inspect,
            ActionType.READ:    self._handle_read,
            ActionType.TAG:     self._handle_tag,
            ActionType.SUBMIT:  self._handle_submit,
        }.get(action.action)

        if handler is None:
            return self.step_rejected(f"Unknown action type: {action.action}")

        step_reward, view, meta, log_msg = handler(action)

        # ----- budget tick --------------------------------------------
        self._obs = ForensicObs(
            current_view=view[:READ_WINDOW],
            artifact_metadata=meta,
            tagged_evidence=dict(self._obs.tagged_evidence),
            remaining_budget=self._obs.remaining_budget - 1,
            last_action_log=log_msg,
        )

        # ----- analytical cost ----------------------------------------
        total_reward = step_reward + REWARD_STEP_COST

        # ----- termination checks -------------------------------------
        if self._obs.remaining_budget <= 0:
            self._done = True
            log_msg = "⚠  Forensic budget exhausted. Episode terminated."
            self._obs = ForensicObs(
                **{**self._obs.model_dump(),
                   "last_action_log": log_msg,
                   "remaining_budget": 0}
            )

        return StepResult(
            observation=self._obs,
            reward=total_reward,
            done=self._done,
        )

    def step_rejected(self, reason: str) -> StepResult:
        """
        Charge a budget unit for an action the env refused to run.

        Covers both unknown action types and payloads that fail schema
        validation at the HTTP boundary.  Without this, a client that emits
        malformed JSON steps forever for free and the budget stops being a
        bound on the episode.
        """
        if self._done:
            return StepResult(
                observation=self._obs, reward=0.0, done=True,
                info={"error": "Episode already finished. Call reset()."},
            )
        self._obs = ForensicObs(
            **{**self._obs.model_dump(),
               "current_view":     f"ERROR: {reason}"[:READ_WINDOW],
               "last_action_log":  reason,
               "remaining_budget": max(self._obs.remaining_budget - 1, 0)}
        )
        if self._obs.remaining_budget <= 0:
            self._done = True
        return StepResult(
            observation=self._obs, reward=REWARD_STEP_COST,
            done=self._done, info={"error": reason},
        )

    def state(self) -> InternalState:
        """
        Return the full InternalState including TruthDAG.
        GRADER / EVALUATOR USE ONLY — never pass this to the agent.
        """
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # Each returns (step_reward, view_str, optional_meta, log_str)
    # ------------------------------------------------------------------

    def _handle_search(
        self, action: ForensicAction
    ) -> tuple[float, str, None, str]:
        """
        Global keyword search across all virtual files.
        Returns a SearchResponse: list of (filename, hit_count, relevance_score).
        Raw file content is NEVER returned — only document-level metadata.
        """
        query = (action.query or "").strip()
        if not query:
            return 0.0, "ERROR: Search requires a non-empty query.", None, "Search failed: empty query."

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        results = []

        for path, vf in self._state.filesystem.items():
            hits = len(pattern.findall(vf.content))
            if hits:
                # Relevance: hits / total lines — noisy files score low
                line_count = max(vf.content.count("\n"), 1)
                results.append((round(min(hits / line_count, 1.0), 3), hits, path))

        if not results:
            view = f'SEARCH "{query}": 0 results.'
            return 0.0, view, None, f'Search "{query}" returned no hits.'

        results.sort(reverse=True)     # by relevance descending
        view = "\n".join(
            [f'SEARCH "{query}": {len(results)} file(s) matched.\n']
            + [f"  {path}  hits={hits}  relevance={rel:.3f}"
               for rel, hits, path in results]
        )
        log  = f'Search "{query}": {len(results)} hits.'
        return 0.0, view, None, log

    def _handle_inspect(
        self, action: ForensicAction
    ) -> tuple[float, str, Optional[FileMetadata], str]:
        """
        Retrieve full stat(1)-style metadata for a file.
        This is the primary action for detecting Timestomping.
        """
        path = (action.path or "").strip()
        vf   = self._state.filesystem.get(path)

        if vf is None:
            return 0.0, f"INSPECT: No such file: {path}", None, f"Inspect failed: {path} not found."

        m = vf.metadata
        # Field names are spelled out (mtime/atime/ctime) so the vocabulary the
        # grader expects for a timestamp-discrepancy IOC is visible to the agent.
        view = (
            f"File: {path}\n"
            f"  Size:        {m.size} bytes\n"
            f"  Permissions: {m.permissions}\n"
            f"  UID/GID:     {m.uid}/{m.gid}\n"
            f"  Modify (mtime): {m.mtime}\n"
            f"  Access (atime): {m.atime}\n"
            f"  Change (ctime): {m.ctime}\n"
        )
        reward, note = self._milestone(path)
        log  = f"Inspected metadata for {path}.{note}"
        return reward, view, m, log

    def _handle_read(
        self, action: ForensicAction
    ) -> tuple[float, str, Optional[FileMetadata], str]:
        """
        Read a 1000-character window of a file at the given byte offset.
        Milestone reward: +0.20 on the first Read of a critical-path artifact.
        """
        path   = (action.path or "").strip()
        offset = max(action.offset or 0, 0)
        vf     = self._state.filesystem.get(path)

        if vf is None:
            return 0.0, f"READ: No such file: {path}", None, f"Read failed: {path} not found."

        chunk = vf.content[offset: offset + READ_WINDOW]
        if not chunk:
            view = f"READ {path}@{offset}: End of file (size={len(vf.content)})."
            return 0.0, view, vf.metadata, f"Read past EOF: {path}."

        def _render(c: str) -> str:
            return (
                f"READ {path} [offset={offset}, +{len(c)} chars]\n"
                f"{'─' * 60}\n"
                f"{c}\n"
                f"{'─' * 60}\n"
                f"[EOF in {max(len(vf.content) - offset - len(c), 0)} chars]"
            )

        # step() caps current_view at READ_WINDOW.  The frame (header, rules,
        # EOF footer) counts against that cap, so an unadjusted chunk loses its
        # tail silently and an agent paging by READ_WINDOW skips those bytes.
        # Trim the chunk to fit, so the "+N chars" header is the true advance.
        view = _render(chunk)
        if len(view) > READ_WINDOW:
            chunk = chunk[: max(len(chunk) - (len(view) - READ_WINDOW), 0)]
            view  = _render(chunk)

        reward, note = self._milestone(path)
        log = f"Read {path} (offset={offset}).{note}"
        return reward, view, vf.metadata, log

    def _handle_tag(
        self, action: ForensicAction
    ) -> tuple[float, str, None, str]:
        """
        Formally record a piece of evidence in tagged_evidence.
        Honeypot check: if the label key matches a honeypot artifact,
        apply the -0.4 deception penalty.
        """
        label = (action.label or "").strip()
        value = (action.value or "").strip()

        if not label or not value:
            return 0.0, "TAG: label and value are both required.", None, "Tag failed: missing field."

        # Honeypot check — value might be a file path or an IOC from a honeypot
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

        truth_iocs = {
            n.expected_ioc
            for n in self._state.truth_dag.nodes.values()
            if not n.is_honeypot
        }

        reward = 0.0
        log_suffix = ""
        if value in honeypot_paths or value in honeypot_iocs:
            reward     = REWARD_HONEYPOT
            log_suffix = f" ⚠ HONEYPOT PENALTY {REWARD_HONEYPOT:.2f}"
        elif value in truth_iocs and value not in self._tag_hits:
            # Without this, Tag costs REWARD_STEP_COST and can only ever lose
            # points, so the optimal policy is to never tag anything — which
            # makes a documented first-class action dead. Credited once per IOC.
            self._tag_hits.add(value)
            reward     = REWARD_TAG_HIT
            log_suffix = f" ✓ EVIDENCE CONFIRMED +{REWARD_TAG_HIT:.2f}"

        # Update the evidence bag (persists in observation)
        new_evidence = dict(self._obs.tagged_evidence)
        new_evidence[label] = value
        self._obs = ForensicObs(
            **{**self._obs.model_dump(), "tagged_evidence": new_evidence}
        )

        view = f"TAG recorded: [{label}] = {value!r}{log_suffix}"
        log  = f"Tagged evidence: {label}={value!r}.{log_suffix}"
        return reward, view, None, log

    def _handle_submit(
        self, action: ForensicAction
    ) -> tuple[float, str, None, str]:
        """
        SubmitCase — end the episode.
        Scoring is deferred to grader.py for separation of concerns.
        Here we store the pivots, mark done, and return a provisional reward
        placeholder.  The true final score is computed by the grader.

        The env grants the +1.0 resolution bonus only when the grader
        (called externally) confirms a correct submission.  Inside the env
        we emit a neutral reward and set done=True so the agent's loop ends.
        """
        pivots = action.pivots or []
        if not pivots:
            self._done = True
            self._last_pivots = []
            view = "SUBMIT: No ForensicPivots provided. Score = 0. Episode terminated."
            return 0.0, view, None, "SubmitCase: empty pivot list — episode ended."

        self._done = True
        # Store pivots in info so the grader can retrieve them
        self._last_pivots = pivots

        # Grade immediately to emit a meaningful resolution reward for RL training.
        # remaining_budget is decremented in step() AFTER the handler returns,
        # so we subtract 1 here to reflect the post-submit state.
        remaining_after = max(self._obs.remaining_budget - 1, 0)
        report = calculate_final_score(
            pivots=pivots,
            truth=self._state.truth_dag,
            remaining_budget=remaining_after,
        )
        # Resolution reward: grader score × REWARD_RESOLUTION scales in [0, 1]
        step_reward = report.score * REWARD_RESOLUTION

        view = (
            f"SUBMIT: Case filed with {len(pivots)} pivot(s).\n"
            f"Grader score: {report.score:.4f} | {report.verdict}\n"
            f"Episode terminated.\n"
            + "\n".join(
                f"  [{i+1}] {p.artifact} → {p.ioc} ({p.type}) | {p.reason}"
                for i, p in enumerate(pivots)
            )
        )
        log = f"SubmitCase: {len(pivots)} pivots filed. Score: {report.score:.4f}"
        return step_reward, view, None, log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _milestone(self, path: str) -> tuple[float, str]:
        """
        Award REWARD_MILESTONE once per critical-path artifact, whichever
        action surfaced it first.  Inspect is the whole point of the
        timestomp task, so it earns credit exactly like Read.
        """
        truth_paths = {n.required_artifact
                       for n in self._state.truth_dag.nodes.values()
                       if not n.is_honeypot}
        if path in truth_paths and path not in self._milestones_hit:
            self._milestones_hit.add(path)
            return REWARD_MILESTONE, f" MILESTONE +{REWARD_MILESTONE:.2f}."
        return 0.0, ""

    def _welcome_banner(self) -> str:
        scenario = self._state.truth_dag.scenario_name
        n_files  = len(self._state.filesystem)
        return (
            f"=== SHADOW_REGISTER // Forensic Terminal ===\n"
            f"Scenario : {scenario}\n"
            f"Artifacts: {n_files} files indexed\n"
            f"Budget   : {BUDGET_MAX} actions remaining\n"
            f"Objective: Reconstruct the Kill Chain and SubmitCase.\n"
            f"{'─' * 46}\n"
            f"Available commands: Search, Inspect, Read, Tag, SubmitCase\n"
            f"Tip: Use Search to locate artifacts, Inspect to check metadata."
        )
    @property
    def last_pivots(self) -> list[ForensicPivot]:
        """Retrieve the pivots from the most recent SubmitCase action."""
        return self._last_pivots
