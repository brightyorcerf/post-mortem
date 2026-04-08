---
title: post-mortem
emoji: 🕵️
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

<div align="center">
  <h1> post-mortem </h1>
  <p><i>A Deterministic Benchmark for Autonomous Forensic Attribution</i></p>
  <img src="https://img.shields.io/badge/OpenEnv-v1.0-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Compliant-success?style=flat-square" />
</div>

---

## 1 · Environment Description & Motivation

### The Problem:

Raw logs don't lie, but they are designed to be ignored. 
If autonomous agents can be trained to perform forensic attribution (identifying *who* attacked, *how* they persisted, and *what* they exfiltrated), the economics of incident response fundamentally change. Significantly.

### The Solution: 
post-mortem, a deterministic, OpenEnv-compliant sandbox that simulates a forensic analyst's workstation on a compromised Linux server. The agent is dropped into a virtual filesystem populated with:

- System logs (`auth.log`, `syslog`, `dpkg.log`, `nginx/access.log`) containing realistic benign activity
- Kill Chain artifacts: injected attack traces following real-world intrusion patterns
- Adversarial deception: honeypot files and timestomped binaries designed to mislead

The environment is a benchmark for relational reasoning under uncertainty. The agent must `Search`, `Read`, `Inspect` metadata, and `Tag` evidence across multiple correlated artifacts before filing a structured `SubmitCase` report consisting of `ForensicPivots` (each binding an artifact path, an Indicator of Compromise (IOC), its classification type, and a causal justification).

> "post-mortem" challenges agents to move beyond simple keyword searching and perform deep metadata analysis to uncover 'living-off-the-land' (LotL) techniques that bypass traditional security layers.

| Feature | post-mortem | Typical CTF Environments |
|:--------|:---------------|:------------------------|
| Grading | Weighted DAG with causal chain validation | Binary flag capture |
| Deception resistance | Active honeypots with penalties | N/A |
| Metadata reasoning | `stat(1)` timestamps as first-class evidence | Text-only |
| Partial credit | Per-node weighted scoring (0.0–1.0 continuous) | Pass/fail |
| Reproducibility | σ=0, seed-deterministic | Often non-deterministic |
| Anti-forensics | Timestomping, naming mimicry, decoy IPs | Rarely modelled |

---

### The Ground Truth: TruthDAG

What makes post-mortem unique is its embedded TruthDAG — a Directed Acyclic Graph that encodes the precise Kill Chain executed by the adversary.
- Nodes: Each node represents a verifiable forensic fact (e.g., IPs, Timestamps, or Paths) as defined in the IOC Taxonomy (Section 2.3).
- Edges: These define causal prerequisite relationships.

Example: An agent that identifies a C2 callback IP without first locating the persistence mechanism receives reduced credit due to a broken chain penalty.

This TruthDAG acts as a Deterministic Oracle for the grader, providing the objective ground truth that real-world forensics often lacks.
 
### Determinism: σ=0 Across Runs

All world generation routes through a single `numpy.RandomState` instance seeded at `reset()`. The same `(task, seed)` pair produces a byte-identical virtual filesystem, identical TruthDAG, and identical grader scores - verified across 100 iterations with zero variance. There is no wall-clock dependency, no external randomness. Reproducibility is mathematically guaranteed.

---

## 2 · Action & Observation Space

### 2.1 · Action Space: `ForensicAction`

The agent communicates through a single Pydantic-validated model: `ForensicAction`. Every action costs 1 budget unit from the 50-unit forensic budget. Budget exhaustion terminates the episode immediately.

| Action | Purpose | Returns | Strategic Role |
|:-------|:--------|:--------|:---------------|
| **`Search`** | Global keyword search across all virtual files | Filename, hit count, and relevance score per file — **never raw content** | Triage: identify which artifacts warrant deeper analysis. Relevance scoring penalises noisy files (high line count, low match density), preventing blind grep strategies. |
| **`Inspect`** | Retrieve `stat(1)`-style metadata for a file | `mtime`, `atime`, `ctime`, `size`, `uid`, `gid`, `permissions` | Critical for detecting **timestomping** — mtime/ctime discrepancies reveal anti-forensic tampering. Also surfaces size mismatches against package records. |
| **`Read`** | Read a 1000-character window at a given byte offset | Content slice + EOF distance indicator | Primary intelligence-gathering action. The 1000-char window forces the agent to reason about *which portion* of a large log to read, rather than consuming the entire file. |
| **`Tag`** | Formally record evidence as a key-value pair | Confirmation + **honeypot penalty check** | Evidence bookkeeping with consequences: tagging a honeypot artifact triggers a **-0.40 deception penalty**. Forces the agent to validate before committing. |
| **`SubmitCase`** | File the final case report and terminate the episode | Grader evaluation score (0.0–1.0) | The agent submits a list of `ForensicPivot` objects, each binding an artifact to an IOC with a classified type and causal reason. Scoring is immediate and deterministic. |


### 2.2 · Observation Space: `ForensicObs`

After every action, the agent receives a strictly typed observation:

```python
class ForensicObs(BaseModel):
    current_view:      str                     # 1000-char terminal output window
    working_directory: str                     # Current filesystem location
    artifact_metadata: Optional[FileMetadata]  # stat(1) data after Inspect
    tagged_evidence:   Dict[str, str]          # Accumulated evidence bag
    remaining_budget:  int                     # Actions left before termination (max: 50)
    last_action_log:   str                     # Human-readable action result
```

### 2.3 · IOC Type Taxonomy

| IOC Type | Grader Matching | Example |
|:---------|:----------------|:--------|
| `NETWORK_IP` | Exact (normalised) | `185.47.92.13` |
| `EVENT_TIMESTAMP` | Exact ISO 8601, or substring within discrepancy string | `2025-11-14T03:22:17Z` |
| `PATH_TO_FILE` | Exact (leading `/` tolerated) | `/var/www/.config/.update_check` |
| `COMMAND_STRING` | **Substring** match (base64 payloads are long) | `Y3VybCAtcyBodHRwOi...` |
| `USER_ACCOUNT` | Exact (normalised) | `www-data` |
| `FILE_HASH` | Exact (normalised) | `a3f8c29d01b7e54f...` |

---

## 3 · Task Descriptions

Ships three investigation scenarios with clear difficulty progression. All three are procedurally generated from a seed, graded against a deterministic TruthDAG, and scored from **0.0 to 1.0**.

---

### 3.1 · `noisy_entry`

> Scenario: A textbook SSH brute-force against a public-facing service.

An external attacker hammers the SSH daemon with **47–120 rapid-fire password attempts** (1–8 seconds between attempts) across multiple usernames (`root`, `admin`, `ubuntu`, `test`, `oracle`), all from a single source IP. The tight inter-packet spacing is the primary signal. Eventually, one `Accepted password` entry appears — the moment of compromise.

**Kill Chain nodes:**

| Node | Weight | Required Artifact | Expected IOC | Type |
|:-----|:------:|:------------------|:-------------|:-----|
| **A** | 0.50 | `/var/log/auth.log` | Attacker source IP | `NETWORK_IP` |
| **B** | 0.50 | `/var/log/auth.log` | Exact successful-login timestamp (ISO 8601) | `EVENT_TIMESTAMP` |

**DAG edges:** `A → B` (IP identification is a prerequisite for pinpointing the success timestamp)

**Honeypot:** `/tmp/ssh_credentials.txt`: a file designed to look like a leaked SSH key backup. Tagging it triggers **-0.40 penalty**. It's bait: the file contains legitimate config and no attacker IOCs.

**What makes it "Easy":** The signal-to-noise ratio is high. The brute-force cluster is unmistakable in `auth.log`, and both IOCs come from a single file. A competent agent can solve this in under 10 actions.

---

### 3.2 · `stealthy_persistence` 

> **Scenario:** A hidden cron job disguised as a PHP session-cleanup routine, beaconing to a remote C2.

The attacker planted a malicious crontab entry under `www-data` (not `root`, a deliberate evasion of searches in `/var/spool/cron/crontabs/root`). The cron entry calls a hidden launcher at `/var/www/.config/.update_check`, whose name mimics a legitimate package manager health check. Inside, a base64-encoded payload decodes to a `curl` beacon to a remote C2 server.

**Kill Chain nodes:**

| Node | Weight | Required Artifact | Expected IOC | Type |
|:-----|:------:|:------------------|:-------------|:-----|
| **A** | 0.30 | `/var/spool/cron/crontabs/www-data` | `/var/www/.config/.update_check` (launcher path) | `PATH_TO_FILE` |
| **B** | 0.50 | `/var/www/.config/.update_check` | C2 server IP (decoded from base64) | `NETWORK_IP` |
| **C** | 0.20 | `/var/www/.config/.update_check` | Raw base64 command string | `COMMAND_STRING` |

**DAG edges:** `A → B → C` (the agent must follow the crontab → launcher → payload chain in order)

**Honeypot:** `/tmp/.cache_clear.sh`: a world-readable shell script with a suspicious hidden dotfile prefix in `/tmp`. It's actually a legitimate Nginx cache cleanup script authored by the ops team. The agent must resist the urge to flag anything with a dotfile prefix in `/tmp`.

**What makes it "Medium":** Multi-hop reasoning is required. The agent must traverse three artifacts, decode base64, and resist a plausible-looking honeypot. Keyword-only agents fail because terms like "cron," "curl," and "cache" appear in both malicious and benign contexts throughout the noise layer.

---

### 3.3 · `timestomp_proxy` — The Timestomp Proxy (Hard)

> **Scenario:** A malicious insider trojaned `/usr/bin/login` and forged its modification timestamp to hide the intrusion window.

An insider with root access replaced the system login binary with a backdoored version that beacons to a C2 on every successful authentication. To mask the modification, they ran touch -t <original_compile_date> /usr/bin/login, forging the mtime back to the original package era (2019–2022). However, the ctime (inode-change time) cannot be forged without low-level filesystem manipulation. The temporal discrepancy between mtime and ctime is the smoking gun. Additionally, the file's on-disk size is 12–40 KB larger than the size recorded in /var/log/dpkg.log at install time.

**Kill Chain nodes:**

| Node | Weight | Required Artifact | Expected IOC | Type |
|:-----|:------:|:------------------|:-------------|:-----|
| **A** | 0.20 | `/usr/bin/login` | `/usr/bin/login` (tampered binary path) | `PATH_TO_FILE` |
| **B** | 0.40 | `/usr/bin/login` | C2 IP embedded in binary strings dump | `NETWORK_IP` |
| **C** | 0.40 | `/usr/bin/login` | `mtime=<forged> vs ctime=<actual>` (discrepancy proof) | `EVENT_TIMESTAMP` |

**DAG edges:** `A → B → C` (identification → extraction → temporal proof)

**Decoys (×2):**
- **`HONEY_1`**: `/usr/bin/sudo`: has a slightly stale `mtime`, but `mtime == ctime`, meaning it's genuinely old, not tampered.
- **`HONEY_2`**: `/var/log/fw.log`: contains a **different** external IP (UFW firewall block log) that is *not* the C2. Agents that surface any external IP without validating its context get penalised.

**What makes it "Hard":** This task requires metadata literacy (a capability beyond text comprehension). The agent must recognize that mtime and ctime should be correlated, identify the impossible temporal state of the binary, and cross-reference on-disk file sizes against historical package records in dpkg.log.

---

### Grader Architecture

The grader (`grader.py`) is fully deterministic and operates in four phases:

1. **Per-node weighted matching**: Each submitted `ForensicPivot` is compared against TruthDAG nodes using type-aware IOC matching (exact, substring, or path-normalised depending on IOC type). Matched nodes contribute their weight to the raw score.

2. **Honeypot penalty**: Any pivot that matches a honeypot node subtracts **-0.40** from the score.

3. **DAG chain validation**: If the agent matched node B but not its prerequisite A (where `A→B` is a DAG edge), the positive score is multiplied by **0.5x per broken link** (floored at 0.25x). This enforces Kill Chain coherence: random correct guesses are worth less than structured forensic reasoning.

4. **Efficiency bonus**: If the agent retains **≥40% of the budget** (≥20 of 50 actions) at submission time and the score is positive, a **+0.10 bonus** is applied. This rewards analytical precision over brute-force exploration.

The final score is clamped to `[0.0, 1.0]`.

---

## 4 · Reward Design

Rewards are emitted per-step to provide dense learning signal:

| Component | Value | Trigger |
|:----------|:-----:|:--------|
| **Evidence Milestone** | `+0.20` | First `Read` of an artifact on the TruthDAG critical path |
| **Analytical Cost** | `-0.05` | Every action (discourages brute-force file enumeration) |
| **Honeypot Penalty** | `-0.40` | `Tag` action targeting a honeypot's artifact or IOC |
| **Resolution Bonus** | `+1.00` | `SubmitCase` with pivots correctly matching the TruthDAG |
| **Efficiency Bonus** | `+0.10` | `SubmitCase` with ≥40% budget remaining AND positive score |

The per-step cost creates an implicit planning pressure: an agent that reads every file in the filesystem burns 20+ actions on noise, leaving insufficient budget for the investigation and sacrificing the efficiency bonus.

---

## 5 · Setup & Usage

### 5.1 Live Environment (Hugging Face Spaces)

The environment is pre-deployed and reachable for remote evaluation:
URL: [https://huggingface.co/spaces/brightyorcerf/post-mortem](https://huggingface.co/spaces/brightyorcerf/post-mortem)

### 5.2 · Docker

```bash
# Build the image
docker build -t shadow-register:latest .

# Run the environment server
docker run -p 7860:7860 shadow-register:latest
```
The server starts on port `7860` with a health check at `/ping`.

### 5.3 · Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
python3 -m server.app
# → Uvicorn running on http://0.0.0.0:7860
```

### 5.4 · Running Inference

```bash
# Set required environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-hf-token-here"

# Run a single episode
python3 inference.py --task noisy_entry --seed 42

# Run all three tasks
python3 inference.py --task noisy_entry --seed 42
python3 inference.py --task stealthy_persistence --seed 42
python3 inference.py --task timestomp_proxy --seed 42
```


### 5.5 · OpenEnv Validation

```bash
openenv validate .
```

The environment exposes the four required OpenEnv endpoints:

| Endpoint | Method | Purpose |
|:---------|:------:|:--------|
| `/ping` | `GET` | Health check → `{"status": "ok"}` |
| `/reset` | `POST` | Start a fresh episode (`task`, `seed`) |
| `/step` | `POST` | Execute a `ForensicAction`, return next observation |
| `/state` | `GET` | Return full `InternalState` + TruthDAG (**grader only**) |

### 5.6 Tests

```bash
python3 tests/evaluateStability.py
python3 tests/testRun.py
python3 tests/testSpec.py
```

---

## 6 · Preliminary Baseline Performance

The following results represent zero-shot performance using a ReAct-style prompting strategy. All evaluations were conducted at `temperature=0` to minimize variance, though model-side non-determinism remains.

| Agent / Model | `noisy_entry` (Easy) | `stealthy_persistence` (Mid) | `timestomp_proxy` (Hard) |
| :--- | :---: | :---: | :---: |
| **Oracle (Hardcoded)** | 1.00 | 1.00 | 1.00 |
| **GPT-4o** | 0.94 | 0.58 | 0.14 |
| **GPT-4o-mini** | 0.88 | 0.32 | 0.04 |
| **Random Baseline** | 0.02 | 0.00 | 0.00 |

The significant performance decay in `timestomp_proxy` highlights a specific "reasoning gap" in current LLMs regarding temporal metadata analysis. While models successfully identify the tampered binary, they frequently fail to provide the exact `mtime/ctime` discrepancy proof required by the TruthDAG, resulting in reduced partial credit.

---

## 7 · Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     inference.py                            │
│              (LLM Agent ↔ OpenAI API)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP (POST /reset, /step)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      server.py                              │
│              FastAPI · OpenEnv HTTP Layer                    │
│         GET /ping · POST /reset · POST /step · GET /state   │
└──────┬─────────────────────────────────────────┬────────────┘
       │                                         │
       ▼                                         ▼
┌──────────────┐                        ┌─────────────────┐
│   env.py     │                        │   grader.py     │
│  Game Loop   │                        │  Scoring Ref    │
│  Budget Mgmt │                        │  DAG Matching   │
│  Milestone   │                        │  Chain Mult     │
│  Honeypot    │                        │  Efficiency ±   │
│  Detection   │                        │  Clamp [0,1]    │
└──────┬───────┘                        └────────┬────────┘
       │                                         │
       ▼                                         │
┌──────────────┐         ┌──────────────┐        │
│ worldGen.py  │────────▶│  schema.py   │◀───────┘
│ Procedural   │         │  Pydantic    │
│ World Gen    │         │  Models      │
│ 3 Builders   │         │  ForensicObs │
│ Noise Layer  │         │  ForensicAct │
│ σ=0 Seeded   │         │  TruthDAG    │
└──────────────┘         └──────────────┘
```

---

License: MIT | OpenEnv v1.0 Compliant