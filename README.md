---
title: post-mortem
emoji: 🕵️
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

<div align="center">

# post-mortem

> A Deterministic Benchmark for Autonomous Forensic Attribution

## 1 · Environment Description & Motivation

### The Problem:

Raw logs don't lie, but they are designed to be ignored. "post-mortem" challenges agents to move beyond simple keyword searching and perform deep metadata analysis to uncover 'living-off-the-land' (LotL) techniques that bypass traditional security layers

If autonomous agents can be trained to perform forensic attribution (identifying *who* attacked, *how* they persisted, and *what* they exfiltrated), the economics of incident response fundamentally change. Significantly.

### The Solution:

post-mortem is a deterministic, OpenEnv-compliant sandbox that simulates a forensic analyst's workstation on a compromised Linux server. The agent is dropped into a virtual filesystem populated with:

- System logs (`auth.log`, `syslog`, `dpkg.log`, `nginx/access.log`) containing realistic benign activity
- Kill Chain artifacts: injected attack traces following real-world intrusion patterns
- Adversarial deception: honeypot files and timestomped binaries designed to mislead

The environment is a benchmark for relational reasoning under uncertainty. The agent must `Search`, `Read`, `Inspect` metadata, and `Tag` evidence across multiple correlated artifacts before filing a structured `SubmitCase` report consisting of `ForensicPivots` (each binding an artifact path, an Indicator of Compromise (IOC), its classification type, and a causal justification).

> ### The Ground Truth: TruthDAG
>
> What makes post-mortem unique among cybersecurity environments is its embedded TruthDAG — a Directed Acyclic Graph that encodes the precise Kill Chain executed by the adversary.
>
> * Nodes: Each node represents a verifiable forensic fact:
>   * `NETWORK_IP` (e.g., C2 callback address)
>   * `EVENT_TIMESTAMP` (e.g., moment of successful brute-force)
>   * `PATH_TO_FILE` (e.g., location of a timestomped binary)
> * Edges: Edges define causal prerequisite relationships. An agent that identifies a C2 callback IP without first locating the persistence mechanism receives reduced credit due to a broken chain penalty.
>
> This TruthDAG is never exposed to the agent; it exists solely as a **Deterministic Oracle** for the grader, providing the objective ground truth that real-world forensics often lacks.

This TruthDAG is never exposed to the agent. It exists solely for the grader, providing the deterministic oracle that real-world forensics lacks.

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

The `current_view` field acts as the agent's rendered terminal: every Search result, file content window, and error message flows through this 1000-character channel. The agent never sees the raw filesystem; only what its actions surface. `artifact_metadata` populates only after an `Inspect` call, providing the MAC timestamps (`mtime`, `atime`, `ctime`), file size, ownership, and permissions required for temporal analysis.

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

**Honeypot:** `/tmp/ssh_credentials.txt` — a file designed to look like a leaked SSH key backup. Tagging it triggers **-0.40 penalty**. It's bait: the file contains legitimate config and no attacker IOCs.

**What makes it "Easy":** The signal-to-noise ratio is high. The brute-force cluster is unmistakable in `auth.log`, and both IOCs come from a single file. A competent agent can solve this in under 10 actions.

---

### 3.2 · `stealthy_persistence` 

> **Scenario:** A hidden cron job disguised as a PHP session-cleanup routine, beaconing to a remote C2.

The attacker planted a malicious crontab entry under `www-data` (not `root` — a deliberate evasion of searches in `/var/spool/cron/crontabs/root`). The cron entry calls a hidden launcher at `/var/www/.config/.update_check`, whose name mimics a legitimate package manager health check. Inside, a base64-encoded payload decodes to a `curl` beacon to a remote C2 server.

**Kill Chain nodes:**

| Node | Weight | Required Artifact | Expected IOC | Type |
|:-----|:------:|:------------------|:-------------|:-----|
| **A** | 0.30 | `/var/spool/cron/crontabs/www-data` | `/var/www/.config/.update_check` (launcher path) | `PATH_TO_FILE` |
| **B** | 0.50 | `/var/www/.config/.update_check` | C2 server IP (decoded from base64) | `NETWORK_IP` |
| **C** | 0.20 | `/var/www/.config/.update_check` | Raw base64 command string | `COMMAND_STRING` |

**DAG edges:** `A → B → C` (the agent must follow the crontab → launcher → payload chain in order)

**Honeypot:** `/tmp/.cache_clear.sh` — a world-readable shell script with a suspicious hidden dotfile prefix in `/tmp`. It's actually a legitimate Nginx cache cleanup script authored by the ops team. The agent must resist the urge to flag anything with a dotfile prefix in `/tmp`.

**What makes it "Medium":** Multi-hop reasoning is required. The agent must traverse three artifacts, decode base64, and resist a plausible-looking honeypot. Keyword-only agents fail because terms like "cron," "curl," and "cache" appear in both malicious and benign contexts throughout the noise layer.

---

### 3.3 · `timestomp_proxy` — The Timestomp Proxy (Hard)

> **Scenario:** A malicious insider trojaned `/usr/bin/login` and forged its modification timestamp to hide the intrusion window.

An insider with root access replaced the system `login` binary with a backdoored version that beacons to a C2 on every successful authentication. To mask the modification, they ran `touch -t <original_compile_date> /usr/bin/login`, forging the `mtime` back to the original package era (2019–2022). However, the **`ctime` (inode-change time) cannot be forged** without low-level filesystem manipulation — and they didn't do that. The temporal discrepancy between `mtime` and `ctime` is the smoking gun. Additionally, the file's on-disk size is **12–40 KB larger** than the size recorded in `/var/log/dpkg.log` at install time — a secondary corroborating signal.

**Kill Chain nodes:**

| Node | Weight | Required Artifact | Expected IOC | Type |
|:-----|:------:|:------------------|:-------------|:-----|
| **A** | 0.20 | `/usr/bin/login` | `/usr/bin/login` (tampered binary path) | `PATH_TO_FILE` |
| **B** | 0.40 | `/usr/bin/login` | C2 IP embedded in binary strings dump | `NETWORK_IP` |
| **C** | 0.40 | `/usr/bin/login` | `mtime=<forged> vs ctime=<actual>` (discrepancy proof) | `EVENT_TIMESTAMP` |

**DAG edges:** `A → B → C` (identification → extraction → temporal proof)

**Decoys (×2):**
- **`HONEY_1`**: `/usr/bin/sudo` — has a slightly stale `mtime`, but `mtime == ctime`, meaning it's genuinely old, not tampered.
- **`HONEY_2`**: `/var/log/fw.log` — contains a **different** external IP (UFW firewall block log) that is *not* the C2. Agents that surface any external IP without validating its context get penalised.

**What makes it "Hard":** This task requires *metadata literacy* — a capability beyond text comprehension. The agent must:
1. Recognise that `mtime` and `ctime` should be correlated for untampered files
2. Identify that `/usr/bin/login`'s `mtime` is years before its `ctime` — an impossible condition without deliberate manipulation
3. Cross-reference the file size against `dpkg.log` package records
4. Distinguish the real C2 IP (in the binary strings dump) from the decoy IP (in the firewall log)
5. Construct the exact discrepancy proof string

---

### Grader Architecture

The grader (`grader.py`) is fully deterministic and operates in four phases:

1. **Per-node weighted matching** — Each submitted `ForensicPivot` is compared against TruthDAG nodes using type-aware IOC matching (exact, substring, or path-normalised depending on IOC type). Matched nodes contribute their weight to the raw score.

2. **Honeypot penalty** — Any pivot that matches a honeypot node subtracts **-0.40** from the score.

3. **DAG chain validation** — If the agent matched node B but not its prerequisite A (where `A→B` is a DAG edge), the positive score is multiplied by **0.5x per broken link** (floored at 0.25x). This enforces Kill Chain coherence: random correct guesses are worth less than structured forensic reasoning.

4. **Efficiency bonus** — If the agent retains **≥40% of the budget** (≥20 of 50 actions) at submission time and the score is positive, a **+0.10 bonus** is applied. This rewards analytical precision over brute-force exploration.

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

### 5.1 · Docker (Recommended)

```bash
# Build the image
docker build -t shadow-register:latest .

# Run the environment server
docker run -p 7860:7860 shadow-register:latest
```

The server starts on port `7860` with a health check at `/ping`.

### 5.2 · Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
python3 server.py
# → Uvicorn running on http://0.0.0.0:7860
```

### 5.3 · Running Inference

```bash
# Set required environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-hf-token-here"

# Run a single episode
python inference.py --task noisy_entry --seed 42

# Run all three tasks
python inference.py --task noisy_entry --seed 42
python inference.py --task stealthy_persistence --seed 42
python inference.py --task timestomp_proxy --seed 42
```

### 5.4 · OpenEnv Validation

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

### 5.5 Tests

```bash
python3 tests/evaluateStability.py
python3 tests/testRun.py
python3 tests/testSpec.py
```

---

## 6 · Creativity & Novelty

### 6.1 · The Honeypot Mechanic

Shadow Register doesn't just test whether an agent can *find* evidence — it tests whether the agent can **resist false evidence**. Each scenario includes at least one honeypot: a file that *looks* suspicious (hidden dotfile, leaked credentials, unfamiliar external IP) but is, on closer inspection, legitimate.

- **Easy:** `/tmp/ssh_credentials.txt` — looks like a leaked SSH key backup but contains standard config
- **Medium:** `/tmp/.cache_clear.sh` — hidden dotfile in `/tmp` with executable permissions, but is a sanctioned Nginx cache helper
- **Hard:** `/usr/bin/sudo` with stale timestamps (but `mtime == ctime` — genuinely old, not tampered) *and* `/var/log/fw.log` containing a different external IP (firewall block, not C2)

An agent that tags a honeypot receives a **-0.40 penalty** — identical to the weight of many truth nodes. This creates a meaningful cost for false positives and models the real-world DFIR principle: *accusing a legitimate user or process is not merely unhelpful; it actively degrades the investigation.*

### 6.2 · Procedural World Generation

Every `(task, seed)` pair produces a unique investigation. The world generator (`worldGen.py`) procedurally constructs:

- Attacker IPs drawn from realistic ASN abuse pools (Tor exit nodes, VPS blocks, bulletproof hosting, residential proxy ranges)
- 45+ benign syslog entries from 15 realistic daemon templates
- 22+ bash history entries from a curated benign admin command set
- 6+ stale `/tmp` session tokens
- Full `/etc/passwd` user list, `/var/log/dpkg.log` package history, and `/var/log/nginx/access.log` HTTP traffic

Changing the seed changes *everything*: attacker IPs, failure counts, timestamps, file sizes, noise patterns. The Kill Chain structure remains consistent per task, but the specific IOC values are unique. This prevents agents from memorising answers and forces generalisation across the forensic methodology.

### 6.3 · What Makes This Different

| Feature | Shadow Register | Typical CTF Environments |
|:--------|:---------------|:------------------------|
| **Grading** | Weighted DAG with causal chain validation | Binary flag capture |
| **Deception resistance** | Active honeypots with penalties | N/A |
| **Metadata reasoning** | `stat(1)` timestamps as first-class evidence | Text-only |
| **Partial credit** | Per-node weighted scoring (0.0–1.0 continuous) | Pass/fail |
| **Reproducibility** | σ=0, seed-deterministic | Often non-deterministic |
| **Anti-forensics** | Timestomping, naming mimicry, decoy IPs | Rarely modelled |

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

## 9 · Technical Constraints

| Constraint | Value | Rationale |
|:-----------|:------|:----------|
| **Max memory** | < 100 MB | Virtual filesystem is string-based; no binary blobs |
| **Compute** | 2 vCPU sufficient | Search uses `re.compile` + `findall`; no embeddings or ML in the loop |
| **World generation** | < 1 second | Pure numpy RNG + string formatting |
| **Max steps** | 50 (budget) / 40 (inference default) | 10-action safety margin for the agent |
| **Container** | Python 3.11-slim | No torch/tensorflow/sklearn dependencies |
| **Workers** | 1 | Single-agent evaluation; no concurrency needed |

---

License: MIT | OpenEnv v1.0 Compliant