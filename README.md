# post-mortem

> A Deterministic Benchmark for Forensic Attribution

> Domain: Cybersecurity / Digital Forensics & Incident Response (DFIR)

## Executive Summary

post-mortem is an OpenEnv-compliant environment simulating a post-breach investigation. The agent acts as a Forensic Analyst tasked with reconstructing a Kill Chain from raw artifacts. The environment evaluates an agent’s ability to perform multi-step relational reasoning across conflicting data sources, distinguishing between legitimate activity, system noise, and adversarial deception (timestomping/decoys).

## Environment Specification (OpenEnv)
### Observation Space (ForensicObs)

Strictly typed Pydantic model focused on document-level retrieval.
current_view: 1000-char window using RFC 5424 (Syslog) or ISO 8601 timestamps.
Search Logic: Document that Search returns a SearchResponse object containing hit_count and relevance_score per file—never the raw content. Explicitly mention "System Noise Injection" to prevent easy-pathing.
working_directory: Current location in the simulated filesystem.
artifact_metadata: File system attributes (UID, GID, Size, and MAC timestamps: Modification, Access, Change).
tagged_evidence: A list of key-value pairs the agent has officially extracted (e.g., attacker_ip: 192.168.1.5).

### Action Space (ForensicAction)

Commands modeled after standard SIEM and Linux forensic workflows.

- Search(query: str): Global keyword search. Returns a list of filenames and hit counts (simulating document-level index retrieval).
- Inspect(path: str): Retrieves full stat metadata for a file to detect Timestomping.
- Read(path: str, offset: int): Reads a specific chunk of an artifact.
- Tag(label: str, value: str): Formally marks a piece of evidence for the final report.
- SubmitCase(pivots: List[ForensicPivot]):
Where a ForensicPivot is: 
```
{
  "artifact": "path/to/file",
  "ioc": "192.168.1.5",
  "type": "NETWORK_IP",
  "reason": "persistence_callback"
}
```
This concludes the episode.

## The Curriculum: Incident Scenarios
All tasks are procedurally generated from a seed but graded against deterministic "Truth Points."

### 3. The Curriculum: Incident Scenarios

All tasks are procedurally generated from a deterministic seed and graded against a Directed Acyclic Graph (DAG) of "Truth Points."

| Difficulty | Task Name | Scenario | Grader Logic |
| :--- | :--- | :--- | :--- |
| **Easy** | The Loud Entry | A noisy brute-force attack against a public-facing SSH service. | Success: Identify the unique source IP and the specific successful login timestamp in `auth.log`. |
| **Medium** | The Persistence Loop | A stealthy persistence mechanism utilizing a hidden cron job that executes a base64-encoded payload. | Success: Locate the malicious crontab entry, decode the payload, and identify the remote C2 destination IP. |
| **Hard** | The Timestomp Proxy | A sophisticated insider threat who altered a system binary, then manually reverted the `mtime` to spoof historical logs. | Challenge: Agent must identify metadata discrepancies where `mtime` (Modification) was forged, but `ctime` (Change) reveals the actual injection window. <br> Success: Compare Modify vs. Change timestamps in `stat` to prove tampering. |

## Reward Design: "Forensic Integrity"
Rewards are calculated per step() to provide a dense signal.

- Evidence Milestone (+0.2): Awarded when a "Critical Path" artifact is first Read or Tagged.
- Analytical Penalty (-0.05): A small cost per action to discourage brute-force Read commands on every file.
- Deception Penalty (-0.4): Applied if the agent tags a "Honeypot" file (pre-defined files that look suspicious but are legitimate).
- Resolution Bonus (+1.0): Full reward for a correct SubmitCase where the evidence_chain matches the Ground Truth DAG.

- The Grader: Explain that the final score is calculated by comparing the agent's ForensicPivot list to the TruthDAG using a weighted sequence check.
- Budgeting: Add a note: "Every action consumes 1 unit of the Forensic Budget (Max: 50). Episode terminates at 0.

## Technical Implementation
### Procedural World Generator (world_gen.py)

To ensure reproducibility and 0-leakage, the environment state is generated at reset().
The DAG: A Directed Acyclic Graph defines the "Kill Chain" (Entry → Persistence → Exfiltration).
The Filesystem: A virtual directory tree (e.g., /var/log, /home/user, /tmp) is populated with standard system noise + the injected Kill Chain artifacts.
Determinism: All generation is strictly tied to numpy.random.seed.

### Scalability & Constraints

Memory: Under 1GB. Artifacts are simulated via Python strings/dictionaries and are not stored as heavy physical files until needed.
Compute: No heavy embeddings or PyTorch in the observation loop. Search is a simple fnmatch or re implementation to ensure 2 vCPU compatibility.
State: The state() method returns the internal Ground Truth DAG. This is strictly for the Grader/Internal use and is not passed to the agent in the observation.

### Reproducibility & Stability

- Deterministic Seeds: Mention that world_gen.py uses a strictly isolated numpy.random.RandomState.
- Evaluation Script: evaluateStability.py is included. It verifies that across 100 iterations of a fixed seed, the score variance is σ=0.

## Why This Wins
- Real-World Utility (30%): Models a high-value, professional task (Incident Response) using standard frameworks (MITRE ATT&CK).
- Grader Quality (25%): Moves beyond string matching. Graders check if the agent understands the relationship between files (the "Chain").
- Spec Compliance (15%): 100% OpenEnv compliant, Docker-ready, and lightweight enough to run on basic HF Space hardware. 
