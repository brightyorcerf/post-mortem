from __future__ import annotations
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

# --- CORE ENUMS (grammar) ---

class IOCType(str, Enum):
    NETWORK_IP = "NETWORK_IP"
    EVENT_TIMESTAMP = "EVENT_TIMESTAMP"
    FILE_PATH = "PATH_TO_FILE"
    COMMAND_STRING = "COMMAND_STRING"
    USER_ACCOUNT = "USER_ACCOUNT"
    FILE_HASH = "FILE_HASH"

class ActionType(str, Enum):
    SEARCH = "Search"
    INSPECT = "Inspect"
    READ = "Read"
    TAG = "Tag"
    SUBMIT = "SubmitCase"

# --- VIRTUAL FILESYSTEM  ---

class FileMetadata(BaseModel):
    """Standard Linux stat attributes for detecting Timestomping."""
    mtime: str  # modification: forged in hard task
    atime: str  # access
    ctime: str  # change: the "Truth" in the hard task
    uid: int = 0
    gid: int = 0
    size: int
    permissions: str = "-rw-r--r--"

class VirtualFile(BaseModel):
    """The raw artifact representation."""
    path: str
    content: str
    metadata: FileMetadata

# --- OBSERVATION SPACE ---

class SearchResult(BaseModel):
    """Response per file for global keyword searches."""
    filename: str
    hit_count: int
    relevance_score: float  # low score indicates likely "System Noise"

class ForensicObs(BaseModel):
    """The strictly typed state passed to the agent every step."""
    current_view: str = Field(..., max_length=1000)
    working_directory: str
    artifact_metadata: Optional[FileMetadata] = None
    tagged_evidence: Dict[str, str] = Field(default_factory=dict)
    remaining_budget: int = Field(default=50)
    last_action_log: str = ""

# --- THE ACTION SPACE (Agent's Input) ---

class ForensicPivot(BaseModel):
    """A single piece of evidence marked by the analyst."""
    artifact: str
    ioc: str
    type: IOCType
    reason: str

class ForensicAction(BaseModel):
    """The command issued by the agent."""
    action: ActionType
    query: Optional[str] = None      # For Search
    path: Optional[str] = None       # For Read/Inspect
    offset: Optional[int] = 0        # For Read (chunking)
    label: Optional[str] = None      # For Tag
    value: Optional[str] = None      # For Tag
    pivots: Optional[List[ForensicPivot]] = None # For SubmitCase

# --- THE GROUND TRUTH  ---

class TruthNode(BaseModel):
    """A deterministic point in the Kill Chain DAG."""
    node_id: str
    required_artifact: str
    expected_ioc: str
    type: IOCType
    is_honeypot: bool = False  # if True, penalty -0.4 if tagged
    weight: float = 1.0

class TruthDAG(BaseModel):
    """The 'Answer Key' for the scenario."""
    scenario_name: str
    seed: int
    nodes: Dict[str, TruthNode]
    edges: List[tuple[str, str]] # list of (from_node, to_node) connections

# --- ENVIRONMENT STATE (Internal Only) ---

class InternalState(BaseModel):
    """The master object Person B passes to Person A."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    filesystem: Dict[str, VirtualFile]
    truth_dag: TruthDAG
    history: List[ForensicAction] = Field(default_factory=list)