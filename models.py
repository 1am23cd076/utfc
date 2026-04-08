"""
UnifiedThreatFusionCenter — models.py
All Pydantic data models for the OpenEnv environment.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class DBStatus(str, Enum):
    NORMAL = "normal"
    SUSPICIOUS_QUERY = "suspicious_query"
    CONFIRMED_BREACH = "confirmed_breach"


class ActionType(str, Enum):
    SCAN_CYBER      = "scan_cyber"       # Analyze logs, return threat details
    PATCH_VULN      = "patch_vuln"       # Remediate exploit on a node/service
    BLOCK_PORT      = "block_port"       # Isolate traffic on a port (counters DDoS)
    ALERT_TEAM      = "alert_team"       # Escalate to human responders
    QUARANTINE_NODE = "quarantine_node"  # Sever connectivity of suspicious server
    DEPLOY_DRONE    = "deploy_drone"     # Reposition drone for fresh physical intel
    VERIFY_ACCESS   = "verify_access"    # Cross-check card swipe vs org database
    LOCKDOWN_ZONE   = "lockdown_zone"    # Full facility isolation


# ─────────────────────────────────────────────
# Sub-models for FusionState
# ─────────────────────────────────────────────

class DroneTelemetry(BaseModel):
    x: float = 0.0
    y: float = 0.0
    thermal_signature: float = 36.5   # Celsius — normal body temp baseline
    alert_flag: bool = False


class PhysicalSensors(BaseModel):
    cctv_summary: str = "All clear. No anomalies detected."
    drone_telemetry: DroneTelemetry = Field(default_factory=DroneTelemetry)


class AccessControlEvent(BaseModel):
    card_id: str = "NONE"
    attempted_org: str = "NONE"
    swipe_timestamp: str = "NONE"
    authorization_status: str = "none"   # "granted", "denied", "mismatch", "none"


class NetworkNode(BaseModel):
    node_id: str
    status: str = "healthy"             # "healthy", "vulnerable", "compromised"
    vulnerability_score: float = 0.0    # 0.0 to 1.0


# ─────────────────────────────────────────────
# Core State Model
# ─────────────────────────────────────────────

class FusionState(BaseModel):
    """Complete state of the SOC simulation."""

    threat_level: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Cumulative normalized risk across all domains"
    )
    network_graph: Dict[str, Any] = Field(
        default_factory=lambda: {
            "nodes": [
                {"node_id": "web_server",  "status": "healthy", "vulnerability_score": 0.0},
                {"node_id": "db_server",   "status": "healthy", "vulnerability_score": 0.0},
                {"node_id": "auth_server", "status": "healthy", "vulnerability_score": 0.0},
                {"node_id": "internal_api","status": "healthy", "vulnerability_score": 0.0},
            ],
            "edges": [
                ["web_server", "internal_api"],
                ["internal_api", "db_server"],
                ["internal_api", "auth_server"],
            ]
        }
    )
    cyber_logs: List[str] = Field(
        default_factory=list,
        description="Timestamped strings of incoming cyber threats"
    )
    physical_sensors: PhysicalSensors = Field(default_factory=PhysicalSensors)
    access_control: AccessControlEvent = Field(default_factory=AccessControlEvent)
    db_status: DBStatus = DBStatus.NORMAL
    recent_query_logs: List[str] = Field(default_factory=list)
    memory_buffer: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Last 5 fused alerts for temporal reasoning"
    )
    time_remaining: int = Field(default=100, ge=0)

    # Internal tracking flags (not exposed to agent directly)
    active_threats: List[str] = Field(
        default_factory=list,
        description="Which threat scenarios are currently active"
    )
    resolved_threats: List[str] = Field(default_factory=list)
    false_alarm_triggered: bool = False
    episode_step: int = 0


# ─────────────────────────────────────────────
# Action Model
# ─────────────────────────────────────────────

class SOCAction(BaseModel):
    """Agent action in the SOC environment."""
    action_type: ActionType
    target: Optional[str] = Field(
        default=None,
        description="Target node, port, zone, or card_id depending on action"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra parameters for the action"
    )


# ─────────────────────────────────────────────
# Observation Model
# ─────────────────────────────────────────────

class SOCObservation(BaseModel):
    """
    What the agent sees after each step.
    Single fused text snapshot for end-to-end reasoning.
    """
    observation_text: str = Field(
        description="Fused human-readable JSON-like string of all active state elements"
    )
    threat_level: float = Field(ge=0.0, le=1.0)
    time_remaining: int
    available_actions: List[str] = Field(
        default_factory=lambda: [a.value for a in ActionType]
    )
    last_action_result: Optional[str] = None
    task_id: str = "easy"
    goal: str = ""


# ─────────────────────────────────────────────
# Step Result (returned by step())
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    observation: SOCObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Reset Result (returned by reset())
# ─────────────────────────────────────────────

class ResetResult(BaseModel):
    observation: SOCObservation
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────────

TASK_DEFINITIONS = {
    "easy": {
        "id": "easy",
        "name": "Single Threat Response",
        "description": (
            "A single isolated cyber or physical threat appears. "
            "The agent must identify and mitigate it within 30 steps."
        ),
        "max_steps": 30,
        "threat_count": 1,
        "threat_pool": ["spam_flood", "phishing", "ddos", "card_swipe_mismatch"],
        "difficulty_score": 0.3,
        "goal": (
            "Detect the active threat and apply the correct mitigation action. "
            "Resolve before threat_level reaches 1.0."
        ),
    },
    "medium": {
        "id": "medium",
        "name": "Multi-Vector Attack",
        "description": (
            "Three simultaneous threats across cyber and physical domains. "
            "The agent must prioritize and resolve all within 60 steps."
        ),
        "max_steps": 60,
        "threat_count": 3,
        "threat_pool": ["ddos", "exploit_chain", "cctv_anomaly", "db_breach", "drone_thermal"],
        "difficulty_score": 0.6,
        "goal": (
            "Detect, correlate, and mitigate three simultaneous threats. "
            "Prioritize by threat_level contribution. Avoid false alarms."
        ),
    },
    "hard": {
        "id": "hard",
        "name": "Full Fusion Breach",
        "description": (
            "All systems are triggered simultaneously — CCTV, drone, card mismatch, "
            "DDoS, exploit chain, and database breach. One false alarm costs heavily. "
            "Agent must sequence actions perfectly within 100 steps."
        ),
        "max_steps": 100,
        "threat_count": 6,
        "threat_pool": [
            "spam_flood", "phishing", "ddos", "exploit_chain",
            "card_swipe_mismatch", "cctv_anomaly", "drone_thermal",
            "db_breach", "physical_fusion_breach", "false_positive"
        ],
        "difficulty_score": 1.0,
        "goal": (
            "Handle a full-scale coordinated cyber-physical breach. "
            "Correlate all fusion signals. Do NOT trigger false alarms. "
            "Achieve zero breaches before time runs out."
        ),
    },
}
