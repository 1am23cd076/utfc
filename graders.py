"""
UnifiedThreatFusionCenter — graders.py
Deterministic rule-based graders for all 3 tasks.
Each grader returns a score from 0.0 to 1.0.
"""

from __future__ import annotations
from typing import Dict, List, Any
from models import FusionState, DBStatus, ActionType


def grade_easy(state: FusionState, action_history: List[Dict[str, Any]]) -> float:
    """
    Easy task: Single threat correctly identified and mitigated.
    Max score: 1.0
    """
    score = 0.0
    actions_taken = [a["action_type"] for a in action_history]
    threats = state.active_threats

    # Base: threat resolved
    resolved_ratio = len(state.resolved_threats) / max(len(threats), 1)
    score += resolved_ratio * 0.6

    # Bonus: correct action used for the threat type
    correct_action_used = _check_correct_actions(threats, actions_taken)
    score += correct_action_used * 0.2

    # Bonus: resolved quickly (under 15 steps)
    if len(action_history) <= 15 and resolved_ratio >= 1.0:
        score += 0.1

    # Bonus: no false alarms
    if not state.false_alarm_triggered:
        score += 0.1

    return round(min(score, 1.0), 4)


def grade_medium(state: FusionState, action_history: List[Dict[str, Any]]) -> float:
    """
    Medium task: 3 simultaneous threats across domains.
    Max score: 1.0
    """
    score = 0.0
    actions_taken = [a["action_type"] for a in action_history]
    threats = state.active_threats

    # Base: proportion of threats resolved
    resolved_ratio = len(state.resolved_threats) / max(len(threats), 1)
    score += resolved_ratio * 0.5

    # Bonus: correct action types used
    correct_action_used = _check_correct_actions(threats, actions_taken)
    score += correct_action_used * 0.2

    # Bonus: cyber + physical threats both addressed (fusion credit)
    cyber_threats = {"ddos", "exploit_chain", "db_breach", "phishing", "spam_flood"}
    physical_threats = {"cctv_anomaly", "drone_thermal", "card_swipe_mismatch"}
    cyber_active = [t for t in threats if t in cyber_threats]
    physical_active = [t for t in threats if t in physical_threats]
    cyber_resolved = [t for t in state.resolved_threats if t in cyber_threats]
    physical_resolved = [t for t in state.resolved_threats if t in physical_threats]

    if cyber_active and cyber_resolved:
        score += 0.1
    if physical_active and physical_resolved:
        score += 0.1

    # Penalty: false alarm
    if state.false_alarm_triggered:
        score -= 0.2

    # Bonus: threat_level stayed below 0.8
    if state.threat_level < 0.8:
        score += 0.1

    return round(max(0.0, min(score, 1.0)), 4)


def grade_hard(state: FusionState, action_history: List[Dict[str, Any]]) -> float:
    """
    Hard task: Full fusion breach — all systems triggered.
    Max score: 1.0
    """
    score = 0.0
    actions_taken = [a["action_type"] for a in action_history]
    threats = state.active_threats

    # Base: proportion of threats resolved
    resolved_ratio = len(state.resolved_threats) / max(len(threats), 1)
    score += resolved_ratio * 0.4

    # Bonus: no breach (threat_level never hit 1.0)
    if state.threat_level < 1.0:
        score += 0.15

    # Bonus: DB not breached
    if state.db_status != DBStatus.CONFIRMED_BREACH:
        score += 0.1

    # Bonus: all nodes healthy or patched
    all_nodes_ok = all(
        n["status"] != "compromised"
        for n in state.network_graph["nodes"]
    )
    if all_nodes_ok:
        score += 0.1

    # Bonus: fusion reasoning — used deploy_drone + verify_access together
    if (ActionType.DEPLOY_DRONE.value in actions_taken and
            ActionType.VERIFY_ACCESS.value in actions_taken):
        score += 0.1

    # Bonus: lockdown used appropriately (physical_fusion_breach was active)
    if ("physical_fusion_breach" in threats and
            ActionType.LOCKDOWN_ZONE.value in actions_taken):
        score += 0.05

    # Penalty: false alarm triggered
    if state.false_alarm_triggered:
        score -= 0.3

    # Penalty: lockdown on false positive
    if ("false_positive" in threats and
            ActionType.LOCKDOWN_ZONE.value in actions_taken and
            "false_positive" not in state.resolved_threats):
        score -= 0.2

    # Bonus: correct action types used
    correct_action_used = _check_correct_actions(threats, actions_taken)
    score += correct_action_used * 0.1

    return round(max(0.0, min(score, 1.0)), 4)


def _check_correct_actions(
    threats: List[str],
    actions_taken: List[str]
) -> float:
    """
    Returns ratio of threats that had at least one correct action applied.
    """
    CORRECT_ACTIONS = {
        "spam_flood":             [ActionType.SCAN_CYBER.value],
        "phishing":               [ActionType.BLOCK_PORT.value, ActionType.QUARANTINE_NODE.value],
        "ddos":                   [ActionType.BLOCK_PORT.value],
        "exploit_chain":          [ActionType.PATCH_VULN.value, ActionType.QUARANTINE_NODE.value],
        "card_swipe_mismatch":    [ActionType.VERIFY_ACCESS.value, ActionType.ALERT_TEAM.value],
        "cctv_anomaly":           [ActionType.DEPLOY_DRONE.value, ActionType.ALERT_TEAM.value],
        "drone_thermal":          [ActionType.DEPLOY_DRONE.value, ActionType.QUARANTINE_NODE.value],
        "db_breach":              [ActionType.SCAN_CYBER.value, ActionType.QUARANTINE_NODE.value],
        "physical_fusion_breach": [ActionType.LOCKDOWN_ZONE.value, ActionType.VERIFY_ACCESS.value],
        "false_positive":         [],  # correct action = do nothing / scan only
    }

    correct_count = 0
    for threat in threats:
        required = CORRECT_ACTIONS.get(threat, [])
        if not required:
            # false_positive: reward NOT doing lockdown
            if ActionType.LOCKDOWN_ZONE.value not in actions_taken:
                correct_count += 1
        elif any(a in actions_taken for a in required):
            correct_count += 1

    return correct_count / max(len(threats), 1)


GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}
