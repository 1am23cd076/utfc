"""
UnifiedThreatFusionCenter — environment.py
Core RL environment logic implementing OpenEnv step/reset/state.
"""

from __future__ import annotations
import json
import copy
from typing import Any, Dict, List, Optional

from models import (
    FusionState, SOCAction, SOCObservation, StepResult, ResetResult,
    ActionType, DBStatus, TASK_DEFINITIONS
)
from threat_generator import ThreatGenerator
from graders import GRADERS


class UnifiedThreatFusionCenter:
    """
    Security Operations Center RL Environment.
    Single autonomous agent detects, correlates, and mitigates
    cyber and physical threats across a virtual corporate campus.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.generator = ThreatGenerator(seed=seed)
        self.state: Optional[FusionState] = None
        self.task_id: str = "easy"
        self.action_history: List[Dict[str, Any]] = []
        self.episode_reward: float = 0.0

    # ─────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────

    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> ResetResult:
        """Start a new episode for the given task."""
        if seed is not None:
            self.seed = seed
            self.generator = ThreatGenerator(seed=seed)

        self.task_id = task_id
        task = TASK_DEFINITIONS[task_id]

        self.state = FusionState(time_remaining=task["max_steps"])
        self.action_history = []
        self.episode_reward = 0.0

        # Inject threats for this task
        self.state = self.generator.inject(self.state, task["threat_pool"])

        obs = self._build_observation(last_action_result="Environment reset. SOC monitoring active.")

        return ResetResult(
            observation=obs,
            done=False,
            info={"task": task_id, "seed": self.seed}
        )

    # ─────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────

    def step(self, action: SOCAction) -> StepResult:
        """Execute one agent action and return next observation + reward."""
        assert self.state is not None, "Call reset() before step()"

        reward = 0.0
        action_result = ""
        done = False

        self.state.episode_step += 1
        self.state.time_remaining -= 1

        # ── Execute action ──
        reward, action_result = self._execute_action(action)

        # ── Escalate only if unresolved threats remain ──
        unresolved = [t for t in self.state.active_threats
                      if t not in self.state.resolved_threats]
        if unresolved:
            self.state = self.generator.escalate(self.state)

        # ── Inaction penalty ──
        if self.state.threat_level > 0.5 and unresolved:
            reward -= 0.1

        # ── Log action ──
        self.action_history.append({
            "step": self.state.episode_step,
            "action_type": action.action_type.value,
            "target": action.target,
            "reward": reward,
        })
        self.episode_reward += reward

        # ── Update memory buffer ──
        entry = f"Step {self.state.episode_step}: {action.action_type.value} → {action_result[:80]}"
        self.state.memory_buffer = (self.state.memory_buffer + [entry])[-5:]

        # ── Check terminal conditions ──
        breached, breach_reason = self.generator.check_breach(self.state)
        if breached:
            reward -= 10.0
            action_result += f" | TERMINAL BREACH: {breach_reason}"
            done = True

        if self.state.time_remaining <= 0:
            done = True

        # Auto-end: all threats resolved — no need to burn remaining steps
        all_resolved = (
            len(self.state.resolved_threats) >= len(self.state.active_threats)
            and len(self.state.active_threats) > 0
        )
        if all_resolved:
            done = True

        # ── Terminal success reward ──
        if done and not breached:
            all_resolved = len(self.state.resolved_threats) == len(self.state.active_threats)
            if all_resolved:
                reward += 10.0
                action_result += " | EPISODE SUCCESS: All threats resolved!"

        # ── Final grader score ──
        grader_score = 0.0
        if done:
            grader_fn = GRADERS.get(self.task_id, GRADERS["easy"])
            grader_score = grader_fn(self.state, self.action_history)

        obs = self._build_observation(last_action_result=action_result)

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=done,
            info={
                "episode_reward": round(self.episode_reward, 4),
                "grader_score": grader_score if done else None,
                "resolved_threats": self.state.resolved_threats,
                "active_threats": self.state.active_threats,
                "breach": breached,
            }
        )

    # ─────────────────────────────────────────────
    # state()
    # ─────────────────────────────────────────────

    def state_snapshot(self) -> Dict[str, Any]:
        """Return current full environment state as dict."""
        assert self.state is not None, "Call reset() before state()"
        return {
            "task_id": self.task_id,
            "episode_step": self.state.episode_step,
            "threat_level": self.state.threat_level,
            "time_remaining": self.state.time_remaining,
            "active_threats": self.state.active_threats,
            "resolved_threats": self.state.resolved_threats,
            "db_status": self.state.db_status.value,
            "cyber_logs": self.state.cyber_logs[-5:],
            "memory_buffer": self.state.memory_buffer,
            "episode_reward": round(self.episode_reward, 4),
        }

    # ─────────────────────────────────────────────
    # Action Execution
    # ─────────────────────────────────────────────

    def _execute_action(self, action: SOCAction):
        """Dispatch action and return (reward, result_string)."""
        at = action.action_type
        target = action.target or ""

        if at == ActionType.SCAN_CYBER:
            return self._scan_cyber(target)
        elif at == ActionType.PATCH_VULN:
            return self._patch_vuln(target)
        elif at == ActionType.BLOCK_PORT:
            return self._block_port(target)
        elif at == ActionType.ALERT_TEAM:
            return self._alert_team()
        elif at == ActionType.QUARANTINE_NODE:
            return self._quarantine_node(target)
        elif at == ActionType.DEPLOY_DRONE:
            return self._deploy_drone()
        elif at == ActionType.VERIFY_ACCESS:
            return self._verify_access()
        elif at == ActionType.LOCKDOWN_ZONE:
            return self._lockdown_zone()
        return 0.0, "Unknown action — no effect."

    def _scan_cyber(self, target: str):
        resolved = []
        reward = 0.0
        for threat in ["spam_flood", "db_breach"]:
            if threat in self.state.active_threats and threat not in self.state.resolved_threats:
                self.state.resolved_threats.append(threat)
                self.state.threat_level = max(0.0, self.state.threat_level - 0.15)
                reward += 1.0
                resolved.append(threat)
        if resolved:
            return reward, f"Scan completed. Neutralized: {resolved}"
        return 0.0, "Scan found no actionable threats."

    def _patch_vuln(self, target: str):
        reward = 0.0
        patched = []
        for node in self.state.network_graph["nodes"]:
            if (not target or node["node_id"] == target) and node["status"] == "vulnerable":
                node["status"] = "healthy"
                node["vulnerability_score"] = 0.0
                reward += 1.0
                patched.append(node["node_id"])
        if "exploit_chain" in self.state.active_threats and patched:
            if "exploit_chain" not in self.state.resolved_threats:
                self.state.resolved_threats.append("exploit_chain")
                self.state.threat_level = max(0.0, self.state.threat_level - 0.2)
                reward += 0.5
        if patched:
            return reward, f"Patched nodes: {patched}"
        return 0.0, "No vulnerable nodes found to patch."

    def _block_port(self, target: str):
        reward = 0.0
        resolved = []
        for threat in ["ddos", "phishing"]:
            if threat in self.state.active_threats and threat not in self.state.resolved_threats:
                self.state.resolved_threats.append(threat)
                self.state.threat_level = max(0.0, self.state.threat_level - 0.15)
                reward += 1.0
                resolved.append(threat)
        if resolved:
            return reward, f"Port blocked. Mitigated: {resolved}"
        return 0.0, "No active threats require port blocking."

    def _alert_team(self):
        reward = 0.0
        resolved = []
        for threat in ["cctv_anomaly", "card_swipe_mismatch"]:
            if threat in self.state.active_threats and threat not in self.state.resolved_threats:
                self.state.resolved_threats.append(threat)
                self.state.threat_level = max(0.0, self.state.threat_level - 0.1)
                reward += 0.5
                resolved.append(threat)
        # False positive penalty
        if "false_positive" in self.state.active_threats:
            self.state.false_alarm_triggered = True
            reward -= 0.5
            return reward, f"Team alerted for: {resolved} | WARNING: False alarm also triggered!"
        if resolved:
            return reward, f"Security team alerted. Responding to: {resolved}"
        return 0.3, "Team alerted. Partial credit for documentation."

    def _quarantine_node(self, target: str):
        reward = 0.0
        resolved = []
        for threat in ["phishing", "exploit_chain", "drone_thermal", "db_breach"]:
            if threat in self.state.active_threats and threat not in self.state.resolved_threats:
                self.state.resolved_threats.append(threat)
                self.state.threat_level = max(0.0, self.state.threat_level - 0.15)
                reward += 1.0
                resolved.append(threat)
        for node in self.state.network_graph["nodes"]:
            if not target or node["node_id"] == target:
                node["status"] = "quarantined"
        if resolved:
            return reward, f"Node quarantined. Contained: {resolved}"
        return 0.1, "Node quarantined but no active threats mitigated."

    def _deploy_drone(self):
        reward = 0.0
        resolved = []
        for threat in ["cctv_anomaly", "drone_thermal"]:
            if threat in self.state.active_threats and threat not in self.state.resolved_threats:
                self.state.resolved_threats.append(threat)
                self.state.threat_level = max(0.0, self.state.threat_level - 0.1)
                reward += 0.8
                resolved.append(threat)
        self.state.physical_sensors.drone_telemetry.alert_flag = False
        if resolved:
            return reward, f"Drone repositioned. Physical threats assessed: {resolved}"
        return 0.2, "Drone deployed. No new physical threats detected."

    def _verify_access(self):
        reward = 0.0
        resolved = []
        ac = self.state.access_control
        for threat in ["card_swipe_mismatch", "physical_fusion_breach"]:
            if threat in self.state.active_threats and threat not in self.state.resolved_threats:
                self.state.resolved_threats.append(threat)
                self.state.threat_level = max(0.0, self.state.threat_level - 0.12)
                reward += 0.8
                resolved.append(threat)
        # False positive: IT employee with valid badge
        if "false_positive" in self.state.active_threats and ac.authorization_status == "granted":
            if "false_positive" not in self.state.resolved_threats:
                self.state.resolved_threats.append("false_positive")
                reward += 0.3
        if resolved:
            return reward, f"Access verified. Status: {ac.authorization_status}. Threats handled: {resolved}"
        return 0.2, f"Access verification complete. Card {ac.card_id}: {ac.authorization_status}"

    def _lockdown_zone(self):
        reward = 0.0
        # False positive: penalize hard
        if "false_positive" in self.state.active_threats:
            self.state.false_alarm_triggered = True
            reward -= 0.5
        # Physical fusion breach: correct action
        if "physical_fusion_breach" in self.state.active_threats:
            if "physical_fusion_breach" not in self.state.resolved_threats:
                self.state.resolved_threats.append("physical_fusion_breach")
                self.state.threat_level = max(0.0, self.state.threat_level - 0.25)
                reward += 1.5
                return reward, "LOCKDOWN INITIATED. Physical breach contained. All zones secured."
        if reward < 0:
            return reward, "LOCKDOWN INITIATED — but this was a false alarm. Unnecessary disruption."
        return 0.5, "Lockdown initiated. Partial containment."

    # ─────────────────────────────────────────────
    # Observation Builder
    # ─────────────────────────────────────────────

    def _build_observation(self, last_action_result: str = "") -> SOCObservation:
        s = self.state
        task = TASK_DEFINITIONS[self.task_id]

        # Fused text observation
        parts = []

        if s.cyber_logs:
            parts.append(f"CYBER_LOGS: {' | '.join(s.cyber_logs[-3:])}")

        cctv = s.physical_sensors.cctv_summary
        if "ALERT" in cctv or "CRITICAL" in cctv:
            parts.append(f"CCTV: {cctv}")

        drone = s.physical_sensors.drone_telemetry
        if drone.alert_flag:
            parts.append(
                f"DRONE: Thermal spike {drone.thermal_signature}°C at ({drone.x}, {drone.y}) — ALERT"
            )

        ac = s.access_control
        if ac.authorization_status in ("mismatch", "denied"):
            parts.append(
                f"ACCESS_CONTROL: Card {ac.card_id} from {ac.attempted_org} — {ac.authorization_status.upper()}"
            )

        if s.db_status != DBStatus.NORMAL:
            parts.append(f"DATABASE: {s.db_status.value.upper()} | Recent: {s.recent_query_logs[-1] if s.recent_query_logs else 'none'}")

        if s.memory_buffer:
            parts.append(f"MEMORY: {' | '.join(s.memory_buffer[-2:])}")

        parts.append(f"THREAT_LEVEL: {s.threat_level:.2f} | TIME_REMAINING: {s.time_remaining} steps")

        obs_text = "\n".join(parts) if parts else "All systems nominal. Monitoring active."

        return SOCObservation(
            observation_text=obs_text,
            threat_level=s.threat_level,
            time_remaining=s.time_remaining,
            available_actions=[a.value for a in ActionType],
            last_action_result=last_action_result,
            task_id=self.task_id,
            goal=task["goal"],
        )
