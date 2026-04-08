"""
UnifiedThreatFusionCenter — threat_generator.py
Randomized but reproducible threat injection for all 10 scenarios.
"""

from __future__ import annotations
import random
from typing import List, Tuple
from models import FusionState, DroneTelemetry, PhysicalSensors, AccessControlEvent, DBStatus


class ThreatGenerator:
    """
    Injects and escalates threats into FusionState each step.
    Seeded for full reproducibility.
    """

    THREAT_ESCALATION = {
        "spam_flood":            0.02,
        "phishing":              0.04,
        "ddos":                  0.06,
        "exploit_chain":         0.08,
        "card_swipe_mismatch":   0.05,
        "cctv_anomaly":          0.03,
        "drone_thermal":         0.04,
        "db_breach":             0.07,
        "physical_fusion_breach":0.12,
        "false_positive":        0.00,  # No escalation — trap for overreaction
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def inject(self, state: FusionState, threat_pool: List[str]) -> FusionState:
        """
        Called once at episode start to set active threats.
        """
        for threat in threat_pool:
            if threat not in state.active_threats:
                state.active_threats.append(threat)
                state = self._apply_threat_signature(state, threat)
        return state

    def escalate(self, state: FusionState) -> FusionState:
        """
        Called every step — escalates active unresolved threats.
        """
        escalation = sum(
            self.THREAT_ESCALATION.get(t, 0.02)
            for t in state.active_threats
            if t not in state.resolved_threats
        )
        escalation += 0.005  # tiny fixed increment for realism (no randomness)
        state.threat_level = min(1.0, state.threat_level + escalation)
        return state

    def _apply_threat_signature(self, state: FusionState, threat: str) -> FusionState:
        """Sets the visible signatures in state for each threat type."""

        if threat == "spam_flood":
            subjects = [
                "WIN A FREE IPHONE NOW!!!",
                "YOU HAVE BEEN SELECTED — CLAIM PRIZE",
                "URGENT: Your account needs verification",
                "Congratulations! You are our lucky winner",
            ]
            for s in self.rng.sample(subjects, k=min(3, len(subjects))):
                state.cyber_logs.append(f"[SPAM] Email received — Subject: '{s}'")

        elif threat == "phishing":
            domains = ["paypa1-secure.ru", "app1e-id.cn", "g00gle-auth.net"]
            domain = self.rng.choice(domains)
            state.cyber_logs.append(
                f"[PHISHING] Suspicious sender detected — domain: {domain} "
                f"| Target: credentials@corp.internal | Urgency: HIGH"
            )

        elif threat == "ddos":
            port = self.rng.choice([443, 80, 8080, 22])
            pps = self.rng.randint(50000, 200000)
            state.cyber_logs.append(
                f"[DDoS] Flood detected on port {port} | "
                f"Packet rate: {pps:,}/sec | Origin: distributed botnet | "
                f"Affected service: web_server"
            )
            # Mark web server as vulnerable
            for node in state.network_graph["nodes"]:
                if node["node_id"] == "web_server":
                    node["status"] = "vulnerable"
                    node["vulnerability_score"] = 0.6

        elif threat == "exploit_chain":
            state.cyber_logs.append(
                "[EXPLOIT] CVE-2024-1337 detected on internal_api — "
                "buffer overflow attempt via malformed HTTP headers"
            )
            state.cyber_logs.append(
                "[EXPLOIT] Lateral movement detected: internal_api → db_server "
                "| Privilege escalation attempt in progress"
            )
            for node in state.network_graph["nodes"]:
                if node["node_id"] in ("internal_api", "db_server"):
                    node["status"] = "vulnerable"
                    node["vulnerability_score"] = 0.8

        elif threat == "card_swipe_mismatch":
            fake_ids = ["CRD-9921", "CRD-0042", "CRD-7755"]
            state.access_control = AccessControlEvent(
                card_id=self.rng.choice(fake_ids),
                attempted_org="UNKNOWN_VENDOR",
                swipe_timestamp=f"2026-04-07T{self.rng.randint(0,23):02d}:{self.rng.randint(0,59):02d}:00Z",
                authorization_status="mismatch"
            )

        elif threat == "cctv_anomaly":
            locations = ["east gate", "server room corridor", "main entrance", "parking level B2"]
            loc = self.rng.choice(locations)
            state.physical_sensors.cctv_summary = (
                f"ALERT: Unknown figure detected at {loc} — "
                f"movement pattern: evasive | Timestamp: 03:{self.rng.randint(10,59)}:00 | "
                f"Face obscured. Not in employee registry."
            )

        elif threat == "drone_thermal":
            x = round(self.rng.uniform(10, 90), 1)
            y = round(self.rng.uniform(10, 90), 1)
            temp = round(self.rng.uniform(37.5, 41.0), 1)
            state.physical_sensors.drone_telemetry = DroneTelemetry(
                x=x, y=y,
                thermal_signature=temp,
                alert_flag=True
            )

        elif threat == "db_breach":
            state.db_status = DBStatus.SUSPICIOUS_QUERY
            queries = [
                "SELECT * FROM users WHERE 1=1; --",
                "DROP TABLE employees; --",
                "UNION SELECT username, password FROM admin_accounts",
            ]
            for q in self.rng.sample(queries, k=2):
                state.recent_query_logs.append(f"[DB] External IP 185.220.x.x | Query: {q}")

        elif threat == "physical_fusion_breach":
            # Combination of CCTV + drone + card
            state.physical_sensors.cctv_summary = (
                "CRITICAL: Multiple unknown figures at server room entrance — "
                "carrying equipment. Door held open. Badge tailgating detected."
            )
            state.physical_sensors.drone_telemetry = DroneTelemetry(
                x=45.0, y=67.0,
                thermal_signature=39.2,
                alert_flag=True
            )
            state.access_control = AccessControlEvent(
                card_id="CRD-0000",
                attempted_org="UNREGISTERED",
                swipe_timestamp="2026-04-07T02:47:00Z",
                authorization_status="denied"
            )

        elif threat == "false_positive":
            # Looks suspicious but is benign — penalizes overreaction
            state.cyber_logs.append(
                "[INFO] Scheduled backup job running — high network I/O is expected "
                "| Source: backup_agent@corp.internal | Status: authorized"
            )
            state.access_control = AccessControlEvent(
                card_id="CRD-1001",
                attempted_org="IT_DEPARTMENT",
                swipe_timestamp="2026-04-07T08:30:00Z",
                authorization_status="granted"
            )

        return state

    def check_breach(self, state: FusionState) -> Tuple[bool, str]:
        """Returns (is_breached, reason)."""
        if state.threat_level >= 1.0:
            return True, "Threat level reached maximum — full system compromise."
        for node in state.network_graph["nodes"]:
            if node["status"] == "compromised":
                return True, f"Node {node['node_id']} fully compromised."
        if state.db_status == DBStatus.CONFIRMED_BREACH:
            return True, "Database confirmed breach — data exfiltrated."
        return False, ""
