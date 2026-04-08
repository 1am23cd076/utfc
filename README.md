---
title: UnifiedThreatFusionCenter
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
tags:
  - openenv
---

# 🛡️ UnifiedThreatFusionCenter

**An OpenEnv-compliant Reinforcement Learning environment simulating a Security Operations Center (SOC)**

A fully text-based, cyber-physical threat fusion environment where an autonomous AI agent must detect, correlate, and mitigate up to 10 simultaneous threats across network, database, physical access, CCTV, and drone surveillance systems — with no external hardware, APIs, or vision models required.

---

## 🎯 Motivation

Security Operations Centers handle thousands of alerts daily across disconnected systems — network logs, CCTV feeds, access control events, and drone surveillance. No single analyst can correlate all of these in real time. This environment benchmarks whether AI agents can perform **multi-domain threat fusion** — the hardest unsolved problem in enterprise security.

This is the first OpenEnv environment to combine cyber and physical threat intelligence into a single RL benchmark.

---

## 🏗️ Environment Architecture

```
UnifiedThreatFusionCenter/
├── models.py            # All Pydantic data models (FusionState, SOCAction, SOCObservation)
├── threat_generator.py  # 10 randomized but reproducible threat scenarios
├── graders.py           # Deterministic graders for all 3 tasks (0.0–1.0)
├── environment.py       # Core RL logic — step() / reset() / state()
├── inference.py         # Baseline LLM agent script
├── openenv.yaml         # OpenEnv manifest
├── local_test.py        # Local validation (no API key needed)
└── server/
    ├── app.py           # FastAPI server
    ├── requirements.txt
    └── Dockerfile
```

---

## 🔍 Observation Space

At each step, the agent receives a **single fused text observation** combining all active sensor streams:

```
CYBER_LOGS: [DDoS] Flood detected on port 443 | Packet rate: 120,000/sec | ...
CCTV: ALERT: Unknown figure detected at east gate — movement pattern: evasive
DRONE: Thermal spike 39.2°C at (45.0, 67.0) — ALERT
ACCESS_CONTROL: Card CRD-9921 from UNKNOWN_VENDOR — MISMATCH
DATABASE: SUSPICIOUS_QUERY | Recent: SELECT * FROM users WHERE 1=1; --
MEMORY: Step 3: block_port → Port blocked | Step 4: deploy_drone → Drone repositioned
THREAT_LEVEL: 0.47 | TIME_REMAINING: 63 steps
```

**State fields (FusionState):**
| Field | Type | Description |
|---|---|---|
| `threat_level` | float [0,1] | Cumulative normalized risk |
| `network_graph` | dict | Nodes, edges, vulnerability scores |
| `cyber_logs` | list[str] | Timestamped threat log entries |
| `physical_sensors` | nested dict | CCTV summary + drone telemetry |
| `access_control` | dict | Card swipe event + authorization status |
| `db_status` | enum | normal / suspicious_query / confirmed_breach |
| `memory_buffer` | list[str] | Last 5 fused alerts for temporal reasoning |
| `time_remaining` | int | Steps left before episode ends |

---

## ⚡ Action Space

8 discrete high-level commands:

| Action | Use for | Effect |
|---|---|---|
| `scan_cyber` | spam_flood, db_breach | Analyzes logs, neutralizes recon threats |
| `patch_vuln` | exploit_chain | Repairs vulnerable nodes |
| `block_port` | ddos, phishing | Isolates malicious traffic |
| `alert_team` | cctv_anomaly, card mismatch | Escalates to human responders |
| `quarantine_node` | exploit_chain, db_breach | Severs compromised server |
| `deploy_drone` | cctv_anomaly, drone_thermal | Repositions drone for physical intel |
| `verify_access` | card_swipe_mismatch | Cross-checks badge vs org database |
| `lockdown_zone` | physical_fusion_breach ONLY | Full facility isolation |

---

## 🎯 Tasks

### Task 1 — Easy: Single Threat Response
One isolated threat from the cyber or physical domain. Agent must identify and apply the correct action within 30 steps.

**Example:** A DDoS flood detected on port 443. Agent must `block_port`.

**Expected difficulty:** A capable LLM agent solves this in 5–10 steps.

### Task 2 — Medium: Multi-Vector Attack
Three simultaneous threats spanning cyber and physical domains. Agent must prioritize by severity and resolve all within 60 steps.

**Example:** DDoS + exploit chain + CCTV anomaly active simultaneously.

**Expected difficulty:** Requires correct ordering. Naive agents over-focus on one domain.

### Task 3 — Hard: Full Fusion Breach
All 10 threat types active simultaneously, including a **false positive** that penalizes overreaction. Agent must perfectly sequence actions within 100 steps. One wrong lockdown loses 0.5 points.

**Expected difficulty:** Challenges frontier models. Most agents score below 0.5.

---

## 📊 Reward Function

```
reward = base_action_reward        # +1.0 for correct threat-matched action
       + speed_bonus               # +0.3 if resolved within 10 steps
       - missed_detection_penalty  # -0.5 if threat escalates unchecked
       - false_positive_penalty    # -0.5 for unnecessary lockdown
       + fusion_correlation_bonus  # +0.2 for correlating multiple sources
       - inaction_penalty          # -0.1/step when threat_level > 0.5
       ± terminal_reward           # +10.0 success / -10.0 full breach
```

Reward range: **[-10.0, +10.0]** per episode.

---

## 🚀 Setup & Usage

### Local (no Docker)

```bash
# Install dependencies
pip install fastapi uvicorn pydantic openenv-core

# Run local test (no API key needed)
python local_test.py

# Start the server
cd server
python app.py
```

### Docker

```bash
# Build
docker build -f server/Dockerfile -t unified-threat-fusion-center .

# Run
docker run -p 7860:7860 unified-threat-fusion-center
```

### Test the API

```bash
# Health check
curl http://localhost:7860/health

# Reset (start easy task)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "scan_cyber", "target": null}'

# Get state
curl http://localhost:7860/state
```

---

## 🤖 Running the Baseline Agent

```bash
# Set environment variables
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_groq_api_key"
export HF_SPACE_URL="https://your-space.hf.space"

# Run baseline
python inference.py
```

### Expected baseline scores

| Task | Expected Grader Score |
|---|---|
| Easy | 0.70 – 0.90 |
| Medium | 0.45 – 0.65 |
| Hard | 0.20 – 0.45 |

---

## 🏆 What Makes This Novel

1. **First cyber-physical fusion RL environment in OpenEnv** — no prior art
2. **False positive trap** — penalizes overreaction, forces nuanced reasoning
3. **Fused text observation** — agent reasons across 6 simultaneous sensor streams
4. **Deterministic reproducibility** — seed parameter ensures identical replays
5. **No hardware, no vision models, no external APIs** — runs on any laptop

---

## 🔒 10 Threat Scenarios

| # | Scenario | Type | Correct Action |
|---|---|---|---|
| 1 | Spam email flood | Cyber | scan_cyber |
| 2 | Phishing attempt | Cyber | block_port |
| 3 | DDoS flood | Cyber | block_port |
| 4 | Exploit chain | Cyber | patch_vuln + quarantine_node |
| 5 | Card swipe mismatch | Physical | verify_access |
| 6 | CCTV anomaly | Physical | deploy_drone + alert_team |
| 7 | Drone thermal spike | Physical | deploy_drone |
| 8 | Database breach | Cyber | scan_cyber + quarantine_node |
| 9 | Physical fusion breach | Combined | verify_access + lockdown_zone |
| 10 | False positive | Trap | Do NOT lockdown |
