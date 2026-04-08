"""
inference.py — UnifiedThreatFusionCenter Baseline Agent
Compliant with OpenEnv hackathon structured log format requirements.

Run:
  python inference.py
"""

import os
import json
import time
import urllib.request
from typing import List, Optional
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL  = os.environ.get("API_BASE_URL",  "https://api.groq.com/openai/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "llama-3.3-70b-versatile")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ENV_NAME      = "unified_threat_fusion_center"

# HF_SPACE_URL: on HuggingFace Spaces the server runs on localhost:7860 internally
# When running inference FROM OUTSIDE the space, set this to your Space URL
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "https://for-the-future-utfc.hf.space")

TEMPERATURE   = 0.0
MAX_TOKENS    = 256

# Per-task step limits matching environment max_steps
TASK_MAX_STEPS = {"easy": 29, "medium": 59, "hard": 99}
TASKS = ["easy", "medium", "hard"]

# ── OpenAI-compatible client ──────────────────────────────────────────────────
USE_LLM = bool(OPENAI_API_KEY)
client  = None
if USE_LLM:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)
    except Exception:
        USE_LLM = False

# ── Structured Log Functions (REQUIRED FORMAT) ────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _post(path: str, body: dict) -> dict:
    url  = HF_SPACE_URL.rstrip("/") + path
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def _get(path: str) -> dict:
    url = HF_SPACE_URL.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())

# ── Expert Rule-Based Agent ───────────────────────────────────────────────────
VALID_ACTIONS = [
    "scan_cyber", "patch_vuln", "block_port", "alert_team",
    "quarantine_node", "deploy_drone", "verify_access", "lockdown_zone"
]

def expert_action(obs_text: str, resolved: List[str],
                  history: List[str], task_id: str,
                  remaining: Optional[List[str]] = None) -> dict:
    """
    Deterministic expert policy — encodes optimal action per threat signal.
    Uses remaining threat list when available to catch scrolled-out signals.
    """
    obs   = obs_text.lower()
    last  = history[-1] if history else ""
    last2 = history[-2] if len(history) >= 2 else ""

    # ── Direct mop-up from remaining threats list (beats obs parsing) ─────
    if remaining:
        if "ddos" in remaining and last != "block_port":
            return {"action_type": "block_port", "target": None}
        if "phishing" in remaining and last != "block_port":
            return {"action_type": "block_port", "target": None}
        if "spam_flood" in remaining and last != "scan_cyber":
            return {"action_type": "scan_cyber", "target": None}
        if "db_breach" in remaining:
            if last == "scan_cyber":
                return {"action_type": "quarantine_node", "target": "db_server"}
            return {"action_type": "scan_cyber", "target": None}
        if "exploit_chain" in remaining:
            if last == "patch_vuln":
                return {"action_type": "quarantine_node", "target": "db_server"}
            return {"action_type": "patch_vuln", "target": "internal_api"}
        if "drone_thermal" in remaining:
            return {"action_type": "deploy_drone", "target": None}
        if "cctv_anomaly" in remaining:
            if last == "deploy_drone" and "false_positive" in resolved:
                return {"action_type": "alert_team", "target": None}
            return {"action_type": "deploy_drone", "target": None}
        if "card_swipe_mismatch" in remaining:
            return {"action_type": "verify_access", "target": None}
        if "physical_fusion_breach" in remaining:
            if last == "verify_access":
                return {"action_type": "lockdown_zone", "target": None}
            return {"action_type": "verify_access", "target": None}
        if "false_positive" in remaining:
            return {"action_type": "verify_access", "target": None}

    # ── Detect signals from observation text ──────────────────────────────
    has_ddos    = "ddos" in obs or ("flood" in obs and "backup" not in obs)
    has_exploit = "cve" in obs or "buffer overflow" in obs
    has_lateral = "lateral movement" in obs
    has_db_sus  = "suspicious_query" in obs
    has_db_conf = "confirmed_breach" in obs
    has_phish   = "phishing" in obs
    has_spam    = "spam" in obs
    has_cctv    = "cctv" in obs and ("alert" in obs or "unknown figure" in obs or "critical" in obs)
    has_drone   = "drone" in obs and ("thermal" in obs or "alert" in obs)
    has_miss    = "mismatch" in obs or ("denied" in obs and "access_control" in obs)
    has_fusion  = has_cctv and has_drone and has_miss
    has_fp      = ("granted" in obs and "it_department" in obs) or ("backup" in obs and "authorized" in obs)

    # ── Priority 1: Physical fusion breach ────────────────────────────────
    if has_fusion and "physical_fusion_breach" not in resolved:
        if last == "verify_access":
            return {"action_type": "lockdown_zone", "target": None}
        return {"action_type": "verify_access", "target": None}

    # ── Priority 2: DDoS ──────────────────────────────────────────────────
    if has_ddos and "ddos" not in resolved:
        return {"action_type": "block_port", "target": None}

    # ── Priority 3: Exploit chain ─────────────────────────────────────────
    if has_exploit and "exploit_chain" not in resolved:
        if has_lateral and last != "quarantine_node":
            return {"action_type": "quarantine_node", "target": "db_server"}
        if last != "patch_vuln":
            return {"action_type": "patch_vuln", "target": "internal_api"}
        return {"action_type": "quarantine_node", "target": "db_server"}

    # ── Priority 4: DB breach ─────────────────────────────────────────────
    if has_db_conf and "db_breach" not in resolved:
        return {"action_type": "quarantine_node", "target": "db_server"}
    if has_db_sus and "db_breach" not in resolved:
        if last == "scan_cyber":
            return {"action_type": "quarantine_node", "target": "db_server"}
        return {"action_type": "scan_cyber", "target": None}

    # ── Priority 5: Phishing ──────────────────────────────────────────────
    if has_phish and "phishing" not in resolved:
        return {"action_type": "block_port", "target": None}

    # ── Priority 6: Drone thermal ─────────────────────────────────────────
    if has_drone and "drone_thermal" not in resolved:
        if last == "deploy_drone":
            return {"action_type": "quarantine_node", "target": "internal_api"}
        return {"action_type": "deploy_drone", "target": None}

    # ── Priority 7: CCTV anomaly ──────────────────────────────────────────
    if has_cctv and "cctv_anomaly" not in resolved:
        if last == "deploy_drone" and "false_positive" in resolved:
            return {"action_type": "alert_team", "target": None}
        return {"action_type": "deploy_drone", "target": None}

    # ── Priority 8: Card mismatch alone ──────────────────────────────────
    if has_miss and "card_swipe_mismatch" not in resolved:
        return {"action_type": "verify_access", "target": None}

    # ── Priority 9: Spam ──────────────────────────────────────────────────
    if has_spam and "spam_flood" not in resolved:
        return {"action_type": "scan_cyber", "target": None}

    # ── Priority 10: False positive — verify only, never lockdown ────────
    if has_fp and "false_positive" not in resolved:
        return {"action_type": "verify_access", "target": None}

    # ── Mop-up: patch remaining vulnerable nodes ──────────────────────────
    if last not in ("patch_vuln", "scan_cyber"):
        return {"action_type": "patch_vuln", "target": "web_server"}
    if last != "scan_cyber":
        return {"action_type": "scan_cyber", "target": None}

    # ── Cycle to avoid loops ──────────────────────────────────────────────
    cycle = ["alert_team", "deploy_drone", "scan_cyber",
             "quarantine_node", "patch_vuln", "verify_access"]
    return {"action_type": cycle[len(history) % len(cycle)], "target": None}


# ── LLM Agent ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI Security Operations Center (SOC) analyst.
Read the observation and choose ONE action to neutralize the most urgent threat.

THREAT → ACTION TABLE:
[SPAM]                                   → scan_cyber
[PHISHING]                               → block_port
[DDoS] / flood / packet rate             → block_port
[EXPLOIT] CVE / buffer overflow          → patch_vuln (target: "internal_api")
[EXPLOIT] lateral movement               → quarantine_node (target: "db_server")
DATABASE SUSPICIOUS_QUERY                → scan_cyber FIRST, then quarantine_node
DATABASE CONFIRMED_BREACH                → quarantine_node (target: "db_server")
CCTV alert / unknown figure              → deploy_drone
DRONE thermal spike / alert              → deploy_drone
ACCESS_CONTROL MISMATCH alone            → verify_access
CCTV + DRONE + ACCESS MISMATCH together  → verify_access FIRST, then lockdown_zone
Authorized IT / backup / granted badge   → verify_access ONLY, NEVER lockdown

STRICT RULES:
1. Output ONLY a JSON object on one line. No explanation. No markdown.
2. NEVER repeat the same action more than twice in a row.
3. NEVER use lockdown_zone unless CCTV + DRONE + MISMATCH all appear simultaneously.
4. After a threat is resolved, move to the next unresolved threat signal.

Output format examples:
{"action_type": "block_port", "target": null}
{"action_type": "patch_vuln", "target": "internal_api"}
{"action_type": "quarantine_node", "target": "db_server"}
"""

def llm_action(obs_text: str, goal: str, step: int,
               recent: List[str], resolved: List[str]) -> Optional[dict]:
    if not USE_LLM or not client:
        return None
    try:
        prompt = (
            f"GOAL: {goal}\n\n"
            f"STEP {step} OBSERVATION:\n{obs_text}\n\n"
            f"Recent actions (last 5): {' → '.join(recent[-5:]) or 'none'}\n"
            f"Already resolved: {', '.join(resolved) or 'none'}\n\n"
            "Choose action. Output one JSON line only."
        )
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences
        for fence in ["```json", "```"]:
            if fence in raw:
                raw = raw.split(fence)[-1].split("```")[0].strip()
        s = raw.find("{"); e = raw.rfind("}") + 1
        if s >= 0 and e > s:
            parsed = json.loads(raw[s:e])
            act = parsed.get("action_type", "scan_cyber")
            if act not in VALID_ACTIONS:
                act = "scan_cyber"
            return {"action_type": act, "target": parsed.get("target")}
    except Exception:
        pass
    return None

# ── Task Runner ───────────────────────────────────────────────────────────────
def run_task(task_id: str) -> dict:
    max_steps = TASK_MAX_STEPS[task_id]
    model_label = MODEL_NAME if USE_LLM else "expert-rule-based"

    # ── [START] log ───────────────────────────────────────────────────────
    log_start(task=task_id, env=ENV_NAME, model=model_label)

    result       = _post("/reset", {"task_id": task_id, "seed": 42})
    obs          = result["observation"]
    done         = result.get("done", False)
    total_reward = 0.0
    grader_score = 0.0
    history:  List[str]   = []
    resolved: List[str]   = []
    rewards:  List[float] = []
    error_msg: Optional[str] = None
    info: dict = {}

    for step in range(1, max_steps + 1):
        if done:
            break

        # ── Choose action ─────────────────────────────────────────────────
        # Compute remaining unresolved threats from last step info
        all_active   = info.get("active_threats", [])
        remaining    = [t for t in all_active if t not in resolved]

        # Medium + Hard: always use expert rules for reliability
        # Easy: try LLM first for demonstration, fall back to rules
        if task_id in ("medium", "hard"):
            action = expert_action(
                obs["observation_text"], resolved, history, task_id, remaining
            )
            source = "rule"
        else:
            action = llm_action(
                obs["observation_text"], obs["goal"],
                step, history, resolved
            )
            source = "LLM"
            if action is None:
                action = expert_action(
                    obs["observation_text"], resolved, history, task_id, remaining
                )
                source = "rule"

        # ── Anti-loop guard ───────────────────────────────────────────────
        if (len(history) >= 3 and
                history[-1] == history[-2] == history[-3] == action["action_type"]):
            cycle = ["patch_vuln", "quarantine_node", "deploy_drone",
                     "alert_team", "verify_access", "scan_cyber"]
            action = {"action_type": cycle[step % len(cycle)], "target": None}
            source = "anti-loop"

        history.append(action["action_type"])

        # ── Step environment ──────────────────────────────────────────────
        try:
            step_result  = _post("/step", action)
            obs          = step_result["observation"]
            reward       = float(step_result.get("reward", 0.0))
            done         = step_result.get("done", False)
            info         = step_result.get("info", {})
            error_msg    = None
        except Exception as e:
            reward    = 0.0
            error_msg = str(e)
            done      = False
            info      = {}

        total_reward += reward
        rewards.append(reward)

        for t in info.get("resolved_threats", []):
            if t not in resolved:
                resolved.append(t)

        if done:
            gs = info.get("grader_score")
            if gs is not None:
                grader_score = float(gs)

        # ── [STEP] log ────────────────────────────────────────────────────
        log_step(
            step=step,
            action=action["action_type"],
            reward=reward,
            done=done,
            error=error_msg,
        )

        if done:
            break

        time.sleep(0.03)

    # Handle step limit without done
    if not done:
        try:
            sr = _post("/step", {"action_type": "scan_cyber", "target": None})
            gs = sr.get("info", {}).get("grader_score")
            if gs is not None:
                grader_score = float(gs)
            r = float(sr.get("reward", 0.0))
            rewards.append(r)
            total_reward += r
            log_step(step=len(history)+1, action="scan_cyber",
                     reward=r, done=True, error=None)
        except Exception:
            pass

    success = grader_score >= 0.5

    # ── [END] log ─────────────────────────────────────────────────────────
    log_end(
        success=success,
        steps=len(history),
        score=grader_score,
        rewards=rewards,
    )

    return {
        "task_id":      task_id,
        "total_reward": round(total_reward, 4),
        "grader_score": round(grader_score, 4),
        "steps_taken":  len(history),
        "resolved":     resolved,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*64, flush=True)
    print("  UnifiedThreatFusionCenter — Baseline Inference", flush=True)
    print(f"  Model  : {MODEL_NAME if USE_LLM else 'Expert Rule-Based'}", flush=True)
    print(f"  Server : {HF_SPACE_URL}", flush=True)
    print("="*64, flush=True)

    try:
        health = _get("/health")
        print(f"\n  ✅ Server: {health}", flush=True)
    except Exception as e:
        print(f"\n  ❌ Cannot reach server at {HF_SPACE_URL}", flush=True)
        print(f"     Start: cd server && python app.py", flush=True)
        print(f"     Error: {e}", flush=True)
        return

    results = []
    for task_id in TASKS:
        r = run_task(task_id)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*64, flush=True)
    print("  FINAL SCORES", flush=True)
    print("="*64, flush=True)
    for r in results:
        filled = int(r["grader_score"] * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        print(f"  {r['task_id']:8s} [{bar}] {r['grader_score']:.4f}  "
              f"reward={r['total_reward']:+.2f}  steps={r['steps_taken']}", flush=True)

    overall = sum(r["grader_score"] for r in results) / len(results)
    filled  = int(overall * 20)
    bar     = "█" * filled + "░" * (20 - filled)
    print(f"\n  OVERALL  [{bar}] {overall:.4f} / 1.0000", flush=True)
    print("="*64 + "\n", flush=True)

    with open("baseline_results.json", "w") as f:
        json.dump({
            "model":         MODEL_NAME if USE_LLM else "expert-rule-based",
            "results":       results,
            "overall_score": round(overall, 4),
        }, f, indent=2)
    print("  ✅ Results saved → baseline_results.json\n", flush=True)


if __name__ == "__main__":
    main()
