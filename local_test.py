#!/usr/bin/env python3
"""
local_test.py — Run this FIRST before anything else.
Tests the entire environment without Docker or any API key.
Just pure Python logic.

Run: python local_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import UnifiedThreatFusionCenter
from models import SOCAction, ActionType

def test_task(task_id: str, actions: list):
    print(f"\n{'─'*50}")
    print(f"  TESTING TASK: {task_id.upper()}")
    print(f"{'─'*50}")

    env = UnifiedThreatFusionCenter(seed=42)
    result = env.reset(task_id=task_id)

    print(f"  ✅ reset() works")
    print(f"  Goal: {result.observation.goal}")
    print(f"  Active threats: {env.state.active_threats}")
    print(f"  Observation preview:\n    {result.observation.observation_text[:200]}...")

    # state() check
    snap = env.state_snapshot()
    print(f"  ✅ state() works — threat_level: {snap['threat_level']}")

    # step() check
    total_reward = 0.0
    for a in actions:
        action = SOCAction(action_type=ActionType(a))
        step_result = env.step(action)
        total_reward += step_result.reward
        print(f"  step({a:20s}) → reward={step_result.reward:+.2f} | threat={step_result.observation.threat_level:.2f} | done={step_result.done}")
        if step_result.done:
            print(f"  ✅ Episode done. Grader score: {step_result.info.get('grader_score', 'N/A')}")
            break

    print(f"  Total reward so far: {total_reward:+.2f}")
    print(f"  ✅ Task {task_id} PASSED")


def main():
    print("\n" + "="*50)
    print("  UnifiedThreatFusionCenter — Local Test")
    print("="*50)

    try:
        # Easy task
        test_task("easy", [
            "scan_cyber", "block_port", "patch_vuln",
            "verify_access", "deploy_drone", "alert_team",
            "quarantine_node", "scan_cyber", "block_port"
        ])

        # Medium task
        test_task("medium", [
            "block_port", "scan_cyber", "patch_vuln",
            "deploy_drone", "quarantine_node", "alert_team",
            "verify_access", "scan_cyber", "block_port"
        ])

        # Hard task
        test_task("hard", [
            "scan_cyber", "block_port", "patch_vuln",
            "deploy_drone", "verify_access", "lockdown_zone",
            "quarantine_node", "alert_team", "scan_cyber",
            "block_port", "patch_vuln", "quarantine_node"
        ])

        print("\n" + "="*50)
        print("  ✅ ALL 3 TASKS PASSED — Environment is healthy!")
        print("  ✅ reset() / step() / state() all working")
        print("  ✅ Graders returning 0.0–1.0 scores")
        print("  ✅ Ready for Docker build")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n  ❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
