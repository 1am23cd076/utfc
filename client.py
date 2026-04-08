"""
UnifiedThreatFusionCenter — client.py
OpenEnv-compliant client for the SOC environment.
"""

from __future__ import annotations
from openenv.core import HTTPEnvClient
from models import SOCAction, SOCObservation


class UnifiedThreatFusionCenterEnv(HTTPEnvClient[SOCAction, SOCObservation]):
    """
    OpenEnv HTTP client for the UnifiedThreatFusionCenter environment.

    Usage:
        async with UnifiedThreatFusionCenterEnv(base_url="http://localhost:7860") as env:
            result = await env.reset()
            result = await env.step(SOCAction(action_type="scan_cyber"))

    Or synchronously:
        with UnifiedThreatFusionCenterEnv(base_url="...").sync() as env:
            result = env.reset()
            result = env.step(SOCAction(action_type="block_port"))
    """

    action_class = SOCAction
    observation_class = SOCObservation
