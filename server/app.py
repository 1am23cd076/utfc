"""
UnifiedThreatFusionCenter — server/app.py
FastAPI server exposing OpenEnv-compliant endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict

from models import SOCAction, ActionType
from environment import UnifiedThreatFusionCenter

app = FastAPI(
    title="UnifiedThreatFusionCenter",
    description="OpenEnv-compliant SOC RL Environment — cyber-physical threat fusion",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (one per server)
env = UnifiedThreatFusionCenter(seed=42)


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action_type: str
    target: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {"status": "ok", "env": "UnifiedThreatFusionCenter"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    try:
        result = env.reset(task_id=request.task_id, seed=request.seed)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    try:
        action = SOCAction(
            action_type=ActionType(request.action_type),
            target=request.target,
            parameters=request.parameters,
        )
        result = env.step(action)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    try:
        return env.state_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def list_tasks():
    from models import TASK_DEFINITIONS
    return TASK_DEFINITIONS


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


def main():
    """Entry point for openenv server script."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
