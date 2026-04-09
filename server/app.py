from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from .env import create_env
from .models import Action

app = FastAPI(title="OpenEnv CSTI Server")

# Global state to keep track of active environments by task
envs = {}

class StepRequest(BaseModel):
    task_id: str
    action: Dict[str, Any]

@app.get("/")
def read_root():
    return {"status": "online", "env": "csti-support-intelligence", "version": "1.0.0"}

@app.api_route("/reset", methods=["GET", "POST"])
def reset(task_id: str = "easy_login_issue"):
    try:
        env = create_env(task_id=task_id)
        envs[task_id] = env
        obs = env.reset()
        # Ensure the response is exactly what the validator expects
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.post("/step")
def step(request: StepRequest):
    if request.task_id not in envs:
        # Auto-reset if not exists for convenience
        envs[request.task_id] = create_env(task_id=request.task_id)
        envs[request.task_id].reset()
    
    try:
        obs, reward, done, info = envs[request.task_id].step(request.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state(task_id: str = "easy_login_issue"):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail="Task environment not initialized.")
    return envs[task_id].state()
