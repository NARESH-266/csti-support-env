from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from decision_env.env import create_env
from models import MLAction, MLObservation

app = FastAPI(title="ML Experiment Debugging Environment")

# Global session storage
envs = {}

@app.get("/")
def health():
    return {"status": "online", "env": "ml-debug-env"}

@app.post("/reset")
def reset(task_id: str = "easy_lr_issue"):
    try:
        env = create_env(task_id=task_id)
        envs[task_id] = env
        obs = env.reset()
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step(task_id: str, action: MLAction):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail="Environment not found. Call /reset first.")
    
    try:
        obs, reward, done, info = envs[task_id].step(action)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state(task_id: str):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail="Environment not found.")
    env = envs[task_id]
    return {
        "task_id": env.task_id,
        "step_count": env.step_count,
        "config": env.config,
        "metrics": env.metrics
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
