import os
import sys
import json
import httpx
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY"))

TASKS = ["easy_lr_issue", "medium_vanishing_gradient", "hard_data_leakage"]

def solve_task(task_id: str, task_index: int):
    print(f"[START] {json.dumps({'task_id': task_id, 'task_index': task_index})}")
    
    try:
        # 1. Reset
        resp = httpx.post(f"{ENV_BASE_URL}/reset?task_id={task_id}", timeout=10.0)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        print(f"[ERROR] Reset failed: {e}", file=sys.stderr)
        print(f"[END] {json.dumps({'task_id': task_id, 'score': 0.01})}")
        return

    done = False
    step = 0
    final_score = 0.01

    while not done and step < 5:
        step += 1
        
        prompt = f"""
        You are a Senior ML Engineer debugging a failed training run.
        Task: {task_id}
        
        Current Config:
        {json.dumps(obs.get('config'), indent=2)}
        
        Latest Metrics (last 5 values):
        {json.dumps({k: v[-5:] for k, v in obs.get('metrics', {}).items()}, indent=2)}
        
        Recent Logs:
        {json.dumps(obs.get('logs'), indent=2)}
        
        Goal: Achieve high validation accuracy.
        Choose ONE Action:
        
        Option A: Update Configuration
        {{ "action_type": "update_config", "config_overrides": {{ "key": "value" }}, "explanation": "..." }}
        
        Option B: Run Training (Advance simulation)
        {{ "action_type": "run_training", "epochs_to_run": 5, "explanation": "..." }}
        
        Option C: Submit (Done)
        {{ "action_type": "submit", "explanation": "..." }}
        
        Return ONLY valid JSON.
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            action_data = json.loads(response.choices[0].message.content)
            
            # 2. Step
            step_resp = httpx.post(
                f"{ENV_BASE_URL}/step?task_id={task_id}",
                json=action_data,
                timeout=10.0
            )
            step_resp.raise_for_status()
            result = step_resp.json()
            
            obs = result.get("observation")
            reward = result.get("reward", 0.01)
            done = result.get("done", False)
            
            print(f"[STEP] {json.dumps({'task_id': task_id, 'step': step, 'action': action_data, 'reward': reward, 'done': done})}")
            
            if done:
                final_score = reward
                
        except Exception as e:
            print(f"[ERROR] Step failed: {e}", file=sys.stderr)
            break

    print(f"[END] {json.dumps({'task_id': task_id, 'score': final_score})}")

def main():
    for i, tid in enumerate(TASKS):
        solve_task(tid, i)

if __name__ == "__main__":
    main()
