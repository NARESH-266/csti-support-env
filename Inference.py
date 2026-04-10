import os
import sys
import json
import httpx
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Configuration from environment variables
# ──────────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY"))

# Tasks defined in openenv.yaml
TASKS = ["easy_login_issue", "medium_billing_angry", "hard_policy_exception"]


def solve_task(task_id: str, task_index: int):
    """Runs the agent against the specified task via the OpenEnv server."""
    
    # 1. START Log
    print(f"[START] {json.dumps({'task_id': task_id, 'task_index': task_index})}")
    sys.stdout.flush()

    # 2. Reset the environment
    try:
        # FastAPI will take task_id from query param
        resp = httpx.post(f"{ENV_BASE_URL}/reset?task_id={task_id}", timeout=30.0)
        resp.raise_for_status()
        obs_data = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to reset {task_id}: {e}", file=sys.stderr)
        print(f"[END] {json.dumps({'task_id': task_id, 'score': 0.01})}")
        return

    done = False
    step = 0
    max_steps = 10
    final_score = 0.01

    while not done and step < max_steps:
        step += 1
        
        # System prompt for advanced diagnostic triage (Level 2)
        messages = obs_data.get("messages", [])
        tools = obs_data.get("available_tools", [])
        
        prompt = f"""
        You are an expert support triage agent with diagnostic tools.
        Current Ticket:
        - ID: {obs_data.get('ticket_id')}
        - Content: "{obs_data.get('content')}"
        - Segment: {obs_data.get('customer_segment')}
        - SLA: {obs_data.get('sla_deadline')}
        - History: {obs_data.get('ticket_history')}
        
        Diagnostic Context:
        {json.dumps(messages, indent=2) if messages else "No diagnostic tools used yet."}
        
        Available Tools:
        {json.dumps(tools, indent=2)}
        
        Your Goal:
        Identify the correct department, priority, and tags. 
        If info is missing (e.g., ambiguous error), use a tool first.
        
        Return JSON Choice A (Diagnostic):
        {{
            "tool_call": {{ "name": "tool_name", "arguments": {{ "arg1": "val" }} }}
        }}
        
        Return JSON Choice B (Final Triage):
        {{
            "department": "technical" | "billing" | "sales" | "logistics" | "general",
            "priority": "low" | "medium" | "high" | "urgent",
            "tags": ["tag1", "tag2"],
            "explanation": "Reasoning based on ticket and diagnostic results."
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a senior support engineer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            action_data = json.loads(response.choices[0].message.content)
            
            # 3. STEP the environment
            step_resp = httpx.post(
                f"{ENV_BASE_URL}/step",
                json={"task_id": task_id, "action": action_data},
                timeout=45.0
            )
            step_resp.raise_for_status()
            step_result = step_resp.json()
            
            obs_data = step_result.get("observation")
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            
            # STEP Log
            print(f"[STEP] {json.dumps({'task_id': task_id, 'step': step, 'action': action_data, 'reward': reward, 'done': done})}")
            sys.stdout.flush()
            
            if done:
                final_score = reward
                
        except Exception as e:
            print(f"[ERROR] Step {step} failed: {e}", file=sys.stderr)
            done = True

    # 4. END Log
    print(f"[END] {json.dumps({'task_id': task_id, 'score': final_score})}")
    sys.stdout.flush()


def main():
    # Verify server health first
    try:
        httpx.get(f"{ENV_BASE_URL}/", timeout=5.0)
    except Exception:
        print(f"[ERROR] Server not reachable at {ENV_BASE_URL}. Ensure it is running.", file=sys.stderr)
    
    for i, task_id in enumerate(TASKS):
        solve_task(task_id, i)


if __name__ == "__main__":
    main()
