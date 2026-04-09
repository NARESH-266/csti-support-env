import os
import sys
import json
from openai import OpenAI
from decision_env.env import create_env
from decision_env.models import Action

# Configuration from environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY"))

def solve_task(task_id: str):
    env = create_env(task_id=task_id)
    obs = env.reset()
    
    print(f"[START] task={task_id} env=csti-support-intelligence model={MODEL_NAME}")
    
    step = 0
    done = False
    all_rewards = []
    
    while not done:
        step += 1
        
        # System prompt for advanced diagnostic triage
        prompt = f"""
        You are an expert support triage agent with diagnostic tools.
        Current Ticket:
        - ID: {obs.ticket_id}
        - Content: "{obs.content}"
        - Segment: {obs.customer_segment}
        - SLA: {obs.sla_deadline}
        - History: {obs.ticket_history}
        
        Diagnostic Context:
        {json.dumps(obs.messages, indent=2) if obs.messages else "No diagnostic tools used yet."}
        
        Available Tools:
        {json.dumps(obs.available_tools, indent=2)}
        
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
            action = Action(**action_data)
            
            # Step the environment
            obs, reward, done, info = env.step(action.model_dump())
            all_rewards.append(reward)
            
            # Log the action (Tool Call or Triage)
            if action.tool_call:
                action_log = f"tool:{action.tool_call.name}"
            else:
                action_tag_str = ",".join(action.tags) if action.tags else "none"
                action_log = f"triage:{action.department}|{action.priority}|{action_tag_str}"
            
            done_str = "true" if done else "false"
            print(f"[STEP] step={step} action={action_log} reward={reward:.2f} done={done_str} error=null")
            
        except Exception as e:
            error_msg = str(e).replace(" ", "_").replace("\n", "").replace("[", "").replace("]", "")
            print(f"[STEP] step={step} action=none reward=0.00 done=true error={error_msg}")
            done = True
            all_rewards.append(0.0)

    # FINAL SCORE calculation
    final_score = all_rewards[-1] if all_rewards else 0.0 # Reward is only given on final triage step
    
    success_str = "true" if final_score > 0.1 else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in all_rewards])
    print(f"[END] success={success_str} steps={step} score={final_score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    task_id = os.getenv("MY_ENV_V4_TASK", "hard_hidden_failure_mode") # Default to the new complex task
    solve_task(task_id)
