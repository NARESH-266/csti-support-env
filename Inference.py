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
    final_score = 0.0
    
    while not done:
        step += 1
        
        # System prompt for triage including the new realism fields
        prompt = f"""
        Analyze the following customer support ticket and classify it.
        Content: "{obs.content}"
        Segment: {obs.customer_segment}
        SLA: {obs.sla_deadline}
        History: {obs.ticket_history}
        
        Return JSON:
        {{
            "department": "technical" | "billing" | "sales" | "logistics" | "general",
            "priority": "low" | "medium" | "high" | "urgent",
            "tags": ["tag1", "tag2"],
            "explanation": "Brief reasoning"
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert support triage agent."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            action_data = json.loads(response.choices[0].message.content)
            action = Action(**action_data)
            
            # Step the environment
            obs, reward, done, info = env.step(action.model_dump())
            all_rewards.append(reward)
            
            # Use a safer, more descriptive action string
            action_tag_str = ",".join(action.tags) if action.tags else "none"
            action_log = f"{action.department}|{action.priority}|{action_tag_str}"
            
            done_str = "true" if done else "false"
            print(f"[STEP] step={step} action={action_log} reward={reward:.2f} done={done_str} error=null")
            
        except Exception as e:
            # Sanitize error: replace spaces and special chars, no newlines
            error_msg = str(e).replace(" ", "_").replace("\n", "").replace("[", "").replace("]", "")
            print(f"[STEP] step={step} action=none reward=0.00 done=true error={error_msg}")
            done = True
            all_rewards.append(0.0)

    # FINAL SCORE: Average of all rewards as recommended
    final_score = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    
    success_str = "true" if final_score > 0.1 else "false" # Using a safer 0.1 threshold
    rewards_str = ",".join([f"{r:.2f}" for r in all_rewards])
    print(f"[END] success={success_str} steps={step} score={final_score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    # Get task_id from environment variable, defaulting to easy_login_issue
    # This ensures exactly one START/END block per execution
    task_id = os.getenv("MY_ENV_V4_TASK", "easy_login_issue")
    solve_task(task_id)
