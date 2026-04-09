from decision_env.env import create_env
from decision_env.models import Action

def verify_variation():
    print("Verifying Score Variation...")
    env = create_env(task_id="hard_policy_exception")
    
    # 1. Perfect Action
    perfect_action = {
        "department": "logistics", # Ground truth
        "priority": "urgent",      # Ground truth
        "tags": ["damaged-item", "policy-exception", "replacement", "churn-risk", "rain-damage"],
        "explanation": "Perfect match"
    }
    
    env.reset()
    _, score_perfect, _, _ = env.step(perfect_action)
    print(f"Perfect Action Score: {score_perfect}")
    
    # 2. Average Action (Correct Dept, Wrong Priority)
    average_action = {
        "department": "logistics",
        "priority": "low", # Far from 'urgent'
        "tags": ["damaged-item"], # Only 1 tag
        "explanation": "Average effort"
    }
    env.reset()
    _, score_average, _, _ = env.step(average_action)
    print(f"Average Action Score: {score_average}")
    
    # 3. Wrong Action
    wrong_action = {
        "department": "sales",
        "priority": "low",
        "tags": ["spam"],
        "explanation": "Wrong routing"
    }
    env.reset()
    _, score_wrong, _, _ = env.step(wrong_action)
    print(f"Wrong Action Score: {score_wrong}")
    
    print("\nSUMMARY:")
    print(f"Variation: Perfect({score_perfect}) vs Average({score_average}) vs Wrong({score_wrong})")
    
    if score_perfect > 0.9 and score_average < 0.7 and score_wrong < 0.2:
        print("VERIFICATION SUCCESSFUL: Meaningful differentiation achieved.")
    else:
        print("VERIFICATION WARNING: Score differentiation might be too narrow.")

if __name__ == "__main__":
    verify_variation()
