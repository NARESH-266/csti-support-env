from decision_env.env import create_env
from decision_env.models import Action

def verify():
    print("Verifying CSTI Environment...")
    env = create_env(task_id="easy_login_issue")
    obs = env.reset()
    print(f"Observation: {obs.content[:50]}...")
    
    # Test an action
    test_action = {
        "department": "technical",
        "priority": "medium",
        "tags": ["login", "password-reset"],
        "explanation": "Test verify"
    }
    
    obs, reward, done, info = env.step(test_action)
    print(f"Score: {reward}")
    print(f"Reason: {info['reason']}")
    print(f"Done: {done}")
    
    if reward > 0.0:
        print("VERIFICATION SUCCESSFUL")
    else:
        print("VERIFICATION FAILED: Zero reward")

if __name__ == "__main__":
    verify()
