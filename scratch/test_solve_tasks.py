import httpx
import json
import sys

ENV_BASE_URL = "http://localhost:7860"

def test_task(task_id, actions):
    print(f"\n--- Testing Task: {task_id} ---")
    
    # 1. Reset
    resp = httpx.post(f"{ENV_BASE_URL}/reset?task_id={task_id}")
    obs = resp.json()
    print(f"Initial Accuracy: {obs['metrics']['val_accuracy'][-1]:.4f}")

    # 2. Run Actions
    for action in actions:
        print(f"Action: {action['action_type']} | {action.get('config_overrides', '')}")
        step_resp = httpx.post(
            f"{ENV_BASE_URL}/step?task_id={task_id}",
            json=action
        )
        result = step_resp.json()
        reward = result['reward']
        done = result['done']
        obs = result['observation']
        print(f"  Current Accuracy: {obs['metrics']['val_accuracy'][-1]:.4f} | Reward: {reward:.4f}")

    if obs['metrics']['val_accuracy'][-1] > 0.5:
        print(f"[PASSED] Task {task_id}")
    else:
        print(f"[FAILED] Task {task_id}")

def main():
    # Easy Task: Fix LR
    test_task("easy_lr_issue", [
        {"action_type": "update_config", "config_overrides": {"learning_rate": 0.01}},
        {"action_type": "run_training", "epochs_to_run": 10},
        {"action_type": "submit"}
    ])

    # Medium Task: Fix BN
    test_task("medium_vanishing_gradient", [
        {"action_type": "update_config", "config_overrides": {"use_batch_norm": True}},
        {"action_type": "run_training", "epochs_to_run": 10},
        {"action_type": "submit"}
    ])

if __name__ == "__main__":
    main()
