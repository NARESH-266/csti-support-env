from typing import Dict, Any

TASKS: Dict[str, Dict[str, Any]] = {
    "easy_lr_issue": {
        "id": "easy_lr_issue",
        "difficulty": "easy",
        "hidden_bug": "low_lr",
        "initial_config": {"learning_rate": 1e-7, "use_batch_norm": True},
        "expected_config": {"learning_rate": 0.01}, # Grader looks for this
        "target_accuracy": 0.85
    },
    "medium_vanishing_gradient": {
        "id": "medium_vanishing_gradient",
        "difficulty": "medium",
        "hidden_bug": "vanishing_gradient",
        "initial_config": {"learning_rate": 1e-3, "num_layers": 30, "use_batch_norm": False},
        "expected_config": {"use_batch_norm": True},
        "target_accuracy": 0.80
    },
    "hard_data_leakage": {
        "id": "hard_data_leakage",
        "difficulty": "hard",
        "hidden_bug": "data_leakage",
        "initial_config": {"learning_rate": 1e-3, "data_split_seed": 42},
        "expected_config": {"data_split_seed": 100}, # Change seed to fix leakage
        "target_accuracy": 0.90
    },
    "bonus_complexity_test": { # 4th task to exceed "at least 3" requirement
        "id": "bonus_complexity_test",
        "difficulty": "hard",
        "hidden_bug": "low_lr",
        "initial_config": {"learning_rate": 1e-8},
        "expected_config": {"learning_rate": 0.005},
        "target_accuracy": 0.70
    }
}

def get_task(task_id: str) -> Dict[str, Any]:
    return TASKS.get(task_id, TASKS["easy_lr_issue"])
