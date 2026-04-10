from typing import Dict, Any

TASKS: Dict[str, Dict[str, Any]] = {
    "easy_lr_issue": {
        "id": "easy_lr_issue",
        "difficulty": "easy",
        "hidden_bug": "low_lr",
        "initial_config": {
            "learning_rate": 1e-7,
            "optimizer": "adam",
            "num_layers": 5,
            "use_batch_norm": True
        },
        "target_accuracy": 0.85,
        "description": "Experiment is not converging because the learning rate is extremely low."
    },
    "medium_vanishing_gradient": {
        "id": "medium_vanishing_gradient",
        "difficulty": "medium",
        "hidden_bug": "vanishing_gradient",
        "initial_config": {
            "learning_rate": 1e-3,
            "optimizer": "sgd",
            "num_layers": 30, # Deep
            "use_batch_norm": False
        },
        "target_accuracy": 0.80,
        "description": "Very deep network with no normalization layers. Gradients are vanishing early."
    },
    "hard_data_leakage": {
        "id": "hard_data_leakage",
        "difficulty": "hard",
        "hidden_bug": "data_leakage",
        "initial_config": {
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "num_layers": 5,
            "use_batch_norm": True,
            "data_split_seed": 42
        },
        "target_accuracy": 0.90, # This is the "hidden" test accuracy
        "description": "Suspiciously high validation accuracy. Might be data leakage or split issues."
    }
}

def get_task(task_id: str) -> Dict[str, Any]:
    return TASKS.get(task_id, TASKS["easy_lr_issue"])
