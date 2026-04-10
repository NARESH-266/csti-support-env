from typing import Dict, Any
from models import MLAction, MLReward

def evaluate(action: Any, task_data: Dict[str, Any]) -> MLReward:
    """
    Grades the ML Engineer's action.
    For MLDebugEnv, we care about the final accuracy achieved.
    """
    # This grader is called by the platform. 
    # Usually it's called with the final state/action.
    
    # In our simulation, the 'reward' is already calculated by the env.
    # But if the platform needs a standalone grader:
    
    target_acc = task_data.get("target_accuracy", 0.85)
    current_acc = task_data.get("current_metrics", {}).get("val_accuracy", [0.0])[-1]
    
    score = (current_acc / target_acc) * 0.99
    
    if current_acc >= target_acc:
        score = 0.99
    
    # Strict clamping
    score = max(0.01, min(0.99, score))
    
    return MLReward(
        score=score,
        reason=f"Reached {current_acc:.4f} accuracy (Target: {target_acc})",
        is_final=True
    )

class MLDebugGrader:
    def grade(self, action: Any, ground_truth: Dict[str, Any]) -> MLReward:
        return evaluate(action, ground_truth)
