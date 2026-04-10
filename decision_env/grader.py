from typing import Dict, Any
from models import MLAction, MLReward

def evaluate_metrics(metrics: Dict[str, Any], target: float) -> float:
    """Helper to clamp metrics based scores to (0.01, 0.99)."""
    val_acc_list = metrics.get("val_accuracy", [0.0])
    current_acc = val_acc_list[-1] if val_acc_list else 0.0
    
    if current_acc >= target:
        return 0.99
    
    # Linear scale but strictly within (0.01, 0.99)
    score = (current_acc / target) * 0.90 + 0.05
    return max(0.01, min(0.99, score))

class MLDebugGrader:
    def grade(self, action: Any, ground_truth: Dict[str, Any]) -> Any:
        # Note: In OpenEnv, 'action' can sometimes be the last state 
        # depending on which validator is running.
        
        target = ground_truth.get("target_accuracy", 0.8)
        
        # If we have metrics in the 'action' (if it's a state object)
        if hasattr(action, "metrics"):
            score = evaluate_metrics(action.metrics, target)
        elif isinstance(action, dict) and "metrics" in action:
            score = evaluate_metrics(action["metrics"], target)
        else:
            # Fallback for baseline checks
            score = 0.01
            
        return MLReward(
            score=max(0.01, min(0.99, score)),
            reason=f"Evaluation completed for task {ground_truth.get('id')}",
            is_final=True
        )
