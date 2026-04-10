from decision_env.models import Reward, Action
from typing import Dict, Any


def evaluate(action: Action, task: Dict[str, Any]) -> Reward:
    """Grades the final triage action of the agent."""
    score = 0.0
    reasons = []

    # Support both 'expected' and 'ground_truth' keys for compatibility
    expected = task.get("expected", task.get("ground_truth", {}))

    # --- Department (50%) ---
    if action.department == expected.get("department"):
        score += 0.5
        reasons.append("Correct department")
    else:
        score += 0.05
        reasons.append("Wrong department")

    # --- Priority (30%) ---
    expected_priority = expected.get("priority", "medium")
    priority_map = {"low": 1, "medium": 2, "high": 3, "urgent": 4}

    if action.priority in priority_map and expected_priority in priority_map:
        diff = abs(priority_map[action.priority] - priority_map[expected_priority])
        if diff == 0:
            score += 0.3
            reasons.append("Correct priority")
        elif diff == 1:
            score += 0.15
            reasons.append("Close priority")
        else:
            score += 0.05
            reasons.append("Wrong priority")
    else:
        score += 0.05
        reasons.append("Invalid priority")

    # --- Tags (20%) ---
    expected_tags = set(expected.get("tags", expected.get("required_tags", [])))
    if action.tags:
        predicted_tags = set(action.tags)
    else:
        predicted_tags = set()

    if expected_tags:
        matches = len(expected_tags & predicted_tags)
        tag_score = (matches / len(expected_tags)) * 0.2
        score += tag_score
        reasons.append(f"Tag match {matches}/{len(expected_tags)}")
    else:
        reasons.append("No tags required")

    # --- FINAL SAFETY CLAMP (CRITICAL) ---
    # Prevent EXACT 0.0 or 1.0 (Round 2 requirement)
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    # Extra safety buffer
    score = max(0.01, min(0.99, score))

    return Reward(
        score=score,
        reason=" | ".join(reasons),
        is_final=True
    )


def evaluate_score(action: Any, ground_truth: Dict[str, Any]) -> float:
    """Wrapper that returns a float score directly, strictly clamped between 0.01 and 0.99."""
    if isinstance(action, dict):
        # Convert dict to Action model for evaluate function
        from .models import Action
        action = Action(**action)
    
    reward = evaluate(action, {"ground_truth": ground_truth})
    
    # Final strict clamping for the platform requirement (strictly >0 and <1)
    score = float(reward.score)
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    
    return max(0.01, min(0.99, score))


class TriageGrader:
    """Wrapper class for openenv.yaml grader reference compatibility."""
    def grade(self, action: Action, ground_truth: Dict[str, Any]) -> Reward:
        score = evaluate_score(action, ground_truth)
        # Re-wrap in Reward for existing env.py compatibility
        return Reward(
            score=score,
            reason="Triage evaluated",
            is_final=True
        )
