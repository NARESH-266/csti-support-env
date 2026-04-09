from decision_env.models import Reward, Action
from typing import Dict, Any


def evaluate(action: Action, task: Dict[str, Any]) -> Reward:
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
    # Support both 'tags' and 'required_tags' keys
    expected_tags = set(expected.get("tags", expected.get("required_tags", [])))
    predicted_tags = set(action.tags)

    if expected_tags:
        matches = len(expected_tags & predicted_tags)
        tag_score = (matches / len(expected_tags)) * 0.2
        score += tag_score
        reasons.append(f"Tag match {matches}/{len(expected_tags)}")
    else:
        reasons.append("No tags required")

    # --- FINAL SAFETY CLAMP (CRITICAL) ---
    # Prevent EXACT 0.0 or 1.0
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    # Extra safety buffer (handles floating precision edge cases)
    score = max(0.01, min(0.99, score))

    return Reward(
        score=score,  # NO ROUNDING - raw float to avoid precision issues
        reason=" | ".join(reasons)
    )


class TriageGrader:
    """Wrapper class for openenv.yaml grader reference compatibility."""
    def grade(self, action: Action, ground_truth: Dict[str, Any]) -> Reward:
        return evaluate(action, {"ground_truth": ground_truth})
