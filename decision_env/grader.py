from .models import Action, Reward
from typing import Dict, Any

class TriageGrader:
    def grade(self, action: Action, ground_truth: Dict[str, Any]) -> Reward:
        """
        Grades the agent's action against the ground truth.
        Returns a Reward model with a score between 0.01 and 0.99.
        (Clamping for strict validation if needed, but the user asked for 0.0 to 1.0).
        Actually, let's stick to 0.0-1.0 as requested, but ensure differentiation.
        """
        score = 0.0
        reasons = []

        # 1. Department Routing (50%)
        # Reward smoothing: Small partial credit (0.1) for just taking an action
        # and 0.5 for a correct match.
        if action.department == ground_truth["department"]:
            score += 0.5
            reasons.append("Correct department routing (+0.5).")
        else:
            score += 0.05 # Tiny participation reward for routing effort
            reasons.append(f"Incorrect department. Expected {ground_truth['department']}, got {action.department}. (+0.05 partial)")

        # 2. Priority Assignment (30%)
        priorities = ["low", "medium", "high", "urgent"]
        try:
            target_idx = priorities.index(ground_truth["priority"])
            actual_idx = priorities.index(action.priority)
            distance = abs(target_idx - actual_idx)
            
            if distance == 0:
                score += 0.3
                reasons.append("Correct priority assignment (+0.3).")
            elif distance == 1:
                score += 0.15
                reasons.append("Priority is near-match (+0.15).")
            elif distance == 2:
                score += 0.05
                reasons.append("Priority is somewhat close (+0.05).")
            else:
                reasons.append("Priority assignment is significantly incorrect.")
        except ValueError:
            reasons.append("Invalid priority value.")

        # 3. Tags (20%)
        required_tags = ground_truth.get("required_tags", [])
        if required_tags:
            matches = set(action.tags).intersection(set(required_tags))
            tag_score = (len(matches) / len(required_tags)) * 0.2
            score += tag_score
            if len(matches) > 0:
                reasons.append(f"Matched {len(matches)}/{len(required_tags)} required tags (+{tag_score:.2f}).")
            else:
                reasons.append("No relevant tags matched.")

        # Strict clamping between 0.0 and 1.0
        score = max(0.0, min(1.0, score))

        return Reward(
            score=round(score, 2),
            reason=" | ".join(reasons),
            is_final=True
        )
