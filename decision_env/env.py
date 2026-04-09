import json
from typing import Tuple, Dict, Any, Optional
from .models import Action, Observation, Reward
from .tasks import get_task
from .grader import TriageGrader

class CSTIEnv:
    """Customer Support Ticket Intelligence Environment."""
    
    def __init__(self, task_id: str = "easy_login_issue"):
        self.task_id = task_id
        self.task_data = get_task(task_id)
        self.grader = TriageGrader()
        self.reset()

    def reset(self) -> Observation:
        """Resets the environment to the initial state."""
        self.current_step = 0
        self.max_steps = 1
        self.done = False
        self.last_reward = None
        
        return self._get_observation()

    def step(self, action: Any) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Takes an action and returns (observation, reward, done, info).
        Supports both dict and Action model as input.
        """
        if self.done:
            raise RuntimeError("Environment is already done. Please call reset().")

        # Parse action if it's a dict or string
        if isinstance(action, str):
            try:
                action_data = json.loads(action)
                action = Action(**action_data)
            except:
                # Fallback or error handling
                pass
        elif isinstance(action, dict):
            action = Action(**action)

        self.current_step += 1
        
        # Grade the action
        reward_model = self.grader.grade(action, self.task_data["ground_truth"])
        self.last_reward = reward_model
        
        # In this environment, it's a one-shot classification task per ticket
        self.done = True
        
        obs = self._get_observation()
        
        return obs, reward_model.score, self.done, {
            "reason": reward_model.reason,
            "task_id": self.task_id
        }

    def state(self) -> Dict[str, Any]:
        """Returns the internal state of the environment."""
        return {
            "task_id": self.task_id,
            "current_step": self.current_step,
            "done": self.done,
            "ground_truth": self.task_data["ground_truth"]
        }

    def _get_observation(self) -> Observation:
        """Constructs the observation model."""
        input_data = self.task_data["input"]
        return Observation(
            ticket_id=input_data["ticket_id"],
            content=input_data["content"],
            customer_segment=input_data["customer_segment"],
            sla_deadline=input_data["sla_deadline"],
            ticket_history=input_data["ticket_history"],
            step_count=self.current_step,
            max_steps=self.max_steps
        )

# Factory function for OpenEnv registration/loading
def create_env(task_id: str = "easy_login_issue", **kwargs):
    return CSTIEnv(task_id=task_id)
