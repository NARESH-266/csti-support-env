import json
from typing import Tuple, Dict, Any, Optional, List
from .models import Action, Observation, Reward
from .tasks import get_task
from .grader import TriageGrader
from .tools import ToolRegistry

class CSTIEnv:
    """Advanced Customer Support Ticket Intelligence Environment (Multi-turn Tool-use)."""
    
    def __init__(self, task_id: str = "easy_login_issue", max_steps: int = 5):
        self.task_id = task_id
        self.task_data = get_task(task_id)
        self.grader = TriageGrader()
        self.tool_registry = ToolRegistry()
        self.init_max_steps = max_steps
        self.reset()

    def reset(self) -> Observation:
        """Resets the environment to the initial state."""
        self.current_step = 0
        self.max_steps = self.init_max_steps
        self.done = False
        self.messages = []
        self.last_reward = None
        
        return self._get_observation()

    def step(self, action: Any) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Takes an action (Triage or ToolCall) and returns (observation, reward, done, info).
        """
        if self.done:
            raise RuntimeError("Environment is already done. Please call reset().")

        # Parse action logic
        if isinstance(action, str):
            try:
                action_data = json.loads(action)
                action = Action(**action_data)
            except:
                pass
        elif isinstance(action, dict):
            action = Action(**action)

        self.current_step += 1
        
        # 1. Handle Tool Calls (Intermediate Steps)
        if action.tool_call:
            tool_name = action.tool_call.name
            tool_args = action.tool_call.arguments
            
            # Execute tool
            result = self.tool_registry.call_tool(tool_name, tool_args, self.task_data)
            
            # Update message history
            self.messages.append({"role": "assistant", "content": f"Call tool: {tool_name}"})
            self.messages.append({"role": "tool", "content": result})
            
            # Check if exceeded max steps
            if self.current_step >= self.max_steps:
                self.done = True
                # No reward for timeout
                return self._get_observation(), 0.0, True, {"reason": "Max steps exceeded", "task_id": self.task_id}
            
            return self._get_observation(), 0.0, False, {"reason": "Tool called", "task_id": self.task_id}

        # 2. Handle Triage Action (Terminal Step)
        if action.department:
            reward_model = self.grader.grade(action, self.task_data["ground_truth"])
            self.last_reward = reward_model
            self.done = True
            
            # Small penalty for efficiency (optional, but good for Round 2)
            # score = reward_model.score * (0.95 ** (self.current_step - 1))
            score = reward_model.score
            
            return self._get_observation(), score, True, {
                "reason": reward_model.reason,
                "task_id": self.task_id
            }

        # 3. Handle Invalid Action
        self.done = True
        return self._get_observation(), 0.0, True, {"reason": "Invalid action format", "task_id": self.task_id}

    def state(self) -> Dict[str, Any]:
        """Returns the internal state of the environment."""
        return {
            "task_id": self.task_id,
            "current_step": self.current_step,
            "done": self.done,
            "ground_truth": self.task_data["ground_truth"],
            "messages": self.messages
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
            max_steps=self.max_steps,
            messages=self.messages,
            available_tools=self.tool_registry.get_available_tools()
        )

def create_env(task_id: str = "easy_login_issue", **kwargs):
    return CSTIEnv(task_id=task_id, **kwargs)
