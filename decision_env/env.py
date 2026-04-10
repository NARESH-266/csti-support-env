from typing import Tuple, Dict, Any
from models import MLObservation, MLAction, MLReward
from .simulation import TrainingSimulator
from .tasks import get_task

class MLEnv:
    """The ML Experiment Debugging Environment."""
    
    def __init__(self, task_id: str = "easy_lr_issue"):
        self.task_id = task_id
        self.task_data = get_task(task_id)
        self.simulator = TrainingSimulator(hidden_bug=self.task_data["hidden_bug"])
        self.config = self.task_data["initial_config"].copy()
        self.step_count = 0
        self.max_steps = 10
        self.done = False
        self.logs = []
        self.metrics = {"train_loss": [], "val_accuracy": []}

    def reset(self) -> MLObservation:
        self.step_count = 0
        self.done = False
        # Initial run to generate some metrics
        self.metrics, new_logs = self.simulator.simulate(self.config, 5)
        self.logs = new_logs
        return self._get_observation()

    def step(self, action: MLAction) -> Tuple[MLObservation, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Env is done")

        self.step_count += 1
        info = {}

        if action.action_type == "update_config":
            if action.config_overrides:
                self.config.update(action.config_overrides)
            info["message"] = "Config updated"
            
        elif action.action_type == "run_training":
            epochs = action.epochs_to_run or 5
            self.metrics, new_logs = self.simulator.simulate(self.config, epochs)
            self.logs.extend(new_logs)
            info["message"] = f"Ran {epochs} epochs"
            
        elif action.action_type == "submit":
            self.done = True
            info["message"] = "Agent submitted solution"

        # Calculate reward
        reward_model = self._calculate_reward()
        
        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_observation(), reward_model.score, self.done, info

    def _get_observation(self) -> MLObservation:
        return MLObservation(
            ticket_id=f"DEBUG-{self.task_id}",
            config=self.config,
            metrics=self.metrics,
            logs=self.logs[-10:], # Last 10 logs
            step_count=self.step_count,
            is_done=self.done
        )

    def _calculate_reward(self) -> MLReward:
        # Success is determined by the final validation accuracy reaching the target
        current_acc = self.metrics["val_accuracy"][-1] if self.metrics["val_accuracy"] else 0.0
        target_acc = self.task_data["target_accuracy"]
        
        # Grading remains strict 0.01 - 0.99
        score = (current_acc / target_acc) * 0.99
        if self.done and current_acc >= target_acc:
            score = 0.99
        
        return MLReward(
            score=max(0.01, min(0.99, score)),
            reason=f"Current accuracy: {current_acc:.4f} / Target: {target_acc}",
            is_final=self.done
        )

def create_env(task_id: str = "easy_lr_issue", **kwargs):
    return MLEnv(task_id=task_id)
