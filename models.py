from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

class MetricSeries(BaseModel):
    """A series of values over steps/epochs."""
    name: str
    values: List[float]

class MLObservation(BaseModel):
    """The diagnostic view of the ML experiment."""
    ticket_id: str = Field(..., description="Unique ID for this debugging session")
    config: Dict[str, Any] = Field(..., description="Current hyperparameters and architecture flags")
    metrics: Dict[str, List[float]] = Field(..., description="Time-series metrics (loss, accuracy, etc.)")
    logs: List[str] = Field(..., description="Simulated console logs and error messages")
    step_count: int = Field(..., description="Current number of debugging steps taken")
    max_steps: int = Field(5, description="Maximum allowed debugging steps")
    is_done: bool = False

class MLAction(BaseModel):
    """An action taken by the ML engineer agent."""
    action_type: Literal["update_config", "run_training", "submit"]
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="New values for hyperparams")
    epochs_to_run: Optional[int] = Field(5, description="Number of epochs to simulate if run_training")
    explanation: Optional[str] = Field(None, description="Agent's reasoning for this action")

class MLReward(BaseModel):
    """Feedback on the agent's debugging progress."""
    score: float = Field(..., description="Current task score (0.01 - 0.99)")
    reason: str = Field(..., description="Explanation for the current score")
    is_final: bool = False

class MLState(BaseModel):
    """Internal hidden state for the environment."""
    task_id: str
    target_accuracy: float
    current_accuracy: float
    hidden_bug: str
    history: List[Dict[str, Any]]
