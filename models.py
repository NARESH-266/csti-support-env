from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

class MLObservation(BaseModel):
    ticket_id: str = Field(..., description="Unique ID for this debugging session")
    config: Dict[str, Any] = Field(..., description="Current hyperparameters and architecture flags")
    metrics: Dict[str, List[float]] = Field(..., description="Time-series metrics")
    logs: List[str] = Field(..., description="Console logs")
    step_count: int = Field(..., description="Steps taken")
    max_steps: int = Field(10, description="Max steps")
    is_done: bool = False

class MLAction(BaseModel):
    action_type: Literal["update_config", "run_training", "submit"]
    config_overrides: Optional[Dict[str, Any]] = None
    epochs_to_run: Optional[int] = 5
    explanation: Optional[str] = None

class MLReward(BaseModel):
    # CRITICAL: Strict (0, 1) range enforcement at the model level
    score: float = Field(..., gt=0.0, lt=1.0, description="Score strictly between 0.01 and 0.99")
    reason: str = Field(..., description="Why this score was given")
    is_final: bool = False
