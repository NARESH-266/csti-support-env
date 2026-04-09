from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Department(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    SALES = "sales"
    LOGISTICS = "logistics"
    GENERAL = "general"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Action(BaseModel):
    """The action taken by the agent for triage."""
    department: Department = Field(..., description="The department to route the ticket to.")
    priority: Priority = Field(..., description="The priority level assigned to the ticket.")
    tags: List[str] = Field(default_factory=list, description="Relevant tags for the ticket (e.g., 'angry', 'refund', 'login').")
    explanation: Optional[str] = Field(None, description="A brief explanation of the reasoning.")

class Observation(BaseModel):
    """The state of the environment seen by the agent."""
    ticket_id: str = Field(..., description="Unique ID for the support ticket.")
    content: str = Field(..., description="The text content of the support ticket.")
    customer_segment: str = Field(..., description="The segment of the customer (e.g., 'standard', 'vip').")
    sla_deadline: str = Field(..., description="The SLA deadline for this ticket.")
    ticket_history: List[str] = Field(default_factory=list, description="Summary of past interactions with this customer.")
    step_count: int = Field(..., description="The current step in the task.")
    max_steps: int = Field(..., description="Maximum allowed steps.")

class Reward(BaseModel):
    """The reward signal returned by the environment."""
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized reward score between 0.0 and 1.0.")
    reason: str = Field(..., description="Explanation for the given score.")
    is_final: bool = Field(False, description="Whether this is the final step/reward.")
