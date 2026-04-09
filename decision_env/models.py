from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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

class ToolCall(BaseModel):
    """A call to a diagnostic tool."""
    name: str = Field(..., description="The name of the tool to call.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="The arguments for the tool.")

class Action(BaseModel):
    """The action taken by the agent."""
    # Choice 1: Triage (Ends episode)
    department: Optional[Department] = Field(None, description="The department to route the ticket to.")
    priority: Optional[Priority] = Field(None, description="The priority level assigned to the ticket.")
    tags: Optional[List[str]] = Field(None, description="Relevant tags for the ticket.")
    explanation: Optional[str] = Field(None, description="A brief explanation of the reasoning.")
    
    # Choice 2: Diagnostic (Continues episode)
    tool_call: Optional[ToolCall] = Field(None, description="A call to a diagnostic tool.")

class Observation(BaseModel):
    """The state of the environment seen by the agent."""
    ticket_id: str = Field(..., description="Unique ID for the support ticket.")
    content: str = Field(..., description="The text content of the support ticket.")
    customer_segment: str = Field(..., description="The segment of the customer (e.g., 'standard', 'vip').")
    sla_deadline: str = Field(..., description="The SLA deadline for this ticket.")
    ticket_history: List[str] = Field(default_factory=list, description="Summary of past interactions.")
    step_count: int = Field(..., description="The current step in the task.")
    max_steps: int = Field(..., description="Maximum allowed steps.")
    
    # Advanced fields
    messages: List[Dict[str, str]] = Field(default_factory=list, description="History of tool calls and results.")
    available_tools: List[Dict[str, Any]] = Field(default_factory=list, description="List of tools the agent can use.")

class Reward(BaseModel):
    """The reward signal returned by the environment."""
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized reward score.")
    reason: str = Field(..., description="Explanation for the given score.")
    is_final: bool = Field(False, description="Whether this is the final step/reward.")
