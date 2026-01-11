from typing import List, Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages

class AgentState(BaseModel):
    messages:  Annotated[List, add_messages]
    phase: str
    situation_summary: str
    reflection_passed: bool
    reflection_reason: str = ""
