from typing import List, TypedDict


class AgentState(TypedDict):
    messages: List[str]
    cur_step: str
    situation_summary: str
    reflection_passed: bool
