from typing import Any, Dict

from langchain_core.messages import SystemMessage

from src.llm import model
from src.logger import logger
from src.state import State


async def node_reflection(state: State) -> Dict[str, Any]:
    logger.info("=== 進入 node_reflection ===")
    profile = state.problem_profile
    missing = state.reflection_result["missing_fields"]
    logger.info(f"reflection: missing fields{missing}")
    logger.info(f"problem profile: {profile}")

    system_prompt = f"""
    你是一個策略顧問，專門協助企業高層釐清專案目標。
    目前已知資訊如下：
    - 痛點: {profile["pain_point"]}
    - 目標: {profile["goal"]}
    - 缺失資訊: {missing}   
    你的任務是設計一個針對性的問題，引導主管補充缺失資訊"{missing}"。
    注意：
    - 不要打招呼也不要給標題，直接問問題
    """

    last_message = state.messages[-1]
    messages_to_send = [SystemMessage(content=system_prompt), last_message]
    msg = await model.ainvoke(messages_to_send)

    # update profile with new info from reflection
    new_profile = profile.copy()
    if profile["pain_point"] is not None:
        new_profile["pain_point"] += msg.content if "pain_point" in missing else ""
    if profile["goal"] is not None:
        new_profile["goal"] += msg.content if "goal" in missing else ""

    return {
        "node_status": "output from reflection.",
        "messages": [msg],
        "last_stage": "reflection",
        "problem_profile": new_profile,
    }
