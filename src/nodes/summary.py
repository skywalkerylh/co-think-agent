from typing import Any, Dict

from langchain_core.messages import SystemMessage

from src.llm import model_strict
from src.logger import logger
from src.state import ProblemExtraction, State


async def node_summary(state: State) -> Dict[str, Any]:
    logger.info("=== 進入 node_summary ===")
    profile = state.problem_profile
    logger.info({"before summary:": profile})

    prompt = f"""
    你是一位策略顧問，請針對痛點與目標的問答文本，分別針對痛點和目標做摘要, 每個資訊都用逗點隔開，不可以漏掉任何資訊。
    痛點: {profile["pain_point"]}
    目標: {profile["goal"]}

    注意：
    - 只需回覆精簡摘要，無需多餘的說明或打招呼
    """

    structured_model = model_strict.with_structured_output(ProblemExtraction)
    last_message = state.messages[-1]
    msg = await structured_model.ainvoke([SystemMessage(content=prompt), last_message])

    new_profile = {
        "pain_point": msg.pain_point,
        "goal": msg.goal,
    }
    logger.info(f"after summary: {new_profile}")
    return {
        "node_status": "Summary generated.",
        "problem_profile": new_profile,
        "last_stage": "summary",
    }
