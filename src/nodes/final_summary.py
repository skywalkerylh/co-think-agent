from langchain_core.messages import SystemMessage

from src.llm import model
from src.logger import logger
from src.state import State


async def node_final_summary(state: State):
    """產生最終的問題描述總結."""
    logger.info("=== 進入 node_final_summary ===")

    profile = state.problem_profile
    prompt = f"""
    你是一位策略顧問，請根據以下資訊，產生一個完整且具體的策略報告，幫助用戶聚焦在核心議題上。
    痛點: {profile["pain_point"]}
    目標: {profile["goal"]}
    跨部門視角: {state.cross_silo_output["result"]}        
    注意：
    - 策略要具體且具備可行性，無需多餘的說明或打招呼
    """
    msg = await model.ainvoke([SystemMessage(content=prompt)])
    return {
        "messages": [msg],
        "node_status": "Strategy summary generated.",
        "last_stage": "final_summary",
    }
