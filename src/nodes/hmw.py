from langchain_core.messages import SystemMessage

from src.llm import model
from src.logger import logger
from src.state import State


async def node_hmw_gen(state: State):
    """將問題改為如何...開頭."""
    logger.info("=== 進入 node_hmw_gen ===")
    prompt = f"""
    你是一位策略顧問，請將以下問題陳述改寫成以下的回覆格式，幫助用戶聚焦在解決方案的探索上。
    問題陳述: {state.problem_profile}
    回覆格式範例: 總結您想解決的問題：在...的情境下，如何...？
    注意：
    - 無需多餘的說明或打招呼
    
    """
    last_message = state.messages[-1]
    msg = await model.ainvoke([SystemMessage(content=prompt), last_message])
    return {
        "messages": [msg],
        "node_status": "HMW question generated.",
        "last_stage": "hmw_gen",
    }
