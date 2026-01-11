from langchain_core.messages import SystemMessage

from src.llm import model
from src.logger import logger
from src.state import State


async def node_refine_ask(state: State):
    logger.info("=== 進入 node_refine_ask ===")
    result = state.evaluation_result
    advice = result["advice"]
    critique = result["critique"]
    missing = result["missing_fields"]
    logger.info(f"refine_ask result: {result}")

    prompt = f"""
    剛才的評估結果顯示高層定義問題可以更好。
    評語：{critique}
    建議方向：{advice}
    缺失資訊：{missing}

    請根據上述建議，扮演親切但專業的顧問，用200字以內的問題引導高層深入挖掘議題以補足缺失資訊。
    注意：
    - 不要打招呼也不要給標題，直接問問題
    - 問題要精簡，且附上一個舉例幫助理解
    """
    last_message = state.messages[-1]
    msg = await model.ainvoke([SystemMessage(content=prompt), last_message])

    return {"messages": [msg], "last_stage": "refine_ask"}
