from langchain_core.messages import SystemMessage, AIMessage

from src.llm import model
from src.logger import logger
from src.state import State


async def node_final_summary(state: State):
    """產生最終的問題描述總結."""
    logger.info("=== 進入 node_final_summary ===")

    prompt = f"""
    你是一位策略顧問，請根據以下資訊，產生一個完整且具體的策略報告，幫助用戶聚焦在核心議題上。
    要有這幾個標題
    - {state.hmw_output}
    - 目標
    - 梯形分析：尋找問題甜蜜點
    - 痛點
    - 跨部門視角的整合分析
    - 實作步驟
    - 結論

    跨部門視角: {state.cross_silo_evaluation["result"]}        
    注意：
    - 策略要具體且具備可行性，無需多餘的說明或打招呼
    - 請用梯級分析，往下生成三個問題。
    梯級分析：一個上下顛倒的三角形，頂端是非常廣泛的問題，底端是非常狹隘的問題。
    在這兩者之間，則有不同的梯度，可以往金字塔的上方或下方走，把問題擴大或縮小，一直嘗試找到適當的範疇。
    舉例：
    擬定的問題是「我們如何減少全世界的飢餓問題？」
    往下走生成提問：「我們如何減少貧困國家的飢餓問題？」
    甚至更往下走，把問題變成：「我們如何減少富裕國家裡窮人的飢餓問題？」
    """
    
    msg = await model.ainvoke([SystemMessage(content=prompt)])
    logger.info(f"Final summary: {msg.content}")
    response_content = msg.content 
    
    return {
        "messages": [AIMessage(content=response_content)],
        "final_summary": msg.content,
        "node_status": "Strategy summary generated.",
        "last_stage": "final_summary",
    }
