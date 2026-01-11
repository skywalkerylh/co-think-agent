from langchain_core.messages import SystemMessage

from src.llm import model_with_tools
from src.logger import logger
from src.state import State


async def node_file_export(state: State):
    """將報告輸出為ppt"""
    logger.info("=== 進入 node_file_export ===")

    prompt = """
    你是一位貼心的助理。
    請根據用戶的回覆決定下一步行動：
    1. 若用戶同意製作 PPT 簡報，請根據對話歷史中的『策略報告』內容，使用 generate_ppt 工具來生成檔案。
       - filename 請使用英文 (例如 strategy_report)
       - title 請使用報告的標題
       - bullet_points 請將報告中的關鍵策略整理成列點
    2. 若用戶不需要或拒絕，請禮貌回應並結束對話。
    """

    msg = await model_with_tools.ainvoke(
        [SystemMessage(content=prompt)] + state.messages
    )
    logger.info(f"File export: {msg.content}, Tool calls: {msg.tool_calls}")
    return {
        "messages": [msg],
        "node_status": "Exporting file",
        "last_stage": "file_export",
    }
