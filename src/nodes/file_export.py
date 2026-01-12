from langchain_core.messages import SystemMessage

from src.llm import model_with_tools
from src.logger import logger
from src.state import State


async def node_file_export(state: State):
    """將報告輸出為ppt"""
    logger.info("=== 進入 node_file_export ===")
    last_message = state.messages[-1]
    prompt = f"""
    你是一位貼心的助理。
    請根據用戶的回覆決定下一步行動：
    1. 若用戶同意製作 PPT 簡報，請根據對話歷史中的『策略報告』內容，使用 generate_ppt 工具來生成檔案。
       - filename 請使用英文 (例如 strategy_report)
       - slides 請將報告內容拆解為多個頁面 (SlideContent)，第一頁標題首頁，第二頁開始每個頁面包含 header (標題) 與 items (重點列表)。
         並確保將報告中的主要章節（如目標、資源需求、實施步驟、結論）分別製作成不同的投影片。
    2. 若用戶不需要或拒絕，請禮貌回應並結束對話。
    策略報告標題首頁：{state.hmw_output}
    策略報告內容: {state.final_summary}
    """

    msg = await model_with_tools.ainvoke([SystemMessage(content=prompt), last_message])
    logger.info(f"File export: {msg.content}, Tool calls: {msg.tool_calls}")
    return {
        "messages": [msg],
        "node_status": "Exporting file",
        "last_stage": "file_export",
    }
