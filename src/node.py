
from src.state import AgentState
from src.logger import logger

def node_situation(state: AgentState):
    # 調用 LLM 進行對話
    # Prompt: "你是一個策略顧問，請引導用戶釐清目標..."

    logger.info("node_situation")
    return {"cur_step": "situation"}


def node_reflection(state: AgentState):
    # 這是專門的 "Checker"
    # Prompt: "請分析用戶的最新回答：{last_msg}。這是否足夠詳細？回答 YES 或 NO。"
    # 如果 NO，生成一個引導問題
    logger.info("node_reflection")
    return {"reflection_passed": True}


def node_cross_silo(state: AgentState):
    logger.info("node_cross_silo")
    pass


def node_integration(state: AgentState):
    logger.info("node_integration")
    pass


def node_proposal(state: AgentState):
    logger.info("node_proposal")
    pass


def node_export(state: AgentState):
    logger.info("node_export")
    pass
