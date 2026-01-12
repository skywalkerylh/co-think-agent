from langchain_core.messages import AIMessage, SystemMessage

from src.llm import model, model_strict
from src.logger import logger
from src.state import CrossSiloEvaluation, State


async def node_cross_silo_ask(state: State):
    """跨部門視角：進行提問 (Ask Phase)"""
    logger.info("=== 進入 node_cross_silo_ask ===")
    
    prompt = f"""
    你是一位跨領域的策略顧問，專門協助高層從跨部門的角度審視問題所需要的資源。
    詢問主管完整句子：“從{state.job_title}職位來看，解決這個問題有什麼影響力？您需要其他部門提供哪些資源或能力來協助解決？
    並從主管職位舉個例子，說明可能需要的資源。並詢問是否有要補充或是資訊是否正確。: 
    職位：{state.job_title}
    要解決的問題：{state.hmw_output}
    注意：
    - 只需回覆，無需多餘的說明或打招呼
    - 例子要從高層的職位出發，並且具體說明部門可能需要的資源
    """
    msg = await model.ainvoke([SystemMessage(content=prompt)])
    updated_result = f"AI Question: {msg.content}"
    logger.info(f"Cross-silo ask: {msg.content}")
    
    return {
        "messages": [msg],
        "cross_silo_evaluation": {
            "result": updated_result,
            "score": 0,
        },
        "node_status": "Asking cross-silo resources.",
        "last_stage": "cross_silo_ask",
    }


async def node_cross_silo_evaluate(state: State):
    """跨部門視角：評估回答 (Evaluate Phase)"""
    logger.info("=== 進入 node_cross_silo_evaluate ===")
    current_result = state.cross_silo_evaluation.get("result", "")
    last_message = state.messages[-1]
    updated_result = current_result + f"\nUser Answer: {last_message.content}"

    prompt = f"""
    你是一位跨領域的策略顧問，專門協助高層從跨部門的角度審視問題所需要的資源。
    根據先前的討論，繼續回答問題或是問一個問題引導主管深入思考。
    同時，給予討論完整性打一個分數 (0-100)，並給予當前總結、建議和理由。
    職位：{state.job_title}
    要解決的問題：{state.hmw_output}
    先前討論：{updated_result}
    
    注意：
    - 只需回覆，無需多餘的說明或打招呼
    - 例子要從高層的職位出發，並且具體說明部門可能需要的資源
    """

    structured_model = model_strict.with_structured_output(CrossSiloEvaluation)
    eval_result = await structured_model.ainvoke(
        [SystemMessage(content=prompt), last_message]
    )

    # based on score, decide whether to continue asking or not
    if eval_result.score < 65:
        response_content = eval_result.advice
        updated_result += f"\nAI Advice: {response_content}"
    else:
        response_content = "您的回答已完整"

    logger.info(f"Cross-silo score: {eval_result.score}")

    return {
        "messages": [AIMessage(content=response_content)],
        "cross_silo_evaluation": {
            "result": updated_result,
            "score": eval_result.score,
            "advice": eval_result.advice,
        },
        "node_status": "Cross-silo perspectives evaluated.",
        "last_stage": "cross_silo_evaluate",
    }
