from langchain_core.messages import AIMessage, SystemMessage

from src.llm import model, model_strict
from src.logger import logger
from src.state import ProblemEvaluation, State


async def node_cross_silo(state: State):
    """從跨部門視角審視痛點和目標"""
    logger.info("=== 進入 node_cross_silo ===")
    current_result = state.cross_silo_output.get("result", "")

    # 如果 result 為空，代表是第一次進入此節點，進行提問
    if not current_result:
        prompt = f"""
        你是一位跨領域的策略顧問，專門協助高層從跨部門的角度審視問題所需要的資源。
        詢問主管完整句子：“從{state.job_title}職位來看，這個痛點對您有什麼影響？您需要其他部門提供哪些資源或能力來協助解決？
        並從主管職為以及先前給的痛點和目標舉個例子引導他思考: 
        職位：{state.job_title}
        痛點與目標：{state.problem_profile}
        注意：
        - 只需回覆，無需多餘的說明或打招呼
        - 例子要從高層的職位出發，並且具體說明部門可能需要的資源
        """
        msg = await model.ainvoke([SystemMessage(content=prompt)])
        updated_result = f"AI Question: {msg.content}"

        return {
            "messages": [msg],
            "cross_silo_output": {
                "result": updated_result,
                "score": 0,
            },
            "node_status": "Asking cross-silo resources.",
            "last_stage": "cross_silo",
        }

    else:
        # 非第一次進入，代表用戶已回覆，進行評估
        last_message = state.messages[-1]
        updated_result = current_result + f"\nUser Answer: {last_message.content}"

        prompt = f"""
        你是一位跨領域的策略顧問，專門協助高層從跨部門的角度審視問題所需要的資源。
        根據先前的討論，繼續回答問題或是問一個問題引導主管深入思考。
        同時，給予討論完整性打一個分數 (0-100)，並給予建議和理由。
        職位：{state.job_title}
        痛點與目標：{state.problem_profile}
        先前討論：{updated_result}
        
        評分標準：
        - < 65分：回答模糊或缺乏具體跨部門資源需求 -> 繼續追問
        - >= 65分：回答具體且完整 -> 不回應
        
        注意：
        - 只需回覆，無需多餘的說明或打招呼
        - 例子要從高層的職位出發，並且具體說明部門可能需要的資源
        """

        structured_model = model_strict.with_structured_output(ProblemEvaluation)
        eval_result = await structured_model.ainvoke(
            [SystemMessage(content=prompt), last_message]
        )
        # update result
        response_content = eval_result.advice
        updated_result += f"\nAI Advice: {response_content}"
        if eval_result.score >= 65:
            response_content = ""
        logger.info(f"Cross-silo score: {eval_result.score}")

        return {
            "messages": [AIMessage(content=response_content)],
            "cross_silo_output": {
                "result": updated_result,
                "score": eval_result.score,
            },
            "node_status": "Cross-silo perspectives evaluated.",
            "last_stage": "cross_silo",
        }
