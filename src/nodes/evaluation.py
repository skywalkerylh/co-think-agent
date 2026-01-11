from typing import Any, Dict

from langchain_core.messages import SystemMessage

from src.llm import model_strict
from src.logger import logger
from src.state import ProblemEvaluation, State


async def node_evaluation(state: State) -> Dict[str, Any]:
    """如果資訊都齊全了，就評估問題描述的品質，並給分數和建議."""
    logger.info("=== 進入 node_evaluation ===")
    profile = state.problem_profile
    prompt = f"""
    # Role
    你是一位專精於「破框思維 (Break-the-Box Thinking)」的台灣策略顧問。你的任務不是單純的聊天，而是用親切且中肯的語氣評估用戶提出的「問題陳述」是否具備戰略解決的價值。

    # Task
    請分析用戶輸入的文字，根據以下四個維度進行嚴格評分 (0-100)，並給出中肯的評語。
    痛點：{profile["pain_point"]}
    目標：{profile["goal"]}

    # Evaluation Criteria (Rubric)

    1. **Pain Point (30分)**
    - 0-10分: 只說了感覺 (e.g., "很累", "很難")。
    - 11-20分: 提到了大致狀況，但缺乏情境。
    - 21-30分: 清楚描述了 "誰" 在 "什麼情境" 下遇到了 "什麼具體阻礙"。

    2. **Goal & Metric (40分)**
    - 0-10分: 完全沒提到目標或數字。
    - 11-25分: 有目標但無量化指標 (e.g., "想提升效率")。
    - 26-40分: 有明確的成功定義與量化指標 (e.g., "提升 20% 轉換率")。

    3. **Solution Bias (Box Trap) (30分)**
    - 警告：這是破框思維的核心。
    - 0分 (陷入框框): 用戶直接把 "解決方案" 當成問題 (e.g., "我需要導入 AI", "我需要做一個 App")。這不是問題，這是手段。
    - 30分 (破框): 用戶專注於 "想解決的本質困難" 或 "想創造的價值"，而非限定某種工具。

        """
    structured_model = model_strict.with_structured_output(ProblemEvaluation)
    last_message = state.messages[-1]
    messages_to_send = [SystemMessage(content=prompt), last_message]
    response = await structured_model.ainvoke(messages_to_send)

    evaluation_message = f"""
        📊 **評分結果** (總分: {response.score}/100)

        **評分細項:**
        - 痛點描述: {response.dimensions.pain_point_score}/30
        - 目標與指標: {response.dimensions.goal_metric_score}/40
        - 破框思維: {response.dimensions.box_trap_score}/30

        **評語:**
        {response.critique}

        **建議:**
        {response.advice}
        """

    if response.score >= 65:
        response.is_passing = True

    eval_result = {
        "score": response.score,
        "critique": response.critique,
        "advice": response.advice,
        "missing_fields": response.missing_fields,
    }
    logger.info(f"Returning evaluation_result: {eval_result}")
    logger.info(f"is_passing_evaluation: {response.is_passing}")

    return {
        "node_status": "output from evaluation.",
        "is_passing_evaluation": response.is_passing,
        "evaluation_result": eval_result,
        "last_stage": "evaluation",
    }
