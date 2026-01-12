from typing import Any, Dict

from langchain_core.messages import SystemMessage

from src.llm import model_strict
from src.logger import logger
from src.state import ProblemExtraction, State


async def node_situation(state: State) -> Dict[str, Any]:
    logger.info("=== 進入 node_situation ===")
    current_profile = state.problem_profile

    extraction_prompt = f"""
    目前已知的資訊: {current_profile}
    你是一個策略顧問，專門協助企業高層釐清他的職位與專案目標。
    
    請分析主管的最新回答，判斷是否需要整合及更新 'job_title', 'pain_point', 'goal'。
    假設主管回答很模糊，需轉換為具體的句子來補充資訊。例如：“做AI” -> "導入AI技術"
    
    規則：
    - 若主管的回答中未提及某項資訊或資訊未變更，請回傳 None。
    - 只有在主管明確想要修改或補充時才更新。
    """
    structured_model = model_strict.with_structured_output(ProblemExtraction)
    messages = state.messages[-1]
    messages_to_send = [SystemMessage(content=extraction_prompt), messages]
    extracted_data: ProblemExtraction = await structured_model.ainvoke(messages_to_send)
    logger.info(f"extracted data: {extracted_data}")

    # update data
    new_profile = current_profile.copy()
    if new_profile["pain_point"] is not None:
        if extracted_data.pain_point is not None:
            new_profile["pain_point"] += "," + extracted_data.pain_point
    else:
        new_profile["pain_point"] = extracted_data.pain_point

    if new_profile["goal"] is not None:
        if extracted_data.goal is not None:
            new_profile["goal"] += "," + extracted_data.goal
    else:
        new_profile["goal"] = extracted_data.goal

    #  update job title
    if state.job_title is None and extracted_data.job_title is not None:
        job_title = extracted_data.job_title
    else:
        job_title = state.job_title

    logger.info(f"new profile: {new_profile}")
    # check missing fields
    missing_fields = [k for k, v in new_profile.items() if not v]
    is_complete = len(missing_fields) == 0

    # generate advice for reflection
    advice = ""
    if not is_complete:
        advice = f"目前還缺少以下資訊：{', '.join(missing_fields)}。請追問用戶。"

    logger.info(f"missing fields: {missing_fields}")
    logger.info(f"is complete: {is_complete}")

    return {
        "problem_profile": new_profile,
        "reflection_result": {
            "is_complete": is_complete,
            "missing_fields": missing_fields,
            "advice": advice,
        },
        "job_title": job_title,
        "node_status": "output from situation.",
        "last_stage": "situation",
    }
