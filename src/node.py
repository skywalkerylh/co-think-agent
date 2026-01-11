from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.llm import model, model_with_tools
from src.logger import logger


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


# --- 定義 LLM 輸出的結構 ---
class ProblemExtraction(BaseModel):
    job_title: Optional[str] = Field(description="用戶的職位，若無則留空")
    pain_point: Optional[str] = Field(description="用戶提到的問題痛點，若無則留空")
    goal: Optional[str] = Field(description="用戶想達成的目標，若無則留空")


class EvaluationDimensions(BaseModel):
    pain_point_score: int = Field(..., description="痛點描述的具體程度 (0-30)")
    goal_metric_score: int = Field(..., description="目標與指標的清晰度 (0-40)")
    box_trap_score: int = Field(..., description="是否跳脫『手段當目的』的陷阱 (0-30)")


class ProblemEvaluation(BaseModel):
    score: int = Field(..., description="總分 (0-100)")
    dimensions: EvaluationDimensions
    is_passing: bool = Field(..., description="是否通過門檻")
    critique: str = Field(..., description="犀利的評語")
    advice: str = Field(..., description="給用戶的引導建議")
    missing_fields: list[str] = Field(..., description="缺少的關鍵資訊欄位")


class CrossSiloOutput(BaseModel):
    result: Optional[str] = Field(..., description="跨部門視角的見解")
    score: int = Field(..., description="策略完整度分數 (0-100)")


class State(BaseModel):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    messages: Annotated[List[Any], add_messages]
    # 用來追蹤三個關鍵資訊蒐集狀況
    problem_profile: dict = Field(
        default_factory=lambda: {
            "pain_point": None,  # 痛點
            "goal": None,  # 目標
        }
    )

    # 評估結果
    reflection_result: dict = Field(
        default_factory=lambda: {
            "is_complete": False,
            "missing_fields": [],  # 例如 ["metric", "goal"]
        }
    )
    evaluation_result: dict = Field(
        default_factory=lambda: {
            "score": 0,
            "critique": "",
            "advice": "",
            "missing_fields": [],
        }
    )
    cross_silo_output: dict = Field(
        default_factory=lambda: {
            "result": "",
            "score": 0,
        }
    )
    job_title: Optional[str] = None
    is_passing_evaluation: bool = False
    node_status: str = "example"
    last_stage: str = ""  # 追蹤上一輪的階段: "reflection", "refine_ask",

    count_node_file_export: int = 0


async def node_situation(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    logger.info("=== 進入 node_situation ===")
    current_profile = state.problem_profile

    # 1. 呼叫 LLM 進行提取
    extraction_prompt = f"""
    目前已知的資訊: {current_profile}
    你是一個策略顧問，專門協助企業高層釐清他的職位與專案目標, 
    請分析主管的最新回答，提取或更新 'job_title', 'pain_point', 'goal'。
    
    規則：
    1. 如果主管的回答中包含了新的資訊，請提取並回傳。
    2. 如果主管的回答中沒有提到某項資訊，請回傳 None。
    3. 只有在主管明確想要修改或補充時才更新。
    """

    # 使用 with_structured_output 綁定 schema
    structured_model = model.with_structured_output(ProblemExtraction)
    messages = state.messages[-1]
    messages_to_send = [SystemMessage(content=extraction_prompt), messages]

    extracted_data: ProblemExtraction = await structured_model.ainvoke(messages_to_send)
    logger.info(f"extracted data: {extracted_data}")

    # 2. 合併資料 (Merge)
    # 因為 LLM 可能只回傳這次提取到的，我們要跟舊的 state 合併
    new_profile = current_profile.copy()

    # 只有在提取到有效內容時才累加，避免 None 覆蓋已有資訊
    if extracted_data.pain_point is not None:
        if new_profile["pain_point"]:
            # 累加新資訊
            new_profile["pain_point"] += " " + extracted_data.pain_point
        else:
            new_profile["pain_point"] = extracted_data.pain_point

    if extracted_data.goal is not None:
        if new_profile["goal"]:
            # 累加新資訊
            new_profile["goal"] += " " + extracted_data.goal
        else:
            new_profile["goal"] = extracted_data.goal

    logger.info(f"new profile: {new_profile}")

    # 3. 檢查缺失 (Check Missing)
    missing_fields = [k for k, v in new_profile.items() if not v]
    is_complete = len(missing_fields) == 0

    advice = ""
    if not is_complete:
        advice = f"目前還缺少以下資訊：{', '.join(missing_fields)}。請追問用戶。"
    logger.info(f"missing fields: {missing_fields}")
    logger.info(f"is complete: {is_complete}")

    if state.job_title is None and extracted_data.job_title is not None:
        job_title = extracted_data.job_title
    else:
        job_title = state.job_title

    return {
        "problem_profile": new_profile,
        "reflection_result": {
            "is_complete": is_complete,
            "missing_fields": missing_fields,
            "advice": advice,
        },
        "job_title": job_title,
        "node_status": f"output from situation. Configured with {(runtime.context or {}).get('my_configurable_param')}",
        "last_stage": "situation",
    }

async def node_reflection(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    logger.info("=== 進入 node_reflection ===")
    profile = state.problem_profile
    missing = state.reflection_result["missing_fields"]
    logger.info(f"reflection: missing fields{missing}")
    logger.info(f"problem profile: {profile}")
    system_prompt = f"""
    你是一個策略顧問，專門協助企業高層釐清專案目標。
    目前已知資訊如下：
    - 痛點: {profile["pain_point"]}
    - 目標: {profile["goal"]}
    - 缺失資訊: {missing}   
    你的任務是設計一個針對性的問題，引導主管補充缺失資訊"{missing}"。
    注意：
    - 不要打招呼也不要給標題，直接問問題
    """

    last_message = state.messages[-1]

    messages_to_send = [SystemMessage(content=system_prompt), last_message]
    msg = await model.ainvoke(messages_to_send)

    new_profile = profile.copy()
    if profile["pain_point"] is not None:
        new_profile["pain_point"] += msg.content if "pain_point" in missing else ""
    if profile["goal"] is not None:
        new_profile["goal"] += msg.content if "goal" in missing else ""

    return {
        "node_status": f"output from reflection. Configured with {(runtime.context or {}).get('my_configurable_param')}",
        "messages": [msg],
        "last_stage": "reflection",
        "problem_profile": new_profile,
    }


async def node_summary(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    logger.info("=== 進入 node_summary ===")
    profile = state.problem_profile
    logger.info({"before summary:": profile})
    prompt = f"""
    你是一位策略顧問，請針對痛點與目標的問答文本，分別針對痛點和目標做摘要, 需要記住主管每一個提到的點, 不可以漏掉任何資訊。
    痛點: {profile["pain_point"]}
    目標: {profile["goal"]}

    注意：
    - 只需回覆精簡摘要，無需多餘的說明或打招呼
    """

    structured_model = model.with_structured_output(ProblemExtraction)
    last_message = state.messages[-1]
    msg = await structured_model.ainvoke([SystemMessage(content=prompt), last_message])

    new_proflile = {
        "pain_point": msg.pain_point,
        "goal": msg.goal,
    }
    logger.info(f"after summary: {new_proflile}")
    return {
        "node_status": "Summary generated.",
        "problem_profile": new_proflile,
        "last_stage": "summary",
    }


async def node_evaluation(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
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
    structured_model = model.with_structured_output(ProblemEvaluation)
    last_message = state.messages[-1]
    messages_to_send = [SystemMessage(content=prompt), last_message]
    response = await structured_model.ainvoke(messages_to_send)

    # Format the evaluation result as a readable message
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
        "node_status": f"output from evaluation. Configured with {(runtime.context or {}).get('my_configurable_param')}",
        # "messages": [SystemMessage(content=evaluation_message)],
        "is_passing_evaluation": response.is_passing,
        "evaluation_result": eval_result,
        "last_stage": "evaluation",
    }


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
    logger.info(f"refine_ask prompt: {prompt}")
    last_message = state.messages[-1]
    msg = await model.ainvoke([SystemMessage(content=prompt), last_message])

    return {"messages": [msg], "last_stage": "refine_ask"}


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

        structured_model = model.with_structured_output(ProblemEvaluation)
        eval_result = await structured_model.ainvoke(
            [SystemMessage(content=prompt), last_message]
        )

        # 由於 ProblemEvaluation 沒有 content 欄位，我們使用 advice 作為回應
        response_content = eval_result.advice
        updated_result += f"\nAI Advice: {response_content}"

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


async def node_final_summary(state: State):
    """產生最終的問題描述總結."""
    logger.info("=== 進入 node_final_summary ===")

    profile = state.problem_profile
    prompt = f"""
    你是一位策略顧問，請根據以下資訊，產生一個完整且具體的策略報告，幫助用戶聚焦在核心議題上。
    痛點: {profile["pain_point"]}
    目標: {profile["goal"]}
    跨部門視角: {state.cross_silo_output["result"]}        
    注意：
    - 策略要具體且具備可行性，無需多餘的說明或打招呼
    """
    msg = await model.ainvoke([SystemMessage(content=prompt)])
    return {
        "messages": [msg],
        "node_status": "Strategy summary generated.",
        "last_stage": "final_summary",
    }


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
