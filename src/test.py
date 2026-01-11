from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from src.settings import settings
from src.state import AgentState

# 假設 llm 已經初始化
llm = ChatGoogleGenerativeAI(
    google_api_key=settings.GOOGLE_API_KEY,
    model=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE,
)
# --- 1. 定義節點 (Nodes) ---


def node_analyze_input(state: AgentState):
    """
    這個節點是「守門員」。
    它根據當前的 phase，決定是否要進行 Reflection (檢查)。
    """
    phase = state.phase
    messages = state.messages
    last_user_msg = messages[-1].content

    # 只有在 situation 階段需要嚴格檢查 (題目要求 Key Challenge)
    if phase == "situation":
        # 呼叫 LLM 判斷品質 (這裡簡化，實作時可用 structured output)
        check_prompt = f"分析用戶回答是否具體詳細：'{last_user_msg}'。如果太短或模糊回答 'NO'，否則回答 'YES'。"
        response = llm.invoke(check_prompt).content
        passed = "YES" in response.upper()

        return {"reflection_passed": passed}

    # 其他階段假設預設通過，或者你可以加入別的檢查邏輯
    return {"reflection_passed": True}


def node_response_router(state: AgentState):
    """
    這個節點不負責思考，只負責「說話」。
    根據 phase 和 reflection_passed 決定 AI 該說什麼。
    """
    phase = state.phase
    passed = state.reflection_passed

    # --- 邏輯分支 ---

    if phase == "situation":
        if not passed:
            # Reflection Loop: 追問
            msg = llm.invoke(
                state.messages
                + [
                    HumanMessage(
                        content="請禮貌地指出我提供的資訊太少，並引導我多說一點關於專案目標的細節。"
                    )
                ]
            )
            return {"messages": [msg]}  # 狀態不變，下次還是在 situation
        else:
            # 通過 -> 進入下一階段 Cross Silo
            # 這裡要做兩件事：1. 總結現況(可選) 2. 提出跨部門問題
            prompt = "用戶已清楚說明現況。請總結重點，並接著詢問：'從您的部門視角(HR/IT等)，這會如何影響其他部門？需要什麼資源？'"
            msg = llm.invoke(state.messages + [HumanMessage(content=prompt)])
            return {"messages": [msg], "phase": "cross_silo"}

    elif phase == "cross_silo":
        # 用戶回答了跨部門問題 -> 進入 AI Integration
        prompt = (
            "整合跨部門需求，並接著詢問：'在這些需求中，您認為 AI 可以應用在哪裡？'"
        )
        msg = llm.invoke(state.messages + [HumanMessage(content=prompt)])
        return {"messages": [msg], "phase": "ai_integration"}

    elif phase == "ai_integration":
        # 用戶回答了 AI 問題 -> 進入 Proposal
        return {"phase": "proposal"}  # 轉移到 proposal node 處理

    return {}


def node_generate_proposal(state: AgentState):
    """
    最終產出節點
    """
    # 這裡呼叫 LLM 生成 JSON
    prompt = "請根據上述所有對話，生成一份結構化的策略 JSON。"
    # 模擬生成
    msg = AIMessage(content="策略報告已生成... (這裡會觸發 File Gen)")
    return {"messages": [msg], "phase": "done", "collected_data": {"file_ready": True}}


def node_file_gen(state: AgentState):
    # 這裡實作 Python Pandas/PPTX 邏輯
    # print("Generating .xlsx file...")
    return {}


# --- 2. 建立 Graph 連線 (Edges) ---

workflow = StateGraph(AgentState)

# 新增節點
workflow.add_node("analyze_input", node_analyze_input)
workflow.add_node("bot_response", node_response_router)
workflow.add_node("create_proposal", node_generate_proposal)
workflow.add_node("create_file", node_file_gen)

# 設定進入點：使用者輸入後，先進入分析
workflow.set_entry_point("analyze_input")


# 設定條件邊 (Conditional Edge)：決定分析完要去哪
def route_after_analysis(state):
    phase = state.phase
    if phase == "proposal":
        return "create_proposal"  # 特例：如果是最後階段，去產報告
    return "bot_response"  # 一般情況：去生成回覆


workflow.add_conditional_edges(
    "analyze_input",
    route_after_analysis,
    {"create_proposal": "create_proposal", "bot_response": "bot_response"},
)

# 機器人回覆完，這一次的執行就結束 (END)，等待使用者下一次輸入
workflow.add_edge("bot_response", END)

# Proposal 生成完 -> 產檔案 -> 結束
workflow.add_edge("create_proposal", "create_file")
workflow.add_edge("create_file", END)

# 編譯 App
#memory = MemorySaver()
app = workflow.compile()
