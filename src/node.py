from langchain_core.messages import SystemMessage

from src.llm import llm
from src.logger import logger
from src.state import AgentState


def node_situation(state: AgentState):
    """
    訪談 Agent：針對日常營運困擾與未來影響力危機進行訪談，收集具體痛點。
    """
    last_user_msg = state.messages[-1].content
    
    # 根據 Reflection 結果調整 System Prompt
    reflection_feedback = ""
    if not state.reflection_passed and state.reflection_reason:
        reflection_feedback = f"\n\n**注意：上一輪的回答不夠具體或完整。Checker 的回饋是：{state.reflection_reason}。請針對此點進行追問。**"
    
    system_prompt = f"""你是一位頂尖的訪談 Agent，專門協助企業高層挖掘核心問題。
你的任務是透過對話，針對高層挑選的面向{last_user_msg}，分別收集到 2-3 個具體的痛點：
根據挑選的面向請分別問不同問題
1. **日常營運困擾**：
   - 請詢問：「在您日常的決策與營運中，近期有哪些讓您感到『困擾』或效率受阻的痛點，特別是那些反覆出現的問題？」
2. **未來影響力危機**：
   - 請詢問：「在哪些瞬間，你深感『如果不改變這件事，我們將失去未來的影響力』？」

**訪談規則 (Best Practice)：**
- **一次只問一個問題**：不要一次拋出所有面向，請循序漸進。
- **追問具體細節**：如果用戶的回答模糊（例如：「效率不高」、「溝通不良」），你**必須**追問：「請給我一個這週發生的具體例子」。
- **確認數量**：確保每個面向都收集到 2-3 個具體痛點。
- **使用台灣用語**

**Output (結束條件)：**
當你收集完所有必要的資訊後，請整理出一份完整的「痛點訪談總結」，列出所有收集到的具體痛點，並確認高層資訊資訊是否正確。
{reflection_feedback}
"""

    # 將 SystemMessage 加入對話歷史的最前端
    messages = [SystemMessage(content=system_prompt)] + state.messages

    response = llm.invoke(messages)

    logger.info("node_situation executed")

    # 回傳 AI 的回應，這會被 append 到 state.messages
    return {"messages": [response],
            "situation_summary": response.content.strip()}
            


def node_reflection(state: AgentState):
    """
    檢查者 Agent：分析對話紀錄，判斷是否已收集到足夠且具體的痛點。
    """
    # 取得最近的對話紀錄
    messages = state.messages
    
    # 檢查 Prompt
    check_prompt = """
    你是一位嚴格的品質檢查員 (Checker)。請分析上述對話紀錄，判斷訪談 Agent 是否已經成功收集到以下資訊：

    1. **日常營運困擾**：至少 2 個「具體」痛點（例如：「每週花 3 小時手動整理報表」是具體的；「效率不高」是不具體的）。
    2. **未來影響力危機**：至少 2 個「具體」痛點。

    **判斷規則：**
    - 如果資訊充足且具體，請回答 "PASS"。
    - 如果資訊不足（數量不夠）或太模糊，請回答 "FAIL: <原因>"，並簡短說明缺少的資訊（例如：「缺少未來影響力危機的具體例子」）。

    **Output Format:**
    只回傳 "PASS" 或 "FAIL: <原因>"
    """
    
    # 將對話紀錄與檢查 Prompt 結合
    check_messages = messages + [SystemMessage(content=check_prompt)]
    
    response = llm.invoke(check_messages)
    content = response.content.strip()
    
    logger.info(f"node_reflection result: {content}")
    
    if "PASS" in content.upper():
        return {"reflection_passed": True, "reflection_reason": ""}
    else:
        # 提取失敗原因
        reason = content.replace("FAIL:", "").strip()
        return {"reflection_passed": False, "reflection_reason": reason}



# def node_cross_silo(state: AgentState):
#     reply = node_cross_silo.__name__
#     messages = state.messages + [AIMessage(content=reply)]
#     logger.info("node_cross_silo")
#     return {"messages": messages}


# def node_integration(state: AgentState):
#     reply = node_integration.__name__
#     messages = state.messages + [AIMessage(content=reply)]
#     logger.info("node_integration")
#     return {"messages": messages}


# def node_proposal(state: AgentState):
#     reply = node_proposal.__name__
#     messages = state.messages + [AIMessage(content=reply)]

#     logger.info("node_proposal")
#     return {"messages": messages}


# def node_export(state: AgentState):
#     reply = node_export.__name__
#     messages = state.messages + [AIMessage(content=reply)]

#     logger.info("node_export")
#     return {"messages": messages}
