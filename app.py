"""Streamlit UI for AI Strategy Consultant Agent."""

import nest_asyncio
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()
import asyncio
import os
from typing import Any, Dict

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.graph import graph
from src.state import State

# Page configuration
st.set_page_config(
    page_title="AI 策略顧問",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    .status-complete {
        background-color: #d4edda;
        color: #155724;
    }
    .status-missing {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "problem_profile" not in st.session_state:
        st.session_state.problem_profile = {
            "pain_point": None,
            "goal": None,
        }
    if "reflection_result" not in st.session_state:
        st.session_state.reflection_result = {
            "is_complete": False,
            "missing_fields": [],
        }
    if "is_passing_evaluation" not in st.session_state:
        st.session_state.is_passing_evaluation = False
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    if "show_greeting" not in st.session_state:
        st.session_state.show_greeting = True
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = {
            "score": 0,
            "critique": "",
            "advice": "",
            "missing_fields": [],
        }
    if "job_title" not in st.session_state:
        st.session_state.job_title = None
    if "cross_silo_output" not in st.session_state:
        st.session_state.cross_silo_output = {
            "result": "",
            "score": 0,
        }
    if "node_status" not in st.session_state:
        st.session_state.node_status = "example"
    if "last_stage" not in st.session_state:
        st.session_state.last_stage = ""


def reset_conversation():
    """Reset the conversation to start fresh."""
    st.session_state.messages = []
    st.session_state.problem_profile = {
        "pain_point": None,
        "goal": None,
    }
    st.session_state.job_title = None
    st.session_state.reflection_result = {
        "is_complete": False,
        "missing_fields": [],
    }
    st.session_state.is_passing_evaluation = False
    st.session_state.conversation_started = False
    st.session_state.show_greeting = True
    st.session_state.cross_silo_output = {
        "result": "",
        "score": 0,
    }
    st.session_state.node_status = "example"
    st.session_state.last_stage = ""
    st.rerun()


async def process_user_input(user_message: str) -> Dict[str, Any]:
    """Process user input through the agent graph."""
    # Create state with current context
    state = State(
        messages=st.session_state.messages + [HumanMessage(content=user_message)],
        problem_profile=st.session_state.problem_profile,
        reflection_result=st.session_state.reflection_result,
        is_passing_evaluation=st.session_state.is_passing_evaluation,
        job_title=st.session_state.job_title,
        cross_silo_output=st.session_state.cross_silo_output,
        node_status=st.session_state.node_status,
        last_stage=st.session_state.last_stage,
    )

    # Run the graph asynchronously
    result = await graph.ainvoke(state)

    return result


def display_message(message: Any):
    """Display a message in the chat interface."""
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
    elif isinstance(message, SystemMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
    elif isinstance(message, ToolMessage):
        with st.chat_message("assistant", avatar="🤖"):
            if "成功生成檔案" in message.content:
                # Extract filename
                file_path = message.content.split("：")[-1].strip()
                if os.path.exists(file_path):
                    with open(file_path, "rb") as file:
                        st.download_button(
                            label="📥 下載策略報告 PPT",
                            data=file,
                            file_name=os.path.basename(file_path),
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            key=f"download_{file_path}",  # Add unique key based on path
                        )
                    st.success(f"報告已準備完成！ ({os.path.basename(file_path)})")
                else:
                    st.error(f"檔案生成回應顯示成功，但找不到檔案: {file_path}")
            else:
                st.info(f"工具執行結果: {message.content}")
    elif isinstance(message, str):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message)


def render_sidebar():
    """Render the sidebar with status and controls."""
    with st.sidebar:
        st.title("📊 問題分析狀態")

        # Problem Profile Status
        st.subheader("收集資訊進度")

        profile = st.session_state.problem_profile

        # Pain Point
        if profile["pain_point"]:
            st.markdown(
                '<div class="status-badge status-complete">✓ 痛點已收集</div>',
                unsafe_allow_html=True,
            )
            with st.expander("查看痛點"):
                st.write(profile["pain_point"])
        else:
            st.markdown(
                '<div class="status-badge status-missing">⊗ 痛點待補充</div>',
                unsafe_allow_html=True,
            )

        # Goal
        if profile["goal"]:
            st.markdown(
                '<div class="status-badge status-complete">✓ 目標已收集</div>',
                unsafe_allow_html=True,
            )
            with st.expander("查看目標"):
                st.write(profile["goal"])
        else:
            st.markdown(
                '<div class="status-badge status-missing">⊗ 目標待補充</div>',
                unsafe_allow_html=True,
            )

        # Overall Status
        st.divider()
        st.subheader("整體評估")

        if st.session_state.is_passing_evaluation:
            st.success("✅ 問題定義已達標準！")
        elif st.session_state.reflection_result["is_complete"]:
            st.info("🔄 資訊已收集完整，正在評估品質...")
        else:
            missing = st.session_state.reflection_result.get("missing_fields", [])
            if missing:
                st.warning(f"⚠️ 待補充資訊: {', '.join(missing)}")
            else:
                st.info("💭 開始對話以收集資訊")

        # Controls
        st.divider()
        if st.button("🔄 重新開始", use_container_width=True):
            reset_conversation()

        # Instructions
        st.divider()
        st.subheader("💡 使用提示")
        st.markdown(
            """
        1. **描述痛點**: 說明你遇到的具體問題
        2. **明確目標**: 你想達成什麼成果？
        3. **避免框架**: 不要直接提解決方案，描述想解決的困難
        4. **量化指標**: 最好有可衡量的成功標準
        """
        )


def main():
    """Main application logic."""
    init_session_state()

    # Header
    st.title("💡 AI 策略顧問")
    st.markdown("幫助你釐清專案目標，定義有價值的問題")

    # Render sidebar
    render_sidebar()

    # Main chat area
    col1, col2 = st.columns([3, 1])

    with col1:
        # Display greeting message if first time
        if st.session_state.show_greeting and not st.session_state.conversation_started:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    """
                👋 你好！我是 **AI 策略顧問**，幫助你釐清專案目標。

                📝 **請告訴我您的職位，並寫出日常會遭遇而且希望自己可以解決的問題。**

                💭 不要受限於可不可能解決，請先跳脫這一點，寫下你想到的每一件事。

                ❓ **思考方向:**
                - 哪些問題讓你非常煩惱、最想解決？
                - 哪些問題不斷出現？
                - 如果問題不斷重複出現，可能就是你選擇解決它的理由

                現在，請告訴我你想解決的問題 👇
                """
                )

        # Display conversation history
        for message in st.session_state.messages:
            display_message(message)

        # Chat input
        user_input = st.chat_input("輸入你的訊息...")

        if user_input:
            st.session_state.conversation_started = True
            st.session_state.show_greeting = False

            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            # Show thinking indicator
            with st.spinner("🤔 AI 正在思考..."):
                # Process through agent using existing event loop
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(process_user_input(user_input))

                # Update session state with results
                st.session_state.messages = result["messages"]
                st.session_state.problem_profile = result["problem_profile"]
                st.session_state.reflection_result = result["reflection_result"]
                st.session_state.is_passing_evaluation = result["is_passing_evaluation"]
                st.session_state.evaluation_result = result["evaluation_result"]
                st.session_state.job_title = result["job_title"]
                st.session_state.cross_silo_output = result["cross_silo_output"]
                st.session_state.node_status = result["node_status"]
                st.session_state.last_stage = result["last_stage"]

                # Display only the latest AI response
                latest_message = result["messages"][-1]
                display_message(latest_message)

            # Rerun to update sidebar
            st.rerun()

    with col2:
        # Quick actions or tips
        st.markdown("### 🎯 快速提示")
        with st.expander("好的問題描述範例", expanded=False):
            st.markdown(
                """
            **範例 1:**
            "我們公司的客戶流失率很高，每季約有 15% 的客戶不再續約。我們希望能在未來半年內將流失率降低到 8%。"

            **範例 2:**
            "業務團隊花太多時間在行政作業上，平均每天要花 3 小時處理報表，導致實際拜訪客戶的時間不足。我們希望讓業務有更多時間專注在銷售上。"
            """
            )

        with st.expander("應避免的描述", expanded=False):
            st.markdown(
                """
            ❌ **太模糊:**
            "我們需要提升效率"

            ❌ **直接說解決方案:**
            "我們需要導入 AI"

            ❌ **缺乏量化:**
            "希望業績變好"

            ✅ **改善後:**
            "我們的訂單處理時間平均需要 3 天，希望能縮短到 1 天內完成，以提升客戶滿意度"
            """
            )


if __name__ == "__main__":
    main()
