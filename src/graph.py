"""LangGraph for Streamlit UI (without interrupt nodes)."""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.llm import tools
from src.logger import logger
from src.nodes import (
    node_cross_silo_ask,
    node_cross_silo_evaluate,
    node_evaluation,
    node_file_export,
    node_final_summary,
    node_hmw_gen,
    node_refine_ask,
    node_reflection,
    node_situation,
    node_summary,
)
from src.state import State


def route_after_situation(state: State) -> str:
    """Route based on whether information is complete."""
    # 如果上一輪是 refine_ask，直接進入 summary, evaluation 重新評估
    if state.last_stage == "refine_ask":
        return "summary"

    if state.reflection_result["is_complete"]:
        return "summary"  # 資訊齊全，進入下一關
    else:
        return "reflection"  # 資訊不齊全，進入追問


def route_after_evaluation(state: State) -> str:
    """Route based on whether evaluation suggests refinement."""
    if state.is_passing_evaluation:
        return "hmw_gen"  # 資訊齊全，進入下一關
    else:
        return "refine_ask"  # 資訊不齊全，進入追問


def route_after_cross_silo(state: State) -> str:
    """Route based on whether cross-silo information is complete."""
    score = state.cross_silo_evaluation.get("score", 0)
    if score < 65:
        return END  # 分數低於 65，中斷等待用戶回答（繼續對話）
    else:
        return "final_summary"  # 分數達標，進入總結


def route_start(state: State) -> str:
    """Determine where to start based on state."""
    logger.info({"route_start": state.last_stage})
    if state.last_stage == "file_export":
        return "file_export"

    result = state.cross_silo_evaluation.get("result", "")
    score = state.cross_silo_evaluation.get("score", 0)

    # 進入 evaluate 的條件：已經有 result (代表 ask 過了) 且 用戶剛回答完 (last_stage 可能是 ask 或 evaluate loop)
    # 如果 score >= 65 會在 route_after_cross_silo 就走 final_summary，所以這裡處理的是未完成的 loop
    if result and score < 65:
        return "cross_silo_evaluate"

    return "situation"


# Create workflow for Streamlit (skip greeting node with interrupt)
workflow_streamlit = StateGraph(State)

# Add nodes (no greeting node for Streamlit)
workflow_streamlit.add_node("situation", node_situation)
workflow_streamlit.add_node("reflection", node_reflection)
workflow_streamlit.add_node("summary", node_summary)
workflow_streamlit.add_node("evaluation", node_evaluation)
workflow_streamlit.add_node("refine_ask", node_refine_ask)
workflow_streamlit.add_node("hmw_gen", node_hmw_gen)
workflow_streamlit.add_node("cross_silo_ask", node_cross_silo_ask)
workflow_streamlit.add_node("cross_silo_evaluate", node_cross_silo_evaluate)
workflow_streamlit.add_node("final_summary", node_final_summary)
workflow_streamlit.add_node("file_export", node_file_export)
workflow_streamlit.add_node("tools", ToolNode(tools))
# Define entry point routing
workflow_streamlit.add_conditional_edges(
    START,
    route_start,
    {
        "situation": "situation",
        "cross_silo_evaluate": "cross_silo_evaluate",
        "file_export": "file_export",
    },
)

workflow_streamlit.add_edge("reflection", END)
workflow_streamlit.add_edge("summary", "evaluation")
workflow_streamlit.add_edge("refine_ask", END)
workflow_streamlit.add_edge("hmw_gen", "cross_silo_ask")
workflow_streamlit.add_edge("cross_silo_ask", END)
workflow_streamlit.add_edge("final_summary", "file_export")
workflow_streamlit.add_edge("tools", END)

workflow_streamlit.add_conditional_edges(
    "situation",
    route_after_situation,
    {
        "summary": "summary",  # 齊全 -> 下一關
        "reflection": "reflection",  # 缺 -> 追問
    },
)
workflow_streamlit.add_conditional_edges(
    "evaluation",
    route_after_evaluation,
    {
        "hmw_gen": "hmw_gen",  # 齊全 -> 下一關
        "refine_ask": "refine_ask",  # 缺 -> 追問
    },
)
workflow_streamlit.add_conditional_edges(
    "cross_silo_evaluate",
    route_after_cross_silo,
    {"final_summary": "final_summary", END: END},
)


# file_export 決定是否調用工具
workflow_streamlit.add_conditional_edges(
    "file_export",
    tools_condition,
)

graph = workflow_streamlit.compile()
graph_image = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)
