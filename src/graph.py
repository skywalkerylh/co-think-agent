from langgraph.graph import END, START, StateGraph

# from langgraph.memory import MemorySaver
from src.node import (
    # node_cross_silo,
    # node_export,
    # node_integration,
    # node_proposal,
    node_reflection,
    node_situation,
)
from src.state import AgentState

workflow = StateGraph(AgentState)

# add nodes
workflow.add_node("situation", node_situation)
workflow.add_node("reflection", node_reflection)
# workflow.add_node("cross_silo", node_cross_silo)
# workflow.add_node("integration", node_integration)
# workflow.add_node("proposal", node_proposal)
# workflow.add_node("export", node_export)

# add edges
# 流程調整：START -> reflection -> situation -> END
# 這樣每次使用者輸入後，先由 reflection 檢查，再由 situation 生成回應，然後等待使用者再次輸入
workflow.add_edge(START, "reflection")
workflow.add_edge("reflection", "situation")
workflow.add_edge("situation", END)

# 移除舊的 conditional edges，因為現在是線性流程，透過使用者輸入來循環
# workflow.add_conditional_edges(
#     "reflection",
#     check_quality,
#     {"end": END, "situation": "situation"}
# )
# compile and draw
# memory = MemorySaver()
app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
