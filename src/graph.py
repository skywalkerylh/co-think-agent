from langgraph.graph import END, START, StateGraph

from src.node import (
    node_cross_silo,
    node_export,
    node_integration,
    node_proposal,
    node_reflection,
    node_situation,
)
from src.edges import check_quality
from src.state import AgentState


workflow = StateGraph(AgentState)

workflow.add_node("situation", node_situation)
workflow.add_node("reflection", node_reflection)
workflow.add_node("cross_silo", node_cross_silo)
workflow.add_node("integration", node_integration)
workflow.add_node("proposal", node_proposal)
workflow.add_node("export", node_export)

workflow.add_edge(START, "situation")
workflow.add_edge("situation", "reflection")
workflow.add_edge("cross_silo", "integration")
workflow.add_edge("integration", "proposal")
workflow.add_edge("proposal", "export")
workflow.add_edge("export", END)

workflow.add_conditional_edges(
    "reflection", check_quality, {"cross_silo": "cross_silo", "situation": "situation"}
)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
