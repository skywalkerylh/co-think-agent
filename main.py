from typing import Annotated, Any, Dict, List

import gradio as gr
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from src.llm import llm
from src.logger import logger
from src.prompt import chatbot_prompt

"""
1. define state 
2. define nodes
3. build graph
4. define chat interface
"""


# Define State
class State(BaseModel):
    """
    Represents the state of our graph.

    Attributes:
        messages: A list of messages. The 'add_messages' reducer automatically
                  appends new messages to the list rather than overwriting.
    """

    messages: Annotated[list, add_messages]


# Define Nodes
def chatbot_node(state: State) -> Dict[str, Any]:
    """
    The main chatbot node that processes the state using the LLM.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the new messages to update the state.
    """
    # Invoke the LLM with the history of messages
    logger.info({"execute_chatbot_node_input": state.messages})
    
    messages = [SystemMessage(content=chatbot_prompt())] + state.messages
    response = llm.invoke(messages)

    logger.info({"execute_chatbot_node_output": response})

    return {"messages": [response]}


# 5. Build Graph
def build_graph():
    """
    Constructs and compiles the StateGraph.
    """
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot_node)

    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # Compile the graph
    return graph_builder.compile()


# 6. Define Chat Interface
def create_chat_interface(graph):
    """
    Creates the Gradio interface wrapper for the graph.
    """

    def chat_function(user_input: str, history: List):
        """
        Processing function for Gradio.
        """
        # Create the initial state with the user's message
        initial_state = {"messages": [{"role": "user", "content": user_input}]}

        # Invoke the graph
        # Note: We pass a simple dict, LangGraph validates it against the State model
        result = graph.invoke(initial_state)

        # Extract the content of the last message (the assistant's reply)
        return result["messages"][-1].content

    return gr.ChatInterface(
        fn=chat_function,
        title="LangGraph Chatbot (Lab 1)",
        description="A simple chatbot built with LangGraph and OpenAI.",
        type="messages",
        chatbot=gr.Chatbot(
            type="messages",
            value=[
                {
                    "role": "assistant",
                    "content": "你好！我是策略 Agent。\n\n您想針對哪個主題開始討論呢？選擇「日常營運困擾」或是「部門事業影響力」？",
                }
            ],
        ),
    )




if __name__ == "__main__":
        # 7. Main Execution
    # Build graph at module level for Gradio CLI compatibility
    graph = build_graph()
    
    app= create_chat_interface(graph)
    print("Building Graph...")
    print("Launching Chat Interface...")
    app.launch(debug=True)