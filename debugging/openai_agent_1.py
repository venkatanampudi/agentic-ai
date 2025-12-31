from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- STATE ----------
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---------- MODEL ----------
model = ChatOpenAI(temperature=0)

# ---------- GRAPH ----------
def make_graph():
    graph = StateGraph(State)

    @tool
    def add(a: float, b: float):
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])

    def call_model(state: State):
        return {
            "messages": [model_with_tools.invoke(state["messages"])]
        }

    def route(state: State):
        last_msg = state["messages"][-1]
        if last_msg.tool_calls:
            return "tools"
        return END

    # Nodes
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    # Edges
    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "agent",
        route,
        {
            "tools": "tools",
            END: END
        }
    )

    return graph.compile()

# ---------- AGENT ----------
agent = make_graph()
