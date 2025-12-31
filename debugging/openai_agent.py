# ### Debugging LangGraph Applications With LangSmith

# from typing import Annotated
# from typing_extensions import TypedDict
# from langchain_openai import ChatOpenAI
# from langgraph.graph import START, END 
# from langgraph.graph.state import StateGraph
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode
# from langchain_core.tools import tool
# from langchain_core.messages import BaseMessage
# import os
# from dotenv import load_dotenv 
# load_dotenv()
# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
# os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

# # STEP 1:
# # Define state class with "messages" variable using annotated with basemessage and reducer.
# # Reducer "add_messages" will keep appending the messages
# # This class will keep adding messages to the list of messages as user keep querying the models (like with chatgpt)
# class State(TypedDict):
#     messages:Annotated[list[BaseMessage],add_messages]

# # STEP 2: Initialize the model with temp=0. ChatOpenAI use "gpt-4o" model by default
# model=ChatOpenAI(temperature=0)

# # STEP 3: Create a function for a workflow using StateGraph. graph_workflow is just name of the graph
# def make_default_graph():
#     graph_workflow=StateGraph(State) 

# # define the node functionality
#     def call_model(state):
#         return {"messages":[model.invoke(state['messages'])]}

# # another tool using function 
# def make_alternative_graph():
#     """ Make a tool-calling agent"""

#     @tool
#     def add(a:float,b:float):
#         return a + b 
    
#     tool_node=ToolNode([add])
#     model_with_tools=model.bind_tools([add])

#     def call_model(state):
#         return {"messages":[model_with_tools.invoke(state["messages"])]}

#     def should_continue(state:State):
#         if state["messages"][-1].tool_calls:
#             return "tools"
#         else:
#             return END
        
# # now, adding the nodes and edges to the workflow 
#     graph_workflow.add_node("agent", call_model)
#     graph_workflow.add_node("tools",tool_node)
    
#     graph_workflow.add_edge("tools", "agent")
#     graph_workflow.add_edge(START,"agent")
#     graph_workflow.add_conditional_edge("agent",should_continue)
 
# # next, create agent using the above graph workflow and compile.agent is nothing but a graph
#     agent=graph_workflow.compile()
#     return agent

# # calling this agent 
# agent=make_default_graph()

