import os
from datetime import datetime
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Create a basic function to get current time and provide it to graph as tool
@tool
def get_current_time():
    """Get current time"""
    return f"Current time is {datetime.now().strftime('%Y-%m-%d %H:%M')}"

llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    openai_api_key=OPENAI_API_KEY,
    temperature=0.2,
).bind_tools([get_current_time])

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def llm_node(state: MessagesState):
    return {"messages": [llm.invoke(state['messages'])]}

# Start Building the Graph
builder = StateGraph(MessagesState)

builder.add_node("llm_node", llm_node)
builder.add_node("tools", ToolNode([get_current_time]))
builder.add_edge(START, "llm_node")
builder.add_conditional_edges("llm_node", tools_condition)
builder.add_edge("tools", "llm_node")

graph = builder.compile()

# If you are running this in a notebook, you can visualize the graph image using following
# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))

# messages = graph.invoke({"messages": 'Hello'})
messages = graph.invoke({"messages": 'What is time now?'})

print(messages['messages'][-1].content)