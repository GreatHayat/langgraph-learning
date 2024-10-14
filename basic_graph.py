import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    openai_api_key=OPENAI_API_KEY,
    temperature=0.2,
)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def llm_node(state: MessagesState):
    return {"messages": [llm.invoke(state['messages'])]}

# Start Building the Graph
builder = StateGraph(MessagesState)

builder.add_node("llm_node", llm_node)
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)

graph = builder.compile()

# If you are running this in a notebook, you can visualize the graph image using following
# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))

messages = graph.invoke({"messages": 'Hello'})

for message in messages['messages']:
    print(message.content)