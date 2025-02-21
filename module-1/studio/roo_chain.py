import os
from typing import Annotated
from typing_extensions import TypedDict
from pprint import pprint
from IPython.display import Image, display
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState

# Ensure OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Initialize chat model
llm = ChatOpenAI(model="gpt-4")

# Define a simple tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b

# Bind tool to the model
llm_with_tools = llm.bind_tools([multiply])

# Create messages state class
class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    pass

# Define the tool calling node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def main():
    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_edge(START, "tool_calling_llm")
    builder.add_edge("tool_calling_llm", END)
    graph = builder.compile()

    # Test cases
    print("\nTest 1: Simple greeting")
    messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
    for m in messages['messages']:
        m.pretty_print()

    print("\nTest 2: Tool usage")
    messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})
    for m in messages['messages']:
        m.pretty_print()

if __name__ == "__main__":
    main()
