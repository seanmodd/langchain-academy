import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

# Ensure OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Optional LangSmith setup
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"

# Define tools
def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.
    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.
    Args:
        a: first int
        b: second int
    """
    return a / b

# Initialize tools and LLM
tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools(tools)

# System message for the assistant
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Define the assistant node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

def main():
    # Build graph
    builder = StateGraph(MessagesState)
    
    # Add nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    # Add edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    
    # Add memory
    memory = MemorySaver()
    react_graph_memory = builder.compile(checkpointer=memory)
    
    # Configuration for thread
    config = {"configurable": {"thread_id": "1"}}
    
    # Test the graph with memory
    print("\nTest 1: Initial calculation")
    messages = [HumanMessage(content="Add 3 and 4.")]
    result = react_graph_memory.invoke({"messages": messages}, config)
    print("\nConversation 1:")
    for m in result['messages']:
        m.pretty_print()
    
    print("\nTest 2: Using memory to reference previous result")
    messages = [HumanMessage(content="Multiply that by 2.")]
    result = react_graph_memory.invoke({"messages": messages}, config)
    print("\nConversation 2:")
    for m in result['messages']:
        m.pretty_print()

if __name__ == "__main__":
    main()
