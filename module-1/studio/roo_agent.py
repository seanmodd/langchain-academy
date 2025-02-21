import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
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
    """Adds a and 
    b.
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
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

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
    
    # Compile graph
    react_graph = builder.compile()
    
    # Test the graph
    print("\nTesting arithmetic operations:")
    messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
    result = react_graph.invoke({"messages": messages})
    
    print("\nConversation:")
    for m in result['messages']:
        m.pretty_print()

if __name__ == "__main__":
    main()
