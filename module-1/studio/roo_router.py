import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display

# Ensure OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Define tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b

# Initialize LLM and bind tool
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools([multiply])

# Define the tool calling node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def main():
    # Build graph
    builder = StateGraph(MessagesState)
    
    # Add nodes
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode([multiply]))
    
    # Add edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition,
    )
    builder.add_edge("tools", END)
    
    # Compile graph
    graph = builder.compile()
    
    # Test the graph
    print("\nTesting the router:")
    messages = [HumanMessage(content="Hello, what is 2 multiplied by 2?")]
    result = graph.invoke({"messages": messages})
    
    print("\nConversation:")
    for m in result['messages']:
        m.pretty_print()

if __name__ == "__main__":
    main()
