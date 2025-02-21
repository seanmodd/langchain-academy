from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
result = llm.invoke("what''s upppppp")
print("LLM Response:", result)
