import json
import operator
import os
from typing import Annotated, Any, Dict, List, Literal, TypedDict
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

load_dotenv()

# --- 1. Tool Definitions ---


@tool
def add(a: float, b: float) -> float:
  """Add two numbers."""
  return a + b


@tool
def multiply(a: float, b: float) -> float:
  """Multiply two numbers."""
  return a * b


@tool
def divide(a: float, b: float) -> float:
  """Divide two numbers."""
  if b == 0:
    return "Error: Division by zero."
  return a / b


@tool
def search_docs(query: str) -> str:
  """Search the knowledge base for AI/ML concepts, LangGraph, embeddings, and Transformers."""
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  vectorstore = FAISS.load_local(
      "faiss_index", embeddings, allow_dangerous_deserialization=True
  )
  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
  docs = retriever.invoke(query)
  return "\n\n".join([doc.page_content for doc in docs])


tools = [add, multiply, divide, search_docs]
tools_by_name = {tool.name: tool for tool in tools}

# --- 2. Model Initialization with Tool Binding ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
model_with_tools = llm.bind_tools(tools)


# --- 3. Graph State Definition ---
class AgentState(TypedDict):
  messages: Annotated[List[AnyMessage], operator.add]
  llm_calls: int


# --- 4. Graph Nodes ---
def llm_call_node(state: AgentState) -> Dict[str, Any]:
  sys_msg = SystemMessage(
      content=(
          "You are a helpful assistant. Use calculator tools for math"
          " operations and search_docs for conceptual queries."
      )
  )
  messages = [sys_msg] + state["messages"]
  response = model_with_tools.invoke(messages)
  return {
      "messages": [response],
      "llm_calls": state.get("llm_calls", 0) + 1,
  }


def tool_node(state: AgentState) -> Dict[str, Any]:
  last_message = state["messages"][-1]
  tool_responses = []

  for tool_call in last_message.tool_calls:
    tool_fn = tools_by_name[tool_call["name"]]
    res = tool_fn.invoke(tool_call["args"])
    tool_responses.append(
        ToolMessage(content=str(res), tool_call_id=tool_call["id"])
    )

  return {"messages": tool_responses}


def should_continue(state: AgentState) -> Literal["tool_node", END]:
  last_message = state["messages"][-1]
  if hasattr(last_message, "tool_calls") and last_message.tool_calls:
    return "tool_node"
  return END


# --- 5. StateGraph Construction & Compilation ---
builder = StateGraph(AgentState)

builder.add_node("llm_call", llm_call_node)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")
builder.add_conditional_edges(
    "llm_call", should_continue, {"tool_node": "tool_node", END: END}
)
builder.add_edge("tool_node", "llm_call")

graph = builder.compile()
