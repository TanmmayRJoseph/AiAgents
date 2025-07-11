from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# creating a tool node for the tools that we will use in the agent
@tool
def add(a:int, b:int):
    print("add called")
    return a + b

@tool
def divide(a:int, b:int):
    print("divide called")
    return a / b

@tool 
def subtract(a:int, b:int):
    print("subtract called")
    return a - b

def subtract(a:int, b:int):
    print("subtract called")
    return a - b


tools=[add,subtract,divide]

model=ChatOpenAI(model_name="gpt-3.5-turbo").bind_tools(tools)

def model_call(s:AgentState)->AgentState:
    """The model call function"""
    print("model call called")
    
    system_promt=SystemMessage(content="You are a helpful assistant")
    
    # this line passes the sytem prompt to the model + the users input
    response=model.invoke([system_promt]+s["messages"])
    return {"message":[response]}

def should_continue(s:AgentState):
    """Should the agent continue?"""
    messages=s["messages"]
    
    last_message=messages[-1]
    
    # if no more tools have to be called it will return end
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph=StateGraph(AgentState)
graph.add_node("our-agent",model_call)


# creating tool node
tool_node=ToolNode(tools=tools)
# adding tool node to the graph
graph.add_node("tools",tool_node)

# setting entry point of the graph
graph.set_entry_point("our-agent")

# adding conditional edge
graph.add_conditional_edges(
    "our-agent",
    should_continue,{
        "continue":"tools",
        "end":END
    }
)

graph.add_edge("tools","our-agent")
app=graph.compile()

inputs={"messages":[{"user","Add 34+54"}]}
app.stream(inputs,stream_node="values")