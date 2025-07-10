from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()
print("-------------ReAct agent example----------------")
# Annotated: provides additional context without affecting the type itself
# Sequence: To automatically handle the state updates for sequence such as by adding new messages to a chat history
# ToolMessage: Passes data back to the LLM after it calls a tool such as the content and the tool_call_id
# SystemMessage: Message for providing instruction to the llm
# BaseMessage: the foundational class for all message types in langgraph

'''
# add_message: Basically its a reducer function which controls how updates from nodes are combined with the existing state
# tells us how to merege new data into the current state
# without a reducer , updates would have replaced the existing value entirely
'''


class AgentState(TypedDict):
    """The state of the agent"""
    messages:Annotated[Sequence[BaseMessage], add_messages]
    
@tool
def add(a:int, b:int):
    """
    tool should have a docstring describing what it does
    Add two numbers"""
    print("add called")
    return a + b

@tool
def subtract(a:int, b:int):
    """
    tool should have a docstring describing what it does
    Subtract two numbers"""
    print("subtract called")
    return a - b

@tool
def multiply(a:int, b:int):
    """
    tool should have a docstring describing what it does
    Multiply two numbers"""
    print("multiply called")
    return a * b


tools=[add,subtract,multiply]

# to use the tools created we use the bind_tools function
model=ChatOpenAI(model_name="gpt-3.5-turbo").bind_tools(tools) #now the model will hace acess to all tools created 


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

# adding a edge that goes back to the entry point to make the circular loop
graph.add_edge("tools","our-agent")
app=graph.compile()

def print_stream(stream):
    """
    Print the messages in the stream.

    Given a stream of states, this prints the final message in each state
    to the console. If the message is a tuple, it will print the tuple
    directly. Otherwise, it will call `pretty_print` on the message.

    Args:
        stream: an iterable of states
    """
    for s in stream:
        message=s['messages'][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()



inputs={"messages":[{"user","Add 34+54"}]}
print_stream(app.stream(inputs,stream_node="values"))