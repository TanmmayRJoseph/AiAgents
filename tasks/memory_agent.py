import os
from typing import List, TypedDict, Union
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.0,
    openai_api_key=os.getenv("OPEN_AI_API_KEY")
)

# Process: Generate trip plan
def process_node(s: AgentState) -> AgentState:
    """
    Process node: Generate trip plan

    The process node uses the travel planner model to generate a detailed
    trip plan based on the user's input. The system prompt is:

    "You are a travel planner who has travelled to all countries. Based on user
    input, make a detailed trip plan."

    The model is invoked with the system prompt and the user's input as the
    conversation history. The response is a detailed trip plan that includes
    flight and accommodation information, as well as any other relevant details.

    The node returns a new state with the updated conversation history that
    includes the trip plan.
    """
    system_prompt = SystemMessage(content="You are a travel planner who has travelled to all countries. Based on user input, make a detailed trip plan.")
    response = llm.invoke([system_prompt] + s["messages"])
    return {"messages": s["messages"] + [response]}

# Summarize: Generate a short summary of the plan
def summarize_node(s: AgentState) -> AgentState:
    """
    Summarize node: Generate a short summary of the plan

    The summarize node takes the detailed trip plan from the process node and
    generates a short summary of the plan. The summary should be 2-3 lines
    and include the important details of the trip.

    The node returns a new state with the updated conversation history that
    includes the summary.
    """
    system_prompt = SystemMessage(content="Summarize the above trip plan in 2-3 lines.")
    response = llm.invoke([system_prompt] + s["messages"])
    return {"messages": s["messages"] + [response]}


graph=StateGraph(AgentState)
graph.add_node("process",process_node)
graph.add_node("summarize",summarize_node)
graph.add_edge(START,"process")
graph.add_edge("process","summarize")
graph.add_edge("summarize",END)
agent=graph.compile()


converstation_history = []

user_input=input("\n👤 You:")

while user_input != "quit":
    # adding user input to the conversation history
    converstation_history.append(HumanMessage(content=user_input))
    
    # invoking the agent and updating the co
    result=agent.invoke({"messages":converstation_history})
    
    converstation_history=result['messages']
    user_input=input("\n👤 You:")
    
    
with open("logging.txt","w") as file:
    file.write("Your converstation history\n")
    
    for message in converstation_history:
        if isinstance(message,HumanMessage):
            file.write(f"👤 You: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"🤖AI: {message.content}\n\n")
    file.write("End of converstation ")
    
    
print ("Your converstation history is saved in logging.txt")