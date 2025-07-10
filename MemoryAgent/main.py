import os
from typing import List, TypedDict, Union
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

class AgentState(TypedDict):
    """ Represents the state of the agent.
    Basically both are data Structures 
    Where both Human and AI messages are stored in a list"""
    
    messages: List[Union[HumanMessage,AIMessage]]


llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.0,
    openai_api_key=os.getenv("OPEN_AI_API_KEY")
)

def process(s:AgentState)->AgentState:
    '''This node will solve the request you input
    here s is the state of the agent
    '''
    response =llm.invoke(s["messages"])
    s["messages"].append(AIMessage(content=response.content))
    print(f'\n🤖AI: {response.content}')
    
    # below line will print the current state of the agent Good for debugging
    print("CURRENT STATE",s["messages"])
    return s

graph=StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent=graph.compile()

converstation_history = []

user_input=input("\n👤 You:")

while user_input!="exit":
    # below line will add the user input to the converstation history
    converstation_history.append(HumanMessage(content=user_input))
    # below line will invoke the agent and get the response and update the converstation history
    result=agent.invoke({"messages":converstation_history})
    
    # below line will print the response
    # print(result['messages'])
    
    # below line will update the converstation history
    converstation_history=result['messages']
    user_input=input("\n👤 You:")


# ❌ There is 2 main problem here 
# 1. The agent is not able to remember the converstation history only when the code is executed 
# 2. if we terminate the code and then ask the same question the agent will not remember the converstation history
# * Solution: to that is either use a Vector database which is more robust or use a txt file to store the converstation history

with open("logging.txt","w") as file:
    file.write("Your converstation history\n")
    
    for message in converstation_history:
        if isinstance(message,HumanMessage):
            file.write(f"👤 You: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"🤖AI: {message.content}\n\n")
    file.write("End of converstation ")
    
    
print ("Your converstation history is saved in logging.txt")