from typing import List,TypedDict # this is the type hinting
from langchain_core.messages import HumanMessage # this is the message that is sent to the llm
from langchain_openai import ChatOpenAI # this is the llm that is used to generate the response
from langgraph.graph import StateGraph,START,END # this is the graph that is used to store the state of the agent
from dotenv import load_dotenv # this is used to load the environment variables

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

# Initialize the llm
llm=ChatOpenAI(model='gpt-3.5-turbo',temperature=0.5)

# better way to initialize the llm from the environment variable
# llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model='gpt-3.5-turbo', temperature=0.5)

def process_node(s:AgentState)->AgentState:
    '''Process a node in the state graph'''
    response=llm.invoke(s['messages'])
    print(f'\nAI: {response.content}')
    return s

graph=StateGraph(AgentState)

graph.add_node("process",process_node)
graph.add_edge(START,"process")
graph.add_edge("process",END)

agent=graph.compile()


user_input=input("You: ")

'''Below is the code that will ask the user for input 
after every reply from the llm it will terminate'''
agent.invoke({"messages":[HumanMessage(content=user_input)]})


'''Below is the code that will ask the user for input 
after every reply from the llm it will not terminate'''
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")