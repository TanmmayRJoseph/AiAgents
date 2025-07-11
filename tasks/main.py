import os
from typing import List, TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# ---- Define the State ----
class AgentState(TypedDict):
    messages: List[BaseMessage]

# ---- Initialize LLM ----
llm = ChatOpenAI(openai_api_key=os.getenv("OPEN_AI_KEY"), model='gpt-3.5-turbo', temperature=0.5)

# ---- Node 1: Validation ----
def validate_node(state: AgentState) -> AgentState:
    return state  # Just pass state forward

# ---- Conditional Edge Function (Router) ----
def is_valid_input(state: AgentState) -> Literal[True, False]:
    """
    Determines if the last message in the agent's state is valid.

    This function checks the content of the last message in the agent's
    state to determine if it meets a minimum length requirement. If the
    message is too short, it prints a warning message and returns False,
    indicating that the conversation should end. Otherwise, it returns
    True, allowing the conversation to continue.

    Args:
        state (AgentState): The current state of the agent containing a list
                            of message objects.

    Returns:
        Literal[True, False]: True if the last message is valid, False if it
                              is too short.
    """

    last_msg = state["messages"][-1].content.strip()
    if len(last_msg) < 5:
        print("⚠️  Message too short. Ending conversation.")
        return False
    return True

# ---- Node 2: Process ----
def process_node(state: AgentState) -> AgentState:
    """
    Process node: Generate a response to the user's input.

    This node takes the current state of the agent (including the user's
    input) and uses the LLM to generate a response. The response is then
    appended to the state's message list and returned.

    Args:
        state (AgentState): The current state of the agent containing a list
                            of message objects.

    Returns:
        AgentState: The updated state with the AI's response appended to the
                    message list.
    """

    response = llm.invoke(state["messages"])
    print("\n🧠 AI:", response.content)
    print("------------------------------------------------")
    return {
        "messages": state["messages"] + [response]
    }

# ---- Build Graph ----
graph = StateGraph(AgentState)

graph.add_node("validate", validate_node)
graph.add_node("process", process_node)

graph.set_entry_point("validate")

# ✅ Use routing function here
graph.add_conditional_edges(
    "validate",
    is_valid_input,  # <- must return True/False
    {
        True: "process",
        False: END
    }
)

graph.add_edge("process", END)

# ---- Compile Agent ----
agent = graph.compile()

# ---- Conversation Loop ----
print("✨ Welcome to the Motivational Quote Agent ✨")
print("Type your mood or issue (e.g., 'I'm feeling stuck') or type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() == "exit":
        print("👋 Goodbye!")
        break

    initial_state = {
        "messages": [HumanMessage(content=user_input)]
    }

    _ = agent.invoke(initial_state)
