from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from state import MenuState
from nodes.nutri_retriever import nutri_retriever
from nodes.voter import voter


def nutri_check(state: MenuState):
    return [Send("nutri_retriever", {"state": state, "item": item}) for item in state['menu']]

def gate(state: MenuState):
    print(state)
    print("Passed through gate")

workflow = StateGraph(MenuState)

# Define the two nodes we will cycle between
workflow.add_node("nutri_retriever", nutri_retriever)
workflow.add_node("voter", voter)
workflow.add_node("gate", gate)

workflow.add_edge(START, "gate")
workflow.add_conditional_edges("gate", nutri_check, ["voter"])
workflow.add_edge("voter", END)


# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()



