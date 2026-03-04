from langgraph.graph import StateGraph, START, END
from state import question_answer
from nodes import relevance_checker, retriever, ans_generator

def route_by_relevance(state: question_answer):
    if state['is_relevant']:
        return "retriever"
    else:
        return "end"

workflow_builder = StateGraph(question_answer)

workflow_builder.add_node("relevance_checker", relevance_checker)
workflow_builder.add_node("retriever", retriever)
workflow_builder.add_node("ans_generator", ans_generator)

workflow_builder.add_edge(START, "relevance_checker")
workflow_builder.add_conditional_edges("relevance_checker", 
                                        route_by_relevance,
                                        {
                                            "retriever": "retriever",
                                            "end": END
                                        }
                                    )
# workflow_builder.add_edge("websearch", "ans_generator")
workflow_builder.add_edge("retriever", "ans_generator")

workflow_builder.add_edge("ans_generator", END)

graph = workflow_builder.compile()

