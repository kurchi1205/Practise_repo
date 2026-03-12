from langgraph.graph import StateGraph, START, END
from state import question_answer
from nodes import relevance_checker, retriever, ans_generator, websearch

def route_by_relevance(state: question_answer):
    if state['is_relevant'] == 'True':
        return "retriever"
    else:
        return "web_search"

workflow_builder = StateGraph(question_answer)

workflow_builder.add_node("relevance_checker", relevance_checker)
workflow_builder.add_node("retriever", retriever)
workflow_builder.add_node("ans_generator", ans_generator)
workflow_builder.add_node("websearch", websearch)

workflow_builder.add_edge(START, "relevance_checker")
workflow_builder.add_conditional_edges("relevance_checker", 
                                        route_by_relevance,
                                        {
                                            "retriever": "retriever",
                                            "web_search": "websearch"
                                        }
                                    )
workflow_builder.add_edge("websearch", "ans_generator")
workflow_builder.add_edge("retriever", "ans_generator")

workflow_builder.add_edge("ans_generator", END)

graph = workflow_builder.compile()

