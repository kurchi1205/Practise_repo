from langgraph.graph import StateGraph, START, END
from state import question_answer
from nodes import relevance_checker, retriever, reranker, ans_generator, websearch, auto_corrector, evaluator

def route_by_relevance(state: question_answer):
    if state['is_relevant'] == 'True':
        return "retriever"
    else:
        return "websearch"
    
def route_by_autocorrection(state: question_answer):
    if state.get('retrieval_count', 0) > 3:
        return "reranker"
    if state['new_question'] == 'no_question':
        return "reranker"
    return "websearch"

workflow_builder = StateGraph(question_answer)

workflow_builder.add_node("relevance_checker", relevance_checker)
workflow_builder.add_node("retriever", retriever)
workflow_builder.add_node("reranker", reranker)
workflow_builder.add_node("auto_corrector", auto_corrector)
workflow_builder.add_node("ans_generator", ans_generator)
workflow_builder.add_node("websearch", websearch)
workflow_builder.add_node("evaluator", evaluator)

workflow_builder.add_edge(START, "relevance_checker")
workflow_builder.add_conditional_edges("relevance_checker", 
                                        route_by_relevance,
                                        {
                                            "retriever": "retriever",
                                            "websearch": "websearch"
                                        }
                                    )
workflow_builder.add_edge("retriever", "auto_corrector")
workflow_builder.add_conditional_edges("auto_corrector",
                                        route_by_autocorrection,
                                        {
                                            "websearch": "websearch",
                                            "reranker": "reranker"
                                        }
                                    )
workflow_builder.add_edge("websearch", "reranker")
workflow_builder.add_edge("reranker", "ans_generator")
# workflow_builder.add_edge("ans_generator", "evaluator")
workflow_builder.add_edge("ans_generator", END)

graph = workflow_builder.compile()

