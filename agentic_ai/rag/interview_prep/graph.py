from langgraph.graph import StateGraph, START, END
from state import question_answer
from nodes import orchestrator, store_web_content, evaluator

workflow_builder = StateGraph(question_answer)

workflow_builder.add_node("orchestrator", orchestrator)
workflow_builder.add_node("store_web_content", store_web_content)
workflow_builder.add_node("evaluator", evaluator)

workflow_builder.add_edge(START, "orchestrator")
workflow_builder.add_edge("orchestrator", "store_web_content")
workflow_builder.add_edge("store_web_content", "evaluator")
workflow_builder.add_edge("evaluator", END)

graph = workflow_builder.compile()
