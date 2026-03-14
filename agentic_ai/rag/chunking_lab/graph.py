from langgraph.graph import StateGraph, START, END
from state import ChunkingExperimentState
from nodes import ingest_chunks, retrieve, generate_answer, evaluate_answer, report


def build_graph():
    builder = StateGraph(ChunkingExperimentState)

    builder.add_node("ingest_chunks",   ingest_chunks)
    builder.add_node("retrieve",        retrieve)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("report",          report)

    builder.add_edge(START,             "ingest_chunks")
    builder.add_edge("ingest_chunks",   "retrieve")
    builder.add_edge("retrieve",        "generate_answer")
    builder.add_edge("generate_answer", "evaluate_answer")
    builder.add_edge("evaluate_answer", "report")
    builder.add_edge("report",          END)

    return builder.compile()


graph = build_graph()
