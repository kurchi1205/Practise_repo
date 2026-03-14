from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_milvus import Milvus
from metrics import context_relevance_metric, faithfulness_metric, answer_relevance_metric
from config import LLM_MODEL, EMBEDDING_MODEL, MILVUS_URI, RETRIEVAL_K


def get_vector_store(collection_name: str):
    """Each strategy gets its own Milvus collection to prevent cross-contamination."""
    return Milvus(
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        connection_args={"uri": MILVUS_URI},
        collection_name=collection_name,
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )


# ------------------------------------------------------------------
# Node 1 — ingest_chunks
# Embeds chunks into a strategy-specific Milvus collection.
# Records structural stats: num_chunks, avg_chunk_length.
# ------------------------------------------------------------------
def ingest_chunks(state: dict) -> dict:
    chunks = state["chunks"]
    store = get_vector_store(state["strategy_name"])
    docs = [
        Document(page_content=c, metadata={"strategy": state["strategy_name"]})
        for c in chunks
    ]
    store.add_documents(docs)
    print(f"[ingest] strategy={state['strategy_name']}  chunks={len(chunks)}")
    return {
        **state,
        "num_chunks": len(chunks),
        "avg_chunk_length": round(sum(len(c) for c in chunks) / max(len(chunks), 1), 1),
    }


# ------------------------------------------------------------------
# Node 2 — retrieve
# Similarity search from the strategy's Milvus collection.
# ------------------------------------------------------------------
def retrieve(state: dict) -> dict:
    store = get_vector_store(state["strategy_name"])
    retriever = store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    docs = retriever.invoke(state["question"])
    print(f"[retrieve] retrieved {len(docs)} docs")
    return {**state, "retrieved_docs": docs}


# ------------------------------------------------------------------
# Node 3 — generate_answer
# Generates answer using llama3.2:3b with retrieved context.
# ------------------------------------------------------------------
def generate_answer(state: dict) -> dict:
    context = "\n\n".join(doc.page_content for doc in state["retrieved_docs"])
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3, num_predict=400)
    response = llm.invoke([
        SystemMessage(content="Answer the question using only the provided context. Be concise."),
        HumanMessage(content=f"Question: {state['question']}\n\nContext:\n{context}"),
    ])
    answer = response.content.strip()
    print(f"[generate] answer length={len(answer)} chars")
    return {**state, "generated_answer": answer}


# ------------------------------------------------------------------
# Node 4 — evaluate
# Calls metrics from metrics.py — same pattern as interview_prep/metrics.py.
# ------------------------------------------------------------------
def evaluate_answer(state: dict) -> dict:
    print("[evaluate] running metrics...")

    llm = ChatOllama(model=LLM_MODEL, temperature=0, num_predict=300)
    context = "\n\n".join(doc.page_content for doc in state["retrieved_docs"])
    answer = state["generated_answer"]
    question = state["question"]

    scores = {
        "context_relevance": context_relevance_metric(
            question=question, context=context, llm=llm
        ),
        "faithfulness": faithfulness_metric(
            question=question, context=context, answer=answer, llm=llm
        ),
        "answer_relevance": answer_relevance_metric(
            question=question, answer=answer, llm=llm
        ),
    }

    print(f"[evaluate] scores={scores}")
    return {**state, "scores": scores}


# ------------------------------------------------------------------
# Node 5 — report
# Prints a formatted summary for this (strategy, question) run.
# ------------------------------------------------------------------
def report(state: dict) -> dict:
    print(f"\n{'='*65}")
    print(f"  Strategy : {state['strategy_name']}")
    print(f"  Question : {state['question']}")
    print(f"  Chunks   : {state['num_chunks']}  |  Avg length: {state['avg_chunk_length']} chars")
    print(f"  Answer   : {state['generated_answer'][:120]}...")
    print(f"  Scores   : {state['scores']}")
    print(f"{'='*65}\n")
    return state
