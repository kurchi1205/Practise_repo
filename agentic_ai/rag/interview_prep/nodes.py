import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from pymilvus.model.reranker import CrossEncoderRerankFunction
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from metrics import (
    routing_accuracy_metric,
    context_relevance_metric,
    reranker_effectiveness_metric, rank_improvement_metric,
    faithfulness_metric, completeness_metric, answer_relevance_metric,
)
from dotenv import load_dotenv
load_dotenv("../../../.env")

# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

embeddings = OllamaEmbeddings(model="mistral")
URI = "./milvus_example.db"

ce_rf = CrossEncoderRerankFunction(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu",
)

mcp_client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "tavily-mcp@0.1.4"],
            "env": {"TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]},
        }
    }
)

def get_vector_store():
    return Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )

# ---------------------------------------------------------------------------
# Orchestrator node
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an intelligent RAG orchestrator for an AON interview preparation assistant.

Your goal is to answer the user's question accurately using the available tools.

Follow this process:
1. Call check_relevance to determine if the question is AON interview-related.
2. If relevant: call retrieve_documents to fetch from the knowledge base, then call rerank_documents.
   - If context is insufficient, call retrieve_documents again with a refined query (max 3 times).
   - If still insufficient after retries, call search_web.
3. If not relevant: call search_web directly.
4. Call generate_answer with the question and best available context.
5. Call evaluate_answer — if the answer is poor, refine context or regenerate (max 2 retries).
6. Return the final answer as your last message.

You have a maximum of 12 tool calls. Be decisive."""


async def orchestrator(state):
    # Shared context written to by tools (closures)
    ctx = {
        "retrieved_documents": [],
        "pre_rerank_documents": [],
        "web_searched_content": None,
    }

    @tool
    def check_relevance(question: str) -> str:
        """Check if a question is related to AON interviews.
        Returns 'relevant' or 'not_relevant'."""
        llm = ChatOllama(model="llama3.2:3b", temperature=0, num_predict=10)
        response = llm.invoke([
            SystemMessage(content="Is this question about AON interview prep? Reply only 'relevant' or 'not_relevant'."),
            HumanMessage(content=question),
        ])
        return response.content.strip()

    @tool
    def retrieve_documents(query: str) -> str:
        """Retrieve relevant documents from the knowledge base for a given query."""
        vectorstore = get_vector_store()
        docs = vectorstore.as_retriever().invoke(query, k=5)
        ctx["retrieved_documents"].extend(docs)
        if not docs:
            return "No documents found."
        return "\n\n".join(f"[Doc {i+1}]: {d.page_content}" for i, d in enumerate(docs))

    @tool
    def rerank_documents(query: str) -> str:
        """Rerank previously retrieved documents by relevance to the query.
        Call this after retrieve_documents to improve ordering."""
        docs = ctx["retrieved_documents"]
        if not docs:
            return "No documents to rerank."
        ctx["pre_rerank_documents"] = list(docs)
        passages = [doc.page_content for doc in docs]
        results = ce_rf(query, passages, top_k=min(3, len(passages)))
        reranked = [docs[r.index] for r in results]
        ctx["retrieved_documents"] = reranked
        return "\n\n".join(f"[Doc {i+1}]: {d.page_content}" for i, d in enumerate(reranked))

    @tool
    async def search_web(query: str) -> str:
        """Search the web for information about a query.
        Use when the knowledge base does not have sufficient information."""
        tools = await mcp_client.get_tools()
        llm = ChatOllama(model="llama3.2:3b", temperature=0, num_predict=300)
        agent = create_agent(llm, tools=tools, system_prompt="Search the web and summarise the result.")
        result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
        content = result["messages"][-1].content
        ctx["web_searched_content"] = content
        return content

    @tool
    def generate_answer(question: str, context: str) -> str:
        """Generate a bullet-point answer to the question using the provided context."""
        llm = ChatOllama(model="llama3.2:3b", temperature=0.3, num_predict=500)
        response = llm.invoke([
            SystemMessage(content="Answer the question in bullet points using only the context below."),
            HumanMessage(content=f"Question: {question}\n\nContext:\n{context}"),
        ])
        return response.content

    @tool
    def evaluate_answer(question: str, context: str, answer: str) -> str:
        """Evaluate the quality of a generated answer.
        Returns JSON with keys: accept (bool), reason (str), suggestion (str)."""
        llm = ChatOllama(model="llama3.2:3b", temperature=0, num_predict=200)
        prompt = f"""Evaluate this RAG-generated answer.

Question: {question}
Context used: {context}
Answer: {answer}

Check:
1. Is the answer grounded in the context (no hallucination)?
2. Does it actually address the question?
3. Is it complete?

Return JSON only: {{"accept": true/false, "reason": "...", "suggestion": "..."}}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    llm = ChatOllama(model="llama3.2:3b", temperature=0.3)
    tools = [check_relevance, retrieve_documents, rerank_documents, search_web, generate_answer, evaluate_answer]
    agent = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": [HumanMessage(content=state["question"])]})

    state["generated_answer"] = result["messages"][-1].content
    state["retrieved_documents"] = ctx["retrieved_documents"]
    state["pre_rerank_documents"] = ctx["pre_rerank_documents"]
    state["web_searched_content"] = ctx["web_searched_content"]
    return state

# ---------------------------------------------------------------------------
# Store web content into Milvus (side-effect node, runs in parallel)
# ---------------------------------------------------------------------------

def store_web_content(state):
    web_content = state.get("web_searched_content")
    if not web_content:
        return {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(
        texts=[web_content],
        metadatas=[{
            "source": "websearch",
            "question": state["question"],
            "timestamp": datetime.utcnow().isoformat(),
        }]
    )
    get_vector_store().add_documents(chunks)
    print(f"Stored {len(chunks)} web content chunks into Milvus")
    return {}

# ---------------------------------------------------------------------------
# Evaluator node — RAGAS metrics
# ---------------------------------------------------------------------------

def evaluator(state):
    print("Running evaluation")
    eval_llm = ChatOllama(model="llama3.2:3b", temperature=0, num_predict=200)
    scores = {}

    if state.get("expected_route"):
        actual_route = "websearch" if state.get("web_searched_content") else "retriever"
        r = routing_accuracy_metric(expected_route=state["expected_route"], actual_route=actual_route)
        scores["routing_accuracy"] = r

    context = "\n".join(doc.page_content for doc in (state.get("retrieved_documents") or []))
    answer = state.get("generated_answer", "")
    docs = state.get("retrieved_documents") or []

    if context:
        r = context_relevance_metric(llm=eval_llm, question=state["question"], context=context)
        scores["context_relevance"] = r

    if len(docs) > 1:
        r = reranker_effectiveness_metric(llm=eval_llm, question=state["question"], top_doc=docs[0].page_content, all_docs=context)
        scores["reranker_effectiveness"] = r

    pre_rerank = state.get("pre_rerank_documents") or []
    if len(pre_rerank) > 1 and docs:
        r = rank_improvement_metric(pre_rerank_docs=pre_rerank, post_rerank_docs=docs)
        scores["rank_improvement"] = r

    if answer and context:
        r = faithfulness_metric(llm=eval_llm, question=state["question"], context=context, answer=answer)
        scores["answer_faithfulness"] = r

    if answer and state.get("ground_truth"):
        r = completeness_metric(llm=eval_llm, question=state["question"], ground_truth=state["ground_truth"], answer=answer)
        scores["answer_completeness"] = r

    if answer:
        r = answer_relevance_metric(llm=eval_llm, question=state["question"], answer=answer)
        scores["answer_relevance"] = r

    print("Scores:", scores)
    state["scores"] = scores
    return state
