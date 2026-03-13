import os
from langchain_ollama import ChatOllama
from metrics import (
    context_relevance_metric, reranker_metric,
    faithfulness_metric, completeness_metric, query_reformulation_metric,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from pymilvus.model.reranker import CrossEncoderRerankFunction

# from langchain_tavily import TavilySearch
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
load_dotenv()

# # Initialize Tavily Search Tool
# tavily_search_tool = TavilySearch(
#     max_results=5,
#     topic="general",
# )

ce_rf = CrossEncoderRerankFunction(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Specify the model name. Defaults to an emtpy string.
    device="cpu" # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)

class RelevanceQuestion(BaseModel):
    """Check if question is relevant or not."""
    is_relevant: bool

def relevance_checker(state):
    print("Checking relevance")
    llm = ChatOllama(
        model="llama3.2:3b",
        validate_model_on_init=True,
        temperature=0,
        num_predict=10
    )
    
    SYSTEM_PROMPT="""You are a relevance checker. Your only job is to determine if the user's question is related to a
    +n AON job interview — such as interview preparation, AON interview questions, AON company culture, or AON hiring process.
     Strictly return just True or False"""
    agent = create_agent(
        llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )
    question = state['question']
    answer = agent.invoke({"messages": [HumanMessage(content=question)]})
    state['is_relevant'] = answer["messages"][-1].content
    return state


client = MultiServerMCPClient(
        {
            "tavily": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "tavily-mcp@0.1.4"],
                "env": {"TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]},
            }
        }
    )
async def websearch(state):
    tools = await client.get_tools()
    llm = ChatOllama(
        model="llama3.2:3b",
        validate_model_on_init=True,
        temperature=0,
        num_predict=100
    )
    
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt="Do a web search for the given question answer the question",
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=state['question'])]})
    state["web_searched_content"] = result["messages"][-1].content
    return state

embeddings = OllamaEmbeddings(model="mistral")
URI = "./milvus_example.db"

def get_vector_store(embeddings):
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )
    return vector_store


def retriever(state):
    query = state['question']
    if state.get('new_question'):
        if state['new_question'] != 'no_question':    
            query = state['new_question']
            state["past_questions"] += state['new_question']
    else:
        state["past_questions"] = state['question']
    vectorstore = get_vector_store(embeddings)
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query, k=10)
    if state.get('retrieved_documents'):
        state['retrieved_documents'] += docs
    else:
        state['retrieved_documents'] = docs
    state['retrieval_count'] = state.get('retrieval_count', 0) + 1
    return state


def reranker(state):
    docs = state.get('retrieved_documents') or []
    web_content = state.get('web_searched_content')

    if not docs and not web_content:
        return state

    query = state['question']

    # Build unified passage list; track where web content sits
    passages = [doc.page_content for doc in docs]
    web_idx = None
    if web_content:
        web_idx = len(passages)
        passages.append(web_content)

    results = ce_rf(query, passages, top_k=len(passages))

    web_doc = Document(page_content=web_content, metadata={"source": "websearch"}) if web_content else None
    reranked_docs = []
    for r in results:
        if r.index == web_idx:
            reranked_docs.append(web_doc)
        else:
            reranked_docs.append(docs[r.index])
    state['retrieved_documents'] = reranked_docs
    return state

def auto_corrector(state):
    llm = ChatOllama(
        model="llama3.2:3b",
        validate_model_on_init=True,
        temperature=0.3,
        num_predict=100
    )
    SYSTEM_PROMPT="""You are auto corrector. Your job is to analyze the retrieved documents, past questions and formulate a new query to answer the user's question. \n
    Either form a new question based on the context or return ```no_question``` if context is enough. Do not repeat same query. return 
    Never repeat past questions and return no_question after 2 - 3 past questions. Give either new question or no_question.
    """
    agent = create_agent(
        llm,
        system_prompt=SYSTEM_PROMPT,
    )
    document_content = "\n".join(doc.page_content for doc in state['retrieved_documents'])
    past_queries = state['past_questions']
    final_user_content = "Past questions: " + past_queries + "\n" + "Question: " + state['question'] + "\n" + "Context: " + document_content
    answer = agent.invoke({"messages": [HumanMessage(content=final_user_content)]})
    state['new_question'] = answer["messages"][-1].content
    return state

def ans_generator(state):
    print("Generating answer")
    llm = ChatOllama(
        model="llama3.2:3b",
        validate_model_on_init=True,
        temperature=0.3,
        num_predict=500
    )
    SYSTEM_PROMPT="""Analyze the context of the information provided and answer the question in bullet points."""
    document_content = "\n".join(doc.page_content for doc in state['retrieved_documents'])
    messages = [
        HumanMessage(content="Question: " + state['question'] + "Context" + document_content),
    ]
    agent = create_agent(
        llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )
    answer = agent.invoke({"messages": messages})
    state['generated_answer'] = answer["messages"][-1].content
    return state

def evaluator(state):
    print("Running evaluation")
    eval_llm = ChatOllama(model="llama3.2:3b", temperature=0, num_predict=200)
    scores = {}

    context = "\n".join(doc.page_content for doc in (state.get("retrieved_documents") or []))
    answer = state.get("generated_answer", "")
    docs = state.get("retrieved_documents") or []

    # Context relevance — only on retriever path
    if context:
        r = context_relevance_metric.score(
            llm=eval_llm, question=state["question"], context=context
        )
        scores["context_relevance"] = r.value

    # Reranker effectiveness — only when multiple docs were retrieved
    if len(docs) > 1:
        r = reranker_metric.score(
            llm=eval_llm,
            question=state["question"],
            top_doc=docs[0].page_content,
            all_docs=context,
        )
        scores["reranker_effectiveness"] = r.value

    # Answer faithfulness
    if answer and context:
        r = faithfulness_metric.score(
            llm=eval_llm, question=state["question"], context=context, answer=answer
        )
        scores["answer_faithfulness"] = r.value

    # Answer completeness — only when ground truth was provided
    if answer and state.get("ground_truth"):
        r = completeness_metric.score(
            llm=eval_llm,
            question=state["question"],
            ground_truth=state["ground_truth"],
            answer=answer,
        )
        scores["answer_completeness"] = r.value

    # Query reformulation — only when auto_corrector fired
    new_q = state.get("new_question")
    if new_q and new_q != "no_question":
        r = query_reformulation_metric.score(
            llm=eval_llm,
            question=state["question"],
            new_question=new_q,
            context=context,
        )
        scores["query_reformulation_quality"] = r.value

    print("Scores:", scores)
    state["scores"] = scores
    return state
