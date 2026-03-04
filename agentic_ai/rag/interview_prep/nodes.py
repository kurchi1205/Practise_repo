from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
# from langchain_tavily import TavilySearch
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain.agents import create_agent
import json

# # Initialize Tavily Search Tool
# tavily_search_tool = TavilySearch(
#     max_results=5,
#     topic="general",
# )

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
    SYSTEM_PROMPT="""You are a relevance checker, From the user question, check whether it is related AON interview or not.Strictly return just True or False"""
    agent = create_agent(
        llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )
    question = state['question']
    answer = agent.invoke({"messages": [HumanMessage(content=question)]})
    state['is_relevant'] = bool(answer["messages"][-1].content)
    return state


# client = MultiServerMCPClient(
#         {
#             "web_search": {
#                 "transport": "stdio",  # Local subprocess communication
#                 "command": "python",
#                 # Absolute path to your math_server.py file
#                 "args": ["/path/to/math_server.py"],
#             },
#         }
#     )

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
    vectorstore = get_vector_store(embeddings)
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query, k=2)
    state['retrieved_documents'] = docs
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
    document_content = []
    for doc in state['retrieved_documents']:
        document_content.append(doc.page_content)
    document_content = "\n".join(document_content)
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
