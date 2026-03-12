import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
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
    state["web_searched_content"] = await agent.ainvoke({"messages": [HumanMessage(content=state['question'])]})
    print(state["web_searched_content"]["messages"][-1].content[0]['text'])
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
    vectorstore = get_vector_store(embeddings)
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query, k=2)
    state['retrieved_documents'] = docs
    return state

def ans_generator(state):
    print(state)
    print("Generating answer")
    llm = ChatOllama(
        model="llama3.2:3b",
        validate_model_on_init=True,
        temperature=0.3,
        num_predict=500
    )
    SYSTEM_PROMPT="""Analyze the context of the information provided and answer the question in bullet points."""
    document_content = []
    if state.get('web_searched_content', None):
        document_content = state['web_searched_content']
    else:
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
