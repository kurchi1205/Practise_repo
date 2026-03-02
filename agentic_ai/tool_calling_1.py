import json
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool, ToolRuntime


@tool
def search_database(runtime: ToolRuntime) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    limit = runtime.state.get("limit", 10)
    query = runtime.state["query"]
    return f"Found {limit} results for '{query}' from database"


@tool("web_search")
def search_web(runtime: ToolRuntime) -> str:
    """Search the web for information about the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    limit = runtime.state.get("limit", 10)
    query = runtime.state["query"]
    return f"Found {limit} results for '{query}' from web"

def build_tool_agent():
    messages = []
    # messages.append(SystemMessage(content="Given the set of tools, choose which tool to use based on the text input."))
    
    SYSTEM_PROMPT = "Given the set of tools, choose which tool to use based on the text input."

    llm = ChatOllama(
            model="mistral",
            validate_model_on_init=True,
            temperature=0.8,
            num_predict=256,
        )
    json_query = {
        "query": "hello, what is the weather today?"
    }
    agent = create_agent(llm, tools=[search_database, search_web], system_prompt=SYSTEM_PROMPT)
    result_1 = agent.invoke({"messages": messages + [HumanMessage(content=json.dumps(json_query))]})
    # result_2 = agent.invoke(messages + [HumanMessage(content="hello, Can you search the database for me?")])
    print(result_1)
    # print(result_2.content)


if __name__ == "__main__":
    build_tool_agent()