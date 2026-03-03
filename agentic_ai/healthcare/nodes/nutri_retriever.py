import requests
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

import json 

import sys
sys.path.append("../")
from load_model import load_nutri_model
from state import MenuState
llm = load_nutri_model()

def make_api_call(item: str) -> str:
    api_url = 'https://api.api-ninjas.com/v1/nutrition'
    api_key = 'g6chxybq30X8GgKcq7c5JQWbeTI7DNoexNTgR8R6'
    header={"X-Api-Key" : api_key}
    print("Calling API key")
    response = requests.get(api_url, params={"query": item}, headers=header)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error:", response.status_code)
        return 
            
@tool
def get_nutri_info(item: str) -> str:
    """get the nutrition info through api call"""
    return make_api_call(item)

SYSTEM_PROMPT = """You are a nutrition analysis assistant. When given a food item, use the get_nutri_info tool to retrieve its nutritional data and return it as a structured JSON object.

Include all available fields from the API response, such as:
- name, serving_size_g, calories
- fat_total_g, fat_saturated_g, cholesterol_mg
- protein_g, fiber_g, sugar_g, carbohydrates_total_g
- sodium_mg, potassium_mg

Return only the JSON object with no additional explanation or text."""
agent = create_agent(
    llm,
    tools=[get_nutri_info],
    system_prompt=SYSTEM_PROMPT
)

def nutri_retriever(curr_state):
    print(curr_state)
    global llm
    
    
    messages = [{"role": "user", "content": f"Retrieve and return the full nutritional information for: {curr_state['item']}"}]
    response = agent.invoke({"messages": messages})
    print("Response from llm: ", response["messages"][-1].content)
    if curr_state['state'].get("nutrition") is None:
        curr_state['state']['nutrition'] = {}

    if curr_state['state'].get("risk_factor") is None:
        curr_state['state']['risk_factor'] = {}
    curr_state['state']["nutrition"][curr_state['item']] = json.dumps(response["messages"][-1].content)
    messages.append({"role": "assistant", "content": response["messages"][-1].content})
    messages.append({"role": "user", "content": (
        "Based on the nutritional data above, classify the health risk of this food item as one of: "
        '"low", "medium", or "high".\n\n'
        "Use these criteria:\n"
        "- low: nutrient-dense, minimal saturated fat/sodium/sugar, good fiber and protein\n"
        "- medium: moderately processed, some elevated values (e.g. moderate sodium or saturated fat) but not extreme\n"
        "- high: high in saturated fat, sodium (>600mg), added sugars, or calorie-dense with low nutritional value\n\n"
        "Respond with exactly one word: low, medium, or high."
    )})
    
    response = agent.invoke({"messages": messages})
    curr_state['state']['risk_factor'][curr_state['item']] = response
    return




if __name__ == "__main__":
    print(make_api_call("apple"))


