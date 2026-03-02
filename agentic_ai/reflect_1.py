from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def load_model():
    llm =  ChatOllama(
            model="mistral",
            validate_model_on_init=True,
            temperature=0.8,
            num_predict=256,
        )
    return llm


COSING_TASK = """
Your task is to create a Python function named `calculate_factorial`.
_
This function should do the following:
1. Accept a single integer n as input.
2. Calculate its factorial (n!).
3. Include a clear docstring explaining what the function does.
4. Handle edge cases: The factorial of 0 is 1.
5. Handle invalid input: Raise a ValueError if the input is a
negative number.
"""
def reflect():
    llm = load_model()
    messages = []
    messages.append(HumanMessage(content=COSING_TASK))
    max_iterations = 3

    for iter in range(max_iterations):
    # Generatr response
        response = llm.invoke(messages)
        messages.append(response)


        ## Reflector
        system_prompt = "You are a coding agent, check whether the current code matches the user task. Answer in Yes or No"
        messages.append(SystemMessage(content=system_prompt))
        response = llm.invoke(messages)
        if "Yes" in response.content:
            break

        messages.append(HumanMessage(content=response.content))

    return messages[-1].content


if __name__ == "__main__":
    print(reflect())
