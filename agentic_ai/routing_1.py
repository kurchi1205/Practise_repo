from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

def get_model():
    llm = ChatOllama(
            model="mistral",
            validate_model_on_init=True,
            temperature=0.8,
            num_predict=256,
    )
    return llm


def booking_agent(text_input):
    print("Booking is done for the following text:", text_input)
    return "Booking is done for the following text: " + text_input


def info_agent(text_input):
    print("Info is done for the following text:", text_input)
    return "Info is done for the following text: " + text_input


def other_agent(text_input):
    print("Other is done for the following text:", text_input)
    return "Other is done for the following text: " + text_input

def routing_to_diff_agents(text_input):
    llm = get_model()
    input_prompt = ChatPromptTemplate.from_template(
        "Classify the intent of the following text: {text_input}. "
        "Reply with ONLY one word: booking, info, or other."
    )

    extracted_intent = input_prompt | llm | StrOutputParser()

    runnable_branch = RunnableBranch(
        (lambda x: x["intent"].lower().strip() == "booking", RunnableLambda(lambda _: booking_agent(text_input))),
        (lambda x: x["intent"].lower().strip() == "info", RunnableLambda(lambda _: info_agent(text_input))),
        RunnableLambda(lambda _: other_agent(text_input)),
    )

    coordinator = {"intent": extracted_intent} | runnable_branch
    result = coordinator.invoke({"text_input": text_input})

    return result


if __name__=="__main__":
    print(routing_to_diff_agents("I want to book a flight"))