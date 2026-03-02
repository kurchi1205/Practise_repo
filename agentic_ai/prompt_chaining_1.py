from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def load_model():
    llm =  ChatOllama(
            model="mistral",
            validate_model_on_init=True,
            temperature=0.8,
            num_predict=256,
        )
    return llm



def create_prompt_chain(text_input):
    llm = load_model()
    prompt_extract = ChatPromptTemplate.from_template(
        "Extract the keywords from the {text_input}"
    )
    
    prompt_transform = ChatPromptTemplate.from_template(
        "Transform the following specifications into a JSON object with\
'extracted' as keys:\n\n{specifications}"
    )
    specifications = prompt_extract | llm | StrOutputParser()
    runnable = ({"specifications": specifications}| prompt_transform | llm | StrOutputParser())
    result = runnable.invoke({"text_input": text_input})

    return result

if __name__ == "__main__":
    print(create_prompt_chain("hello, LLM, tooling, chaining"))