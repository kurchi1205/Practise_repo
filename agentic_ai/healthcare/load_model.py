from langchain_ollama import ChatOllama


def load_nutri_model():
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        max_tokens=50,
    )
    return llm
