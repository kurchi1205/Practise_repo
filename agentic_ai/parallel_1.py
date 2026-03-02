from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


def load_model():
    llm =  ChatOllama(
            model="mistral",
            validate_model_on_init=True,
            temperature=0.8,
            num_predict=256,
        )
    return llm


def create_parallel_agents(text_input):
    llm = load_model()
    summarizer = ChatPromptTemplate.from_template(
        "Summarize the following text: {Text}"
    )

    topic_extractor = ChatPromptTemplate.from_template(
        "Extract the topic of the following text: {Text}. Give the output as a JSON object with 'topic' as key."
    )

    summarizer_chain = summarizer | llm | StrOutputParser()
    topic_extractor_chain = topic_extractor | llm | StrOutputParser()


    parallel_executor = RunnableParallel(
    {
        "Summary": summarizer_chain,
        "Topic": topic_extractor_chain,
        "Text": RunnablePassthrough(),
    }
    )

    result_accum_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "From the summary and topic, give the result in a paragraph, Summary: {Summary}, Topic: {Topic}"),
        ("human", "Text: {Text}"),
        ]
    )

    runnable = parallel_executor | result_accum_prompt | llm | StrOutputParser()
    result = runnable.invoke({"Text": text_input})

    return result


if __name__ == "__main__":
    print(create_parallel_agents('''The prerequisites for this implementation include the installation of the requisite
Python packages, such as langchain, langchain-community, and a model provider
library like langchain-openai. Furthermore, a valid API key for the chosen language
model must be configured in the local environment for authentication.'''))