from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus



def get_doc_loader(file_paths):
    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(
            path,
            mode="single",
        )
        docs = loader.load()
        print("Length of docs: ", len(docs))
    all_docs.append(docs)
    return all_docs


def split_text(all_docs):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=100,
        chunk_overlap=0,
    )
    all_texts = []
    for doc in all_docs:
        texts = text_splitter.split_text(doc[0].page_content)
        all_texts.extend(texts)
    return all_texts

embeddings = OllamaEmbeddings(model="mistral")
URI = "./milvus_example.db"

def get_vector_store(embeddings):
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )
    return vector_store


if __name__ == "__main__":
    file_paths = ["doc_1.pdf", "doc_2.pdf"]
    all_docs = get_doc_loader(file_paths)
    all_texts = split_text(all_docs)
    vector_store = get_vector_store(embeddings)
    vector_store.add_texts(texts=all_texts)
    results = vector_store.similarity_search(
        "What are AON values?"
    )

    print(results[0])
    




    