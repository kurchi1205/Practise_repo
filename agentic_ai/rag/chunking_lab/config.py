LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "mistral"

MILVUS_URI = "./chunking_lab.db"

DOCUMENT_PATHS = [
    "../interview_prep/doc_1.pdf",
    "../interview_prep/doc_2.pdf",
]

TEST_QUESTIONS = [
    "What are the AON company values?",
    "How should I prepare for an AON interview?",
    "What competencies does AON look for in candidates?",
]

RETRIEVAL_K = 2

# Add more strategy names here as they are implemented in chunkers.py
STRATEGIES_TO_RUN = [
    "fixed_medium",
    "semantic_medium",
    "markdown",
    "hierarchical",
]
