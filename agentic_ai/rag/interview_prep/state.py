from typing import TypedDict, List

class question_answer(TypedDict):
    question: str
    is_relevant: bool
    retrieved_documents: List
    document_relevance: bool
    web_searched_content: str
    generated_answer: str


