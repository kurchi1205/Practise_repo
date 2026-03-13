from typing import TypedDict, List, Dict, Any

class question_answer(TypedDict):
    question: str
    is_relevant: bool
    retrieved_documents: List
    pre_rerank_documents: List
    document_relevance: bool
    web_searched_content: str
    generated_answer: str
    new_question: str
    past_questions: str
    retrieval_count: int
    ground_truth: str
    expected_route: str
    scores: Dict[str, Any]


