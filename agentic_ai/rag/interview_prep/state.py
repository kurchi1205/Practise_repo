from typing import TypedDict, List, Dict, Any

class question_answer(TypedDict):
    question: str
    generated_answer: str
    retrieved_documents: List
    pre_rerank_documents: List
    web_searched_content: str
    ground_truth: str
    expected_route: str
    scores: Dict[str, Any]
