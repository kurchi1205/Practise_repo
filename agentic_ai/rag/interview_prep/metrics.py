from ragas.metrics import numeric_metric, discrete_metric
from langchain_core.messages import HumanMessage

@discrete_metric(name="routing_accuracy", allowed_values=["pass", "fail"])
def routing_accuracy_metric(expected_route: str, actual_route: str) -> str:
    return "pass" if expected_route.strip().lower() == actual_route.strip().lower() else "fail"

@numeric_metric(name="context_relevance", allowed_values=(0.0, 1.0))
def context_relevance_metric(question: str, context: str, llm=None) -> float:
    prompt = f"""Score how relevant the retrieved context is for answering the question.
- 1.0: fully covers the question with high detail
- 0.7: mostly covers the question with minor gaps
- 0.4: partially relevant but missing key information
- 0.1: barely related to the question
- 0.0: completely irrelevant

Question: {question}
Context: {context}

Return only a float between 0.0 and 1.0. No explanation."""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        return float(response.content.strip())
    except ValueError:
        return 0.0

# 3.1 — Is the top doc after reranking the most relevant? (LLM judge)
@discrete_metric(name="reranker_effectiveness", allowed_values=["pass", "fail"])
def reranker_effectiveness_metric(question: str, top_doc: str, all_docs: str, llm=None) -> str:
    prompt = f"""You are evaluating whether a cross-encoder reranker placed the best document first.

Question: {question}
Top-ranked document (after reranking): {top_doc}
All retrieved documents: {all_docs}

Is the top-ranked document the most relevant to the question compared to the others?
Return only 'pass' or 'fail'."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return "pass" if "pass" in response.content.strip().lower() else "fail"


# 3.2 — Did the most relevant doc move up after reranking? (deterministic)
@numeric_metric(name="rank_improvement", allowed_values=(0.0, 1.0))
def rank_improvement_metric(pre_rerank_docs: list, post_rerank_docs: list) -> float:
    if not pre_rerank_docs or not post_rerank_docs or len(pre_rerank_docs) == 1:
        return 1.0
    top_doc_content = post_rerank_docs[0].page_content
    pre_rank = next(
        (i for i, d in enumerate(pre_rerank_docs) if d.page_content == top_doc_content),
        len(pre_rerank_docs) - 1
    )
    return pre_rank / (len(pre_rerank_docs) - 1)

# 7.1 — Is the answer grounded in context with no hallucination?
@discrete_metric(name="answer_faithfulness", allowed_values=["pass", "fail"])
def faithfulness_metric(question: str, context: str, answer: str, llm=None) -> str:
    prompt = f"""You are evaluating whether a generated answer is grounded in the provided context.

Question: {question}
Context: {context}
Answer: {answer}

Does the answer only use information present in the context?
Return only 'pass' if fully grounded, 'fail' if there is hallucination or invented content."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return "pass" if "pass" in response.content.strip().lower() else "fail"


# 7.2 — Does the answer cover all key points from ground truth?
@discrete_metric(name="answer_completeness", allowed_values=["pass", "fail"])
def completeness_metric(question: str, ground_truth: str, answer: str, llm=None) -> str:
    prompt = f"""You are evaluating whether a generated answer is complete.

Question: {question}
Ground truth key points: {ground_truth}
Generated answer: {answer}

Does the generated answer cover all key points from the ground truth?
Return only 'pass' if all key points are addressed, else 'fail'."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return "pass" if "pass" in response.content.strip().lower() else "fail"


# 7.3 — How well does the answer address the question?
@numeric_metric(name="answer_relevance", allowed_values=(0.0, 1.0))
def answer_relevance_metric(question: str, answer: str, llm=None) -> float:
    prompt = f"""Score how well the answer addresses the question asked.
- 1.0: directly and fully answers the question
- 0.7: mostly answers the question with minor off-topic content
- 0.4: partially answers but drifts significantly
- 0.1: barely addresses the question
- 0.0: completely irrelevant to the question

Question: {question}
Answer: {answer}

Return only a float between 0.0 and 1.0. No explanation."""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        return float(response.content.strip())
    except ValueError:
        return 0.0

