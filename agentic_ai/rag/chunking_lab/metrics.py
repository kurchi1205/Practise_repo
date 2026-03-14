from ragas.metrics import numeric_metric, discrete_metric
from langchain_core.messages import HumanMessage


@numeric_metric(name="context_relevance", allowed_values=(0.0, 1.0))
def context_relevance_metric(question: str, context: str, llm) -> float:
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


@discrete_metric(name="answer_faithfulness", allowed_values=["pass", "fail"])
def faithfulness_metric(question: str, context: str, answer: str, llm) -> str:
    prompt = f"""You are evaluating whether a generated answer is grounded in the provided context.

Question: {question}
Context: {context}
Answer: {answer}

Does the answer only use information present in the context?
Return only 'pass' if fully grounded, 'fail' if there is hallucination or invented content."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return "pass" if "pass" in response.content.strip().lower() else "fail"



@numeric_metric(name="answer_relevance", allowed_values=(0.0, 1.0))
def answer_relevance_metric(question: str, answer: str, llm) -> float:
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
