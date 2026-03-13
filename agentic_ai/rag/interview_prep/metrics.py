from ragas.metrics import DiscreteMetric

context_relevance_metric = DiscreteMetric(
    name="context_relevance",
    prompt="""You are evaluating retrieved documents for a RAG pipeline.

Question: {question}
Retrieved context: {context}

Do the retrieved documents contain information useful for answering the question?
Return 'pass' if the context is relevant, else 'fail'.""",
    allowed_values=["pass", "fail"],
)

reranker_metric = DiscreteMetric(
    name="reranker_effectiveness",
    prompt="""You are evaluating whether a cross-encoder reranker placed the best document first.

Question: {question}
Top-ranked document (after reranking): {top_doc}
All retrieved documents: {all_docs}

Is the top-ranked document the most relevant to the question compared to the others?
Return 'pass' if yes, else 'fail'.""",
    allowed_values=["pass", "fail"],
)

faithfulness_metric = DiscreteMetric(
    name="answer_faithfulness",
    prompt="""You are evaluating whether a generated answer is grounded in the provided context.

Question: {question}
Context used: {context}
Generated answer: {answer}

Does the answer only use information present in the context?
Return 'pass' if fully grounded, 'fail' if there is hallucination or invented content.""",
    allowed_values=["pass", "fail"],
)

completeness_metric = DiscreteMetric(
    name="answer_completeness",
    prompt="""You are evaluating whether a generated answer is complete.

Question: {question}
Ground truth key points: {ground_truth}
Generated answer: {answer}

Does the generated answer cover all key points from the ground truth?
Return 'pass' if all key points are addressed, else 'fail'.""",
    allowed_values=["pass", "fail"],
)

query_reformulation_metric = DiscreteMetric(
    name="query_reformulation_quality",
    prompt="""You are evaluating a query reformulation step in a RAG pipeline.

Original question: {question}
Reformulated query: {new_question}
Previously retrieved context: {context}

Is the reformulated query meaningfully different from the original and likely to retrieve better information?
Return 'pass' if useful, 'fail' if it is a repeat or unhelpful.""",
    allowed_values=["pass", "fail"],
)
