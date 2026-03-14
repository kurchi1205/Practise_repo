from collections import defaultdict
from langchain_community.document_loaders import PyPDFLoader
from chunkers import CHUNKING_STRATEGIES
from graph import graph
from config import DOCUMENT_PATHS, TEST_QUESTIONS, STRATEGIES_TO_RUN


# ------------------------------------------------------------------
# Step 1 — Load raw text from PDFs
# ------------------------------------------------------------------
def load_documents() -> str:
    all_text = []
    for path in DOCUMENT_PATHS:
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_text.append("\n".join(d.page_content for d in docs))
            print(f"Loaded: {path}  ({len(docs)} pages)")
        except Exception as e:
            print(f"Could not load {path}: {e}")
    return "\n\n".join(all_text)


# ------------------------------------------------------------------
# Step 2 — Run one (strategy, question) pair through the LangGraph
# ------------------------------------------------------------------
def run_single(strategy_name: str, question: str, raw_text: str) -> dict:
    chunk_fn = CHUNKING_STRATEGIES[strategy_name]
    chunks = chunk_fn(raw_text)

    initial_state = {
        "strategy_name":   strategy_name,
        "question":        question,
        "chunks":          chunks,
        "num_chunks":      0,
        "avg_chunk_length": 0.0,
        "retrieved_docs":  [],
        "generated_answer": "",
        "scores":          {},
    }
    return graph.invoke(initial_state)


# ------------------------------------------------------------------
# Step 3 — Loop over all strategies × questions
# ------------------------------------------------------------------
def run_experiment():
    print("Loading documents...")
    raw_text = load_documents()
    print(f"Total text length: {len(raw_text)} characters\n")

    all_results = []

    for strategy in STRATEGIES_TO_RUN:
        for question in TEST_QUESTIONS:
            print(f"\n--- strategy={strategy} | question={question[:50]}... ---")
            try:
                final_state = run_single(strategy, question, raw_text)
                all_results.append({
                    "strategy":          strategy,
                    "question":          question[:45] + "...",
                    "num_chunks":        final_state["num_chunks"],
                    "avg_chunk_len":     final_state["avg_chunk_length"],
                    **final_state["scores"],
                })
            except Exception as e:
                print(f"  ERROR: {e}")

    print_comparison_table(all_results)
    return all_results


# ------------------------------------------------------------------
# Step 4 — Print comparison table averaged across questions
# ------------------------------------------------------------------
def print_comparison_table(results: list):
    if not results:
        print("No results to display.")
        return

    # Collect all metric keys dynamically (handles different metric sets)
    metric_keys = [k for k in results[0] if k not in ("strategy", "question", "num_chunks", "avg_chunk_len")]

    print("\n" + "=" * 90)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 90)

    col_w = 16
    header = f"{'Strategy':<{col_w}} | {'#Chunks':>7} | {'AvgLen':>7}"
    for m in metric_keys:
        header += f" | {m[:10]:>10}"
    print(header)
    print("-" * 90)

    grouped = defaultdict(list)
    for r in results:
        grouped[r["strategy"]].append(r)

    for strategy, rows in grouped.items():
        def avg(key):
            vals = [r[key] for r in rows if isinstance(r.get(key), float)]
            return f"{sum(vals)/len(vals):.3f}" if vals else "N/A"

        row = (
            f"{strategy:<{col_w}} | "
            f"{rows[0]['num_chunks']:>7} | "
            f"{rows[0]['avg_chunk_len']:>7}"
        )
        for m in metric_keys:
            row += f" | {avg(m):>10}"
        print(row)

    print("=" * 90)
    print("Scores averaged across test questions.")
    print("faithfulness, answer_relevancy, context_precision: 0.0-1.0\n")


if __name__ == "__main__":
    run_experiment()
