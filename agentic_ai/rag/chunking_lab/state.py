from typing import TypedDict, List, Dict, Any


class ChunkingExperimentState(TypedDict):
    # --- Inputs ---
    strategy_name: str          # e.g. "fixed_medium"
    question: str
    chunks: List[str]           # raw text chunks produced by chunking strategy

    # --- Set by ingest node ---
    num_chunks: int
    avg_chunk_length: float

    # --- Set by retrieve node ---
    retrieved_docs: List        # List[Document]

    # --- Set by generate node ---
    generated_answer: str

    # --- Set by evaluate node ---
    scores: Dict[str, Any]      # metric_name -> value
