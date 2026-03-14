from langchain_text_splitters import RecursiveCharacterTextSplitter


# ------------------------------------------------------------------
# Strategy: fixed_medium — 500 char chunks, 50 char overlap
# Good baseline: captures enough context without being too broad
# ------------------------------------------------------------------
def chunk_fixed_medium(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


# Registry — experiment.py imports this dict
# Add new strategies here as you implement them
CHUNKING_STRATEGIES = {
    "fixed_medium": chunk_fixed_medium,
}
