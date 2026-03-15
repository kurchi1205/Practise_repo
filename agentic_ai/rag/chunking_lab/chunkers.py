from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL


# ------------------------------------------------------------------
# Strategy: fixed_medium — 500 char chunks, 50 char overlap
# ------------------------------------------------------------------
def chunk_fixed_medium(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


# ------------------------------------------------------------------
# Strategy: semantic_medium — splits on meaning shifts using embeddings
# ------------------------------------------------------------------
def chunk_semantic_medium(text: str):
    splitter = SemanticChunker(
        embeddings=OllamaEmbeddings(model=EMBEDDING_MODEL),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    return splitter.split_text(text)


# ------------------------------------------------------------------
# Strategy: markdown — splits on markdown headers (#, ##, ###)
# ------------------------------------------------------------------
def chunk_markdown(text: str):
    headers_to_split_on = [
        ("#",   "h1"),
        ("##",  "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    md_docs = md_splitter.split_text(text)

    fallback = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    chunks = []
    for doc in md_docs:
        if len(doc.page_content) > 600:
            chunks.extend(fallback.split_text(doc.page_content))
        else:
            chunks.append(doc.page_content)
    return chunks


# ------------------------------------------------------------------
# Strategy: hierarchical — indexes small child chunks for precise
# matching but retrieves the larger parent chunk as context.
#
# How it works:
#   1. Split document into large parent chunks (1000 chars)
#   2. Split each parent into small child chunks (200 chars)
#   3. Embed and index child chunks in Milvus
#   4. Store parent content in each child's metadata
#   5. At retrieval time, child chunks match the query precisely,
#      then the node swaps in the parent content for generation
#
# Result: retrieval precision of a small chunk + context richness
# of a large chunk.
# ------------------------------------------------------------------

# Module-level map: child text -> parent text
# Populated when chunk_hierarchical() is called, read by nodes.py
_parent_map: dict = {}

def chunk_hierarchical(text: str):
    global _parent_map
    _parent_map = {}   # reset for each new run

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    child_splitter  = RecursiveCharacterTextSplitter(chunk_size=200,  chunk_overlap=20)

    parents = parent_splitter.split_text(text)
    child_chunks = []
    for parent in parents:
        children = child_splitter.split_text(parent)
        for child in children:
            _parent_map[child] = parent
            child_chunks.append(child)

    return child_chunks


# Registry
CHUNKING_STRATEGIES = {
    "fixed_medium":    chunk_fixed_medium,
    "semantic_medium": chunk_semantic_medium,
    "markdown":        chunk_markdown,
    "hierarchical":    chunk_hierarchical,
}
