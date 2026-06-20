import os
import sys

# -- Resolve project root ------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "embeddings"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "vectorstore"))
# ------------------------------------------------------------------------------

# -- Configuration -------------------------------------------------------------
TOP_K = 8   # number of chunks to retrieve per query
# ------------------------------------------------------------------------------

from embedder import load_embedding_model
from chroma_store import load_vector_store


def load_retriever(source_filter=None):
    """
    Load the Chroma vector store and return a retriever object.
    The retriever takes a query string and returns the top-k
    most semantically similar chunks.

    Load retriever with optional metadata filtering.
    source_filter options:
        None           -- search all documents
        'luminary'     -- search only synthetic docs
        'gds'          -- search only GDS blog posts
    """
    vector_store = load_vector_store()

    if source_filter == "luminary":
        search_kwargs = {
            "k": 8,
            "filter": {"source_type": "luminary_synthetic"}
        }
    elif source_filter == "gds":
        search_kwargs = {
            "k": 8,
            "filter": {"source_type": "gds_blog"}
        }
    else:
        search_kwargs = {"k": 8}

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    print(f"Retriever ready. Filter: {source_filter or 'none'}. k=8")
    return retriever


if __name__ == "__main__":
    print("=" * 60)
    print("Luminary RAG Assistant -- Retriever Test")
    print("=" * 60)

    retriever = load_retriever()

    test_query = "How did Luminary help BrightMart with their data platform?"
    print(f"\nTest query: {test_query}")

    results = retriever.invoke(test_query)

    print(f"\nRetrieved {len(results)} chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\nChunk {i}:")
        print(f"  File    : {doc.metadata.get('filename', 'unknown')}")
        print(f"  Type    : {doc.metadata.get('source_type', 'unknown')}")
        print(f"  Preview : {doc.page_content[:200]}...")