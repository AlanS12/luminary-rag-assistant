import os
import sys
from langchain_chroma import Chroma

# -- Resolve project root ------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ingestion"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "embeddings"))
# ------------------------------------------------------------------------------

# -- Configuration -------------------------------------------------------------
CHROMA_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")
COLLECTION_NAME = "luminary_knowledge"
# ------------------------------------------------------------------------------

from loader import load_all_documents
from chunker import chunk_documents
from embedder import load_embedding_model


def build_vector_store():
    """
    Full pipeline:
    1. Load all documents
    2. Chunk them
    3. Embed each chunk
    4. Store in Chroma on disk
    """
    print("=" * 60)
    print("Luminary RAG Assistant -- Building Vector Store")
    print("=" * 60)

    # Step 1 -- Load
    print("\nStep 1: Loading documents...")
    documents = load_all_documents()

    # Step 2 -- Chunk
    print("\nStep 2: Chunking documents...")
    chunks = chunk_documents(documents)

    # Step 3 -- Load embedding model
    print("\nStep 3: Loading embedding model...")
    embeddings = load_embedding_model()

    # Step 4 -- Build and persist Chroma vector store
    print(f"\nStep 4: Embedding {len(chunks)} chunks and storing in Chroma...")
    print("This will take a few minutes -- each chunk is being embedded...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    print(f"\nVector store built successfully.")
    try:
        count = vector_store._collection.count()
    except Exception:
        count = len(chunks)
    print(f"  Chunks stored : {count}")
    print(f"  Location      : {CHROMA_DIR}")

    return vector_store


def load_vector_store():
    """
    Load an existing Chroma vector store from disk.
    Use this after the vector store has been built once --
    no need to rebuild every time.
    """
    print("Loading existing vector store from disk...")
    embeddings = load_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    try:
            count = vector_store._collection.count()
    except Exception:
        count = "unknown"
    print(f"Vector store loaded. {count} chunks available.")
    
    return vector_store


def test_retrieval(vector_store):
    """
    Run a few test queries against the vector store
    to verify retrieval is working correctly.
    """
    test_queries = [
        "What data platform does Luminary recommend for retail clients?",
        "How does predictive maintenance reduce machine downtime?",
        "What is a medallion architecture?",
        "How did BrightMart reduce their reporting time?",
        "What is RAG and how does it work?"
    ]

    print("\n--- Testing Retrieval ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"  Result {i}:")
            print(f"    Source : {doc.metadata.get('filename', 'unknown')}")
            print(f"    Type   : {doc.metadata.get('source_type', 'unknown')}")
            print(f"    Preview: {doc.page_content[:150]}...")


if __name__ == "__main__":
    # Check if vector store already exists
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Existing vector store found.")
        print("To rebuild it, delete the data/chroma_db folder and rerun.")
        vector_store = load_vector_store()
    else:
        vector_store = build_vector_store()

    # Test retrieval
    test_retrieval(vector_store)