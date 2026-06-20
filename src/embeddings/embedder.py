import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings

# -- Resolve project root ------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ingestion"))
# ------------------------------------------------------------------------------

# -- Configuration -------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
# ------------------------------------------------------------------------------


def load_embedding_model():
    """
    Load the BAAI/bge-large-en-v1.5 embedding model from HuggingFace.
    This runs entirely locally -- no API calls, no cost.
    First run will download the model (~1.3GB).
    Subsequent runs load from local cache instantly.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print("(First run will download ~1.3GB -- this is normal)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Embedding model loaded successfully.")
    return embeddings


if __name__ == "__main__":
    # Quick test -- embed a sample sentence and print the vector shape
    model = load_embedding_model()

    test_sentences = [
        "What data platform does Luminary recommend for retail clients?",
        "How does predictive maintenance reduce machine downtime?",
        "What is a medallion architecture in Databricks?"
    ]

    print("\nTesting embeddings on sample sentences...")
    for sentence in test_sentences:
        vector = model.embed_query(sentence)
        print(f"\nSentence : {sentence}")
        print(f"Vector dimensions : {len(vector)}")
        print(f"First 5 values    : {[round(v, 4) for v in vector[:5]]}")

    print("\nEmbedding model working correctly.")