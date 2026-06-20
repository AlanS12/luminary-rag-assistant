import os
from langchain_core.documents import Document


# ── Configuration ──────────────────────────────────────────────────────────────
# ── Resolve paths relative to project root ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GDS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "gds_scraped")
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "synthetic")
# GDS_DIR = "luminary-rag-assistant/data/raw/gds_scraped"
# SYNTHETIC_DIR = "luminary-rag-assistant/data/raw/synthetic"
# ───────────────────────────────────────────────────────────────────────────────


def load_documents_from_directory(directory, source_tag):
    """
    Load all .md files from a directory using plain Python file reading.
    Creates LangChain Document objects with metadata.
    """
    documents = []

    for filename in os.listdir(directory):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(directory, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                continue

            doc = Document(
                page_content=content,
                metadata={
                    "source": filepath,
                    "filename": filename,
                    "source_type": source_tag
                }
            )
            documents.append(doc)

        except Exception as e:
            print(f"  Warning: could not read {filename}: {e}")

    print(f"  Loaded {len(documents)} documents from {directory}")
    return documents


def load_all_documents():
    """
    Load all documents from both corpus folders.
    Returns a single combined list of Document objects.
    """
    print("=" * 60)
    print("Luminary RAG Assistant — Document Loader")
    print("=" * 60)

    all_documents = []

    print("\nLoading GDS blog posts...")
    gds_docs = load_documents_from_directory(GDS_DIR, "gds_blog")
    all_documents.extend(gds_docs)

    print("\nLoading synthetic Luminary documents...")
    synthetic_docs = load_documents_from_directory(SYNTHETIC_DIR, "luminary_synthetic")
    all_documents.extend(synthetic_docs)

    print(f"\nTotal documents loaded : {len(all_documents)}")
    print(f"  GDS blog posts       : {len(gds_docs)}")
    print(f"  Synthetic docs       : {len(synthetic_docs)}")

    return all_documents


if __name__ == "__main__":
    docs = load_all_documents()

    print("\n--- Sample of first 3 documents ---")
    for doc in docs[:3]:
        print(f"\nFile        : {doc.metadata.get('filename')}")
        print(f"Source type : {doc.metadata.get('source_type')}")
        print(f"Length      : {len(doc.page_content)} characters")
        print(f"Preview     : {doc.page_content[:200]}...")