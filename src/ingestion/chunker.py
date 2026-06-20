from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_all_documents

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 700        # maximum characters per chunk
CHUNK_OVERLAP = 150      # overlap between consecutive chunks
# ───────────────────────────────────────────────────────────────────────────────


def chunk_documents(documents):
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    This splitter tries to split on paragraph breaks first (\n\n),
    then single newlines (\n), then sentences, then words.
    This means it tries to keep natural language units together
    rather than cutting mid-sentence.

    CHUNK_SIZE = 512  : each chunk is at most 512 characters
    CHUNK_OVERLAP = 50: consecutive chunks share 50 characters
                        of overlap so context is not lost at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Carry forward all metadata from parent document into each chunk
    print(f"  Original documents : {len(documents)}")
    print(f"  Chunks produced    : {len(chunks)}")
    print(f"  Avg chunks per doc : {len(chunks) / len(documents):.1f}")

    return chunks


def inspect_chunks(chunks, n=5):
    """
    Print a sample of chunks so you can visually verify
    the chunking is working sensibly.
    """
    print(f"\n--- Sample of first {n} chunks ---")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk {i + 1}")
        print(f"  Source     : {chunk.metadata.get('filename', 'unknown')}")
        print(f"  Source type: {chunk.metadata.get('source_type', 'unknown')}")
        print(f"  Length     : {len(chunk.page_content)} characters")
        print(f"  Content    : {chunk.page_content[:300]}...")
        print("-" * 40)


if __name__ == "__main__":
    print("=" * 60)
    print("Luminary RAG Assistant — Document Chunker")
    print("=" * 60)

    # Load all documents
    print("\nLoading documents...")
    documents = load_all_documents()

    # Chunk them
    print("\nChunking documents...")
    chunks = chunk_documents(documents)

    # Inspect a sample
    inspect_chunks(chunks, n=5)

    print(f"\nChunking complete. {len(chunks)} chunks ready for embedding.")