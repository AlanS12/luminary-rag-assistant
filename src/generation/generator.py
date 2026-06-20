import os
import sys
from anthropic import Anthropic

# -- Resolve project root ------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "retrieval"))
# ------------------------------------------------------------------------------

# -- Configuration -------------------------------------------------------------
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
# ------------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from retriever import load_retriever

# Load Anthropic client
client = Anthropic()


def format_context(retrieved_chunks):
    """
    Format retrieved chunks into a single context string.
    Each chunk is labelled with its source file so Claude
    can cite where information came from.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        filename = chunk.metadata.get("filename", "unknown")
        source_type = chunk.metadata.get("source_type", "unknown")
        context_parts.append(
            f"[Source {i}: {filename} ({source_type})]\n{chunk.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


def build_prompt(question, context):
    """
    Build the full prompt that gets sent to Claude.
    This is the prompt template -- the most important
    piece of prompt engineering in a RAG system.
    """
    return f"""You are a knowledge assistant for Luminary Data & AI.

STRICT INSTRUCTIONS -- READ CAREFULLY:
1. You MUST answer ONLY using information explicitly stated in the context below
2. You MUST NOT use any knowledge from your training data
3. If the context does not contain the answer, say exactly: "I cannot find this information in the Luminary knowledge base."
4. Every claim you make MUST be traceable to a specific part of the context
5. Do NOT infer, extrapolate, or add information that is not explicitly in the context
6. Keep your answer concise and directly supported by the context

Context:
{context}

Question: {question}

Answer (using ONLY the context above):"""


def ask(question, retriever):
    """
    Full RAG pipeline for a single question:
    1. Retrieve relevant chunks
    2. Format context
    3. Build prompt
    4. Call Claude API
    5. Return answer with sources
    """
    print(f"\nQuestion: {question}")
    print("Retrieving relevant chunks...")

    # Step 1 -- Retrieve
    # BGE models perform better with this prefix on queries
    prefixed_question = f"Represent this sentence for searching relevant passages: {question}"
    chunks = retriever.invoke(prefixed_question)
    print(f"Retrieved {len(chunks)} chunks from vector store.")

    # Step 2 -- Format context
    context = format_context(chunks)

    # Step 3 -- Build prompt
    prompt = build_prompt(question, context)

    # Step 4 -- Call Claude API
    print("Calling Claude API...")
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.content[0].text

    # Step 5 -- Return answer with sources
    sources = [c.metadata.get("filename", "unknown") for c in chunks]

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Luminary RAG Assistant -- Generation Test")
    print("=" * 60)

    retriever = load_retriever()

    # Test questions -- mix of synthetic doc questions and GDS questions
    test_questions = [
        "What were the main data problems at BrightMart and how did Luminary solve them?",
        "What is a medallion architecture and why does Luminary use it?",
        "How did the Hartwell RAG assistant achieve a 58% deflection rate?",
        "What lessons did Luminary learn from the Shopify integration on the BrightMart project?"
    ]

    for question in test_questions:
        result = ask(question, retriever)
        print(f"\n{'=' * 60}")
        print(f"Q: {result['question']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources used: {result['sources']}")
        print("=" * 60)