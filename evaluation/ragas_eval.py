import os
import sys
import json
from anthropic import Anthropic

# -- Resolve project root ------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "retrieval"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "generation"))
# ------------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from retriever import load_retriever
from generator import ask

client = Anthropic()
MODEL = "claude-sonnet-4-6"


def evaluate_faithfulness(question, answer, contexts):
    """
    Faithfulness: Are all claims in the answer supported
    by the retrieved context? Score 0.0 to 1.0.
    """
    context_text = "\n\n---\n\n".join(contexts)

    prompt = f"""You are an evaluation assistant. Your job is to assess whether an answer is faithful to the provided context.

Context:
{context_text}

Question: {question}
Answer: {answer}

Instructions:
1. Break the answer into individual factual claims
2. For each claim, check whether it is supported by the context
3. Calculate: supported claims / total claims
4. Return ONLY a JSON object in this exact format:
{{"score": 0.85, "supported": 7, "total": 8, "reason": "brief explanation"}}

Return ONLY the JSON, no other text."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        return float(result["score"])
    except Exception as e:
        print(f"    Warning: faithfulness parse error: {e}")
        return 0.0


def evaluate_answer_relevancy(question, answer):
    """
    Answer Relevancy: Does the answer actually address
    the question asked? Score 0.0 to 1.0.
    """
    prompt = f"""You are an evaluation assistant. Your job is to assess whether an answer is relevant to the question asked.

Question: {question}
Answer: {answer}

Instructions:
1. Assess how directly and completely the answer addresses the question
2. A score of 1.0 means the answer perfectly addresses the question
3. A score of 0.0 means the answer is completely irrelevant
4. Return ONLY a JSON object in this exact format:
{{"score": 0.85, "reason": "brief explanation"}}

Return ONLY the JSON, no other text."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        return float(result["score"])
    except Exception as e:
        print(f"    Warning: answer relevancy parse error: {e}")
        return 0.0


def evaluate_context_precision(question, contexts, ground_truth):
    """
    Context Precision: Are the retrieved chunks actually
    relevant to answering the question? Score 0.0 to 1.0.
    """
    scores = []
    for i, context in enumerate(contexts):
        prompt = f"""You are an evaluation assistant. Your job is to assess whether a retrieved context chunk is relevant for answering a question.

Question: {question}
Ground truth answer: {ground_truth}
Retrieved chunk: {context}

Instructions:
1. Assess whether this chunk contains information useful for answering the question
2. Return ONLY a JSON object in this exact format:
{{"relevant": true, "reason": "brief explanation"}}

Return ONLY the JSON, no other text."""

        response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text.strip())
            scores.append(1.0 if result["relevant"] else 0.0)
        except Exception as e:
            print(f"    Warning: context precision parse error: {e}")
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_context_recall(question, contexts, ground_truth):
    """
    Context Recall: Does the retrieved context contain
    all the information needed to answer correctly?
    Score 0.0 to 1.0.
    """
    context_text = "\n\n---\n\n".join(contexts)

    prompt = f"""You are an evaluation assistant. Your job is to assess whether the retrieved context contains enough information to produce the ground truth answer.

Question: {question}
Ground truth answer: {ground_truth}
Retrieved context: {context_text}

Instructions:
1. Break the ground truth answer into individual pieces of information
2. For each piece, check whether it can be found in the retrieved context
3. Calculate: found pieces / total pieces
4. Return ONLY a JSON object in this exact format:
{{"score": 0.85, "found": 7, "total": 8, "reason": "brief explanation"}}

Return ONLY the JSON, no other text."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        return float(result["score"])
    except Exception as e:
        print(f"    Warning: context recall parse error: {e}")
        return 0.0


def run_evaluation():
    """
    Run full evaluation across all questions in the eval set.
    Uses Claude directly as the judge -- no RAGAS dependency.
    """
    print("=" * 60)
    print("Luminary RAG Assistant -- RAG Evaluation")
    print("=" * 60)

    # Load retriever
    print("\nLoading retriever...")
    retriever = load_retriever(source_filter="luminary")

    # Load eval set
    eval_path = os.path.join(PROJECT_ROOT, "evaluation", "eval_set.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Running evaluation on {len(eval_set)} questions...")

    # Store scores
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []

    results_detail = []

    for i, item in enumerate(eval_set, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n[{i}/{len(eval_set)}] {question[:60]}...")

        # Get RAG response
        result = ask(question, retriever)
        answer = result["answer"]

        # Get retrieved chunks
        chunks = retriever.invoke(question)
        context_texts = [chunk.page_content for chunk in chunks]

        # Run four evaluations
        print(f"  Evaluating faithfulness...")
        f_score = evaluate_faithfulness(question, answer, context_texts)

        print(f"  Evaluating answer relevancy...")
        r_score = evaluate_answer_relevancy(question, answer)

        print(f"  Evaluating context precision...")
        p_score = evaluate_context_precision(question, context_texts, ground_truth)

        print(f"  Evaluating context recall...")
        c_score = evaluate_context_recall(question, context_texts, ground_truth)

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        precision_scores.append(p_score)
        recall_scores.append(c_score)

        print(f"  Scores -- F:{f_score:.2f} R:{r_score:.2f} P:{p_score:.2f} C:{c_score:.2f}")

        results_detail.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "faithfulness": round(f_score, 3),
            "answer_relevancy": round(r_score, 3),
            "context_precision": round(p_score, 3),
            "context_recall": round(c_score, 3)
        })

    # Calculate averages
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Faithfulness      : {avg_faithfulness:.3f}")
    print(f"  Answer Relevancy  : {avg_relevancy:.3f}")
    print(f"  Context Precision : {avg_precision:.3f}")
    print(f"  Context Recall    : {avg_recall:.3f}")
    print("=" * 60)
    print("\nScore guide:")
    print("  > 0.85 = Excellent")
    print("  > 0.70 = Good")
    print("  > 0.50 = Needs improvement")
    print("  < 0.50 = Poor -- revisit chunking or retrieval")

    # Save results
    results_path = os.path.join(PROJECT_ROOT, "evaluation", "ragas_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "summary": {
                "faithfulness": round(avg_faithfulness, 3),
                "answer_relevancy": round(avg_relevancy, 3),
                "context_precision": round(avg_precision, 3),
                "context_recall": round(avg_recall, 3)
            },
            "detail": results_detail
        }, f, indent=2)

    print(f"\nResults saved to evaluation/ragas_results.json")


if __name__ == "__main__":
    run_evaluation()