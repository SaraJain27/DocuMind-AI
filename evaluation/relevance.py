
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from retrieval.semantic_search import search_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

judge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are evaluating document retrieval relevance.

Score how relevant the retrieved passage is to the question:
5 = Directly answers the question.
4 = Highly relevant and contains useful evidence.
3 = Partially relevant.
2 = Weakly relevant.
1 = Not relevant.

Return only valid JSON in this format:
{{"score": 1, "reason": "Brief explanation"}}

Do not judge whether the passage is factually correct.
Judge only its relevance to the question."""
    ),
    (
        "human",
        """Question:
{question}

Retrieved passage:
{passage}"""
    ),
])


def evaluate_relevance(
    question: str,
    document_id: str,
    top_k: int = 3,
) -> dict:
    """Retrieve document passages and evaluate their relevance."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    results = search_documents(
        question,
        top_k=top_k,
        document_id=document_id,
    )

    if not results:
        return {
            "question": question,
            "evaluated_chunks": 0,
            "average_relevance": 0.0,
            "results": [],
        }

    evaluated = []

    for result in results:
        messages = judge_prompt.format_messages(
            question=question,
            passage=result["text"],
        )

        response = llm.invoke(messages)

        try:
            judgment = json.loads(response.content)
            score = int(judgment["score"])

            if score < 1 or score > 5:
                raise ValueError("Score must be between 1 and 5.")

            evaluated.append({
                "page_number": result["page_number"],
                "passage": result["text"],
                "similarity": round(result["similarity"], 3),
                "relevance_score": score,
                "reason": judgment.get("reason", ""),
            })

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            evaluated.append({
                "page_number": result["page_number"],
                "passage": result["text"],
                "similarity": round(result["similarity"], 3),
                "relevance_score": None,
                "reason": f"Could not parse evaluator response: {error}",
            })

    valid_scores = [
        item["relevance_score"]
        for item in evaluated
        if item["relevance_score"] is not None
    ]

    average = (
        sum(valid_scores) / len(valid_scores)
        if valid_scores
        else None
    )

    return {
        "question": question,
        "evaluated_chunks": len(evaluated),
        "average_relevance": (
            round(average, 2) if average is not None else None
        ),
        "results": evaluated,
    }


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing from your .env file.")

    document_id = input("Enter the document ID: ").strip()
    question = input("Enter a question about that document: ").strip()

    try:
        report = evaluate_relevance(question, document_id)

        print(json.dumps(report, indent=2, ensure_ascii=False))

    except Exception as error:
        print(f"Evaluation failed: {error}")
